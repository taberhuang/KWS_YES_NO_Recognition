# Copyright 2022 The Pigweed Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Creates a new Pigweed module."""

from __future__ import annotations

import abc
import argparse
import dataclasses
from dataclasses import dataclass
import datetime
import difflib
from enum import Enum
import functools
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Collection, Iterable, Type

from prompt_toolkit import prompt

from pw_build import generate_modules_lists
import pw_cli.color
import pw_cli.env
from pw_cli.diff import colorize_diff
from pw_cli.status_reporter import StatusReporter

from pw_module.templates import get_template

_COLOR = pw_cli.color.colors()
_LOG = logging.getLogger(__name__)
_PW_ENV = pw_cli.env.pigweed_environment()
_PW_ROOT = _PW_ENV.PW_ROOT

_PIGWEED_LICENSE = f"""
# Copyright {datetime.datetime.now().year} The Pigweed Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.""".lstrip()

_PIGWEED_LICENSE_CC = _PIGWEED_LICENSE.replace('#', '//')

_CREATE = _COLOR.green('create    ')
_REPLACE = _COLOR.green('replace   ')
_UPDATE = _COLOR.yellow('update    ')
_UNCHANGED = _COLOR.blue('unchanged ')
_IDENTICAL = _COLOR.blue('identical ')
_REPORT = StatusReporter()


def _report_write_file(file_path: Path) -> None:
    """Print a notification when a file is newly created or replaced."""
    relative_file_path = str(file_path.relative_to(_PW_ROOT))
    if file_path.is_file():
        _REPORT.info(_REPLACE + relative_file_path)
        return
    _REPORT.new(_CREATE + relative_file_path)


def _report_unchanged_file(file_path: Path) -> None:
    """Print a notification a file was not updated/changed."""
    relative_file_path = str(file_path.relative_to(_PW_ROOT))
    _REPORT.ok(_UNCHANGED + relative_file_path)


def _report_identical_file(file_path: Path) -> None:
    """Print a notification a file is identical."""
    relative_file_path = str(file_path.relative_to(_PW_ROOT))
    _REPORT.ok(_IDENTICAL + relative_file_path)


def _report_edited_file(file_path: Path) -> None:
    """Print a notification a file was modified/edited."""
    relative_file_path = str(file_path.relative_to(_PW_ROOT))
    _REPORT.new(_UPDATE + relative_file_path)


class PromptChoice(Enum):
    """Possible prompt responses."""

    YES = 'yes'
    NO = 'no'
    DIFF = 'diff'


def _prompt_user(message: str, allow_diff: bool = False) -> PromptChoice:
    """Prompt the user for to choose between yes, no and optionally diff.

    If the user presses enter with no text the response is assumed to be NO.
    If the user presses ctrl-c call sys.exit(1).

    Args:
      message: The message to display at the start of the prompt.
      allow_diff: If true add a 'd' to the help text in the prompt line.

    Returns:
      A PromptChoice enum value.
    """
    help_text = '[y/N]'
    if allow_diff:
        help_text = '[y/N/d]'

    try:
        decision = prompt(f'{message} {help_text} ')
    except KeyboardInterrupt:
        sys.exit(1)  # Ctrl-C pressed

    if not decision or decision.lower().startswith('n'):
        return PromptChoice.NO
    if decision.lower().startswith('y'):
        return PromptChoice.YES
    if decision.lower().startswith('d'):
        return PromptChoice.DIFF

    return PromptChoice.NO


def _print_diff(file_name: Path | str, in_text: str, out_text: str) -> None:
    result_diff = list(
        difflib.unified_diff(
            in_text.splitlines(True),
            out_text.splitlines(True),
            f'{file_name}  (original)',
            f'{file_name}  (updated)',
        )
    )
    if not result_diff:
        return
    print()
    print(''.join(colorize_diff(result_diff)))


def _prompt_overwrite(file_path: Path, new_contents: str) -> bool:
    """Returns true if a file should be written, prompts the user if needed."""
    # File does not exist
    if not file_path.is_file():
        return True

    # File exists but is identical.
    old_contents = file_path.read_text(encoding='utf-8')
    if new_contents and old_contents == new_contents:
        _report_identical_file(file_path)
        return False

    file_name = file_path.relative_to(_PW_ROOT)
    # File exists and is different.
    _REPORT.wrn(f'{file_name} already exists.')

    while True:
        choice = _prompt_user('Overwrite?', allow_diff=True)
        if choice == PromptChoice.DIFF:
            _print_diff(file_name, old_contents, new_contents)
        else:
            if choice == PromptChoice.YES:
                return True
            break

    # By default do not overwrite.
    _report_unchanged_file(file_path)
    return False


# TODO(frolv): Adapted from pw_protobuf. Consolidate them.
class _OutputFile:
    DEFAULT_INDENT_WIDTH = 2

    def __init__(self, file: Path, indent_width: int = DEFAULT_INDENT_WIDTH):
        self._file = file
        self._content: list[str] = []
        self._indent_width: int = indent_width
        self._indentation = 0

    def line(self, line: str = '') -> None:
        if line:
            self._content.append(' ' * self._indentation)
            self._content.append(line)
        self._content.append('\n')

    def indent(
        self,
        width: int | None = None,
    ) -> _OutputFile._IndentationContext:
        """Increases the indentation level of the output."""
        return self._IndentationContext(
            self, width if width is not None else self._indent_width
        )

    @property
    def path(self) -> Path:
        return self._file

    @property
    def content(self) -> str:
        return ''.join(self._content)

    def write(self, content: str | None = None) -> None:
        """Write file contents. Prompts the user if necessary.

        Args:
          content: If provided will write this text to the file instead of
              calling self.content.
        """
        output_text = self.content
        if content:
            output_text = content

        if not output_text.endswith('\n'):
            output_text += '\n'

        if _prompt_overwrite(self._file, new_contents=output_text):
            _report_write_file(self._file)
            self._file.write_text(output_text)

    def write_template(self, template_name: str, **template_args) -> None:
        template = get_template(template_name)
        rendered_template = template.render(**template_args)
        self.write(content=rendered_template)

    class _IndentationContext:
        """Context that increases the output's indentation when it is active."""

        def __init__(self, output: _OutputFile, width: int):
            self._output = output
            self._width: int = width

        def __enter__(self):
            self._output._indentation += self._width

        def __exit__(self, typ, value, traceback):
            self._output._indentation -= self._width


class _ModuleName:
    _MODULE_NAME_REGEX = re.compile(
        # Match the two letter character module prefix e.g. 'pw':
        r'^(?P<prefix>[a-zA-Z]{2,})'
        # The rest of the module name consisting of multiple groups of a single
        # underscore followed by alphanumeric characters. This prevents multiple
        # underscores from appearing in a row and the name from ending in a an
        # underscore.
        r'(?P<main>'
        r'(_[a-zA-Z0-9]+)+'
        r')$'
    )

    def __init__(self, prefix: str, main: str, path: Path) -> None:
        self._prefix = prefix
        self._main = main.lstrip('_')  # Remove the leading underscore
        self._path = path

    @property
    def path(self) -> str:
        # Check if there are no parent directories for the full path.
        # Note: This relies on Path('pw_module').parents returning Path('.') for
        # paths that have no parent directories:
        # https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parents
        if self._path == Path('.'):
            return self.full
        return (self._path / self.full).as_posix()

    @property
    def full(self) -> str:
        return f'{self._prefix}_{self._main}'

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def main(self) -> str:
        return self._main

    @property
    def default_namespace(self) -> str:
        return f'{self._prefix}::{self._main}'

    def upper_camel_case(self) -> str:
        return ''.join(s.capitalize() for s in self._main.split('_'))

    @property
    def header_line(self) -> str:
        return '=' * len(self.full)

    def __str__(self) -> str:
        return self.full

    def __repr__(self) -> str:
        return self.full

    @classmethod
    def parse(cls, name: str) -> _ModuleName | None:
        module_path = Path(name)
        module_name = module_path.name
        match = _ModuleName._MODULE_NAME_REGEX.fullmatch(module_name)
        if not match:
            return None

        parts = match.groupdict()
        return cls(parts['prefix'], parts['main'], module_path.parents[0])


@dataclass
class _ModuleContext:
    name: _ModuleName
    dir: Path
    root_build_files: list[_BuildFile]
    sub_build_files: list[_BuildFile]
    build_systems: list[str]
    is_upstream: bool

    def build_files(self) -> Iterable[_BuildFile]:
        yield from self.root_build_files
        yield from self.sub_build_files

    def add_docs_file(self, file: Path):
        for build_file in self.root_build_files:
            build_file.add_docs_source(str(file.relative_to(self.dir)))

    def add_cc_target(self, target: _BuildFile.CcTarget) -> None:
        for build_file in self.root_build_files:
            build_file.add_cc_target(target)

    def add_cc_test(self, target: _BuildFile.CcTarget) -> None:
        for build_file in self.root_build_files:
            build_file.add_cc_test(target)


class _BuildFile:
    """Abstract representation of a build file for a module."""

    @dataclass
    class Target:
        name: str

        # TODO(frolv): Shouldn't be a string list as that's build system
        # specific. Figure out a way to resolve dependencies from targets.
        deps: list[str] = dataclasses.field(default_factory=list)

    @dataclass
    class CcTarget(Target):
        sources: list[Path] = dataclasses.field(default_factory=list)
        headers: list[Path] = dataclasses.field(default_factory=list)

        def rebased_sources(self, rebase_path: Path) -> Iterable[str]:
            return (str(src.relative_to(rebase_path)) for src in self.sources)

        def rebased_headers(self, rebase_path: Path) -> Iterable[str]:
            return (str(hdr.relative_to(rebase_path)) for hdr in self.headers)

    def __init__(self, path: Path, ctx: _ModuleContext):
        self._path = path
        self._ctx = ctx

        self._docs_sources: list[str] = []
        self._cc_targets: list[_BuildFile.CcTarget] = []
        self._cc_tests: list[_BuildFile.CcTarget] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dir(self) -> Path:
        return self._path.parent

    def add_docs_source(self, filename: str) -> None:
        self._docs_sources.append(filename)

    def add_cc_target(self, target: CcTarget) -> None:
        self._cc_targets.append(target)

    def add_cc_test(self, target: CcTarget) -> None:
        self._cc_tests.append(target)

    @property
    def get_license(self) -> str:
        if self._ctx.is_upstream:
            return _PIGWEED_LICENSE
        return ''

    @property
    def docs_sources(self) -> list[str]:
        return self._docs_sources

    @property
    def cc_targets(self) -> list[_BuildFile.CcTarget]:
        return self._cc_targets

    @property
    def cc_tests(self) -> list[_BuildFile.CcTarget]:
        return self._cc_tests

    def relative_file(self, file_path: Path | str) -> str:
        if isinstance(file_path, str):
            return file_path
        return str(file_path.relative_to(self._path.parent))

    def write(self) -> None:
        """Writes the contents of the build file to disk."""
        file = _OutputFile(self._path, self._indent_width())

        if self._ctx.is_upstream:
            file.line(_PIGWEED_LICENSE)
            file.line()

        self._write_preamble(file)

        for target in self._cc_targets:
            file.line()
            self._write_cc_target(file, target)

        for target in self._cc_tests:
            file.line()
            self._write_cc_test(file, target)

        if self._docs_sources:
            file.line()
            self._write_docs_target(file, self._docs_sources)

        file.write()

    @abc.abstractmethod
    def _indent_width(self) -> int:
        """Returns the default indent width for the build file's code style."""

    @abc.abstractmethod
    def _write_preamble(self, file: _OutputFile) -> None:
        """Formats"""

    @abc.abstractmethod
    def _write_cc_target(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        """Defines a C++ library target within the build file."""

    @abc.abstractmethod
    def _write_cc_test(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        """Defines a C++ unit test target within the build file."""

    @abc.abstractmethod
    def _write_docs_target(
        self,
        file: _OutputFile,
        docs_sources: list[str],
    ) -> None:
        """Defines a documentation target within the build file."""


# TODO(frolv): The dict here should be dict[str, '_GnVal'] (i.e. _GnScope),
# but mypy does not yet support recursive types:
# https://github.com/python/mypy/issues/731
_GnVal = bool | int | str | list[str] | dict[str, Any]
_GnScope = dict[str, _GnVal]


class _GnBuildFile(_BuildFile):
    _DEFAULT_FILENAME = 'BUILD.gn'
    _INCLUDE_CONFIG_TARGET = 'public_include_path'

    def __init__(
        self,
        directory: Path,
        ctx: _ModuleContext,
        filename: str = _DEFAULT_FILENAME,
    ):
        super().__init__(directory / filename, ctx)

    def _indent_width(self) -> int:
        return 2

    def _write_preamble(self, file: _OutputFile) -> None:
        # Upstream modules always require a tests target, even if it's empty.
        has_tests = len(self._cc_tests) > 0 or self._ctx.is_upstream

        imports = []

        if self._cc_targets:
            imports.append('$dir_pw_build/target_types.gni')

        if has_tests:
            imports.append('$dir_pw_unit_test/test.gni')

        if self._docs_sources:
            imports.append('$dir_pw_docgen/docs.gni')

        file.line('import("//build_overrides/pigweed.gni")\n')
        for imp in sorted(imports):
            file.line(f'import("{imp}")')

        if self._cc_targets:
            file.line()
            _GnBuildFile._target(
                file,
                'config',
                _GnBuildFile._INCLUDE_CONFIG_TARGET,
                {
                    'include_dirs': ['public'],
                    'visibility': [':*'],
                },
            )

        if has_tests:
            file.line()
            _GnBuildFile._target(
                file,
                'pw_test_group',
                'tests',
                {
                    'tests': list(f':{test.name}' for test in self._cc_tests),
                },
            )

    def _write_cc_target(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        """Defines a GN source_set for a C++ target."""

        target_vars: _GnScope = {}

        if target.headers:
            target_vars['public_configs'] = [
                f':{_GnBuildFile._INCLUDE_CONFIG_TARGET}'
            ]
            target_vars['public'] = list(target.rebased_headers(self.dir))

        if target.sources:
            target_vars['sources'] = list(target.rebased_sources(self.dir))

        if target.deps:
            target_vars['deps'] = target.deps

        _GnBuildFile._target(file, 'pw_source_set', target.name, target_vars)

    def _write_cc_test(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        _GnBuildFile._target(
            file,
            'pw_test',
            target.name,
            {
                'sources': list(target.rebased_sources(self.dir)),
                'deps': target.deps,
            },
        )

    def _write_docs_target(
        self,
        file: _OutputFile,
        docs_sources: list[str],
    ) -> None:
        """Defines a pw_doc_group for module documentation."""
        _GnBuildFile._target(
            file,
            'pw_doc_group',
            'docs',
            {
                'sources': docs_sources,
            },
        )

    @staticmethod
    def _target(
        file: _OutputFile,
        target_type: str,
        name: str,
        args: _GnScope,
    ) -> None:
        """Formats a GN target."""

        file.line(f'{target_type}("{name}") {{')

        with file.indent():
            _GnBuildFile._format_gn_scope(file, args)

        file.line('}')

    @staticmethod
    def _format_gn_scope(file: _OutputFile, scope: _GnScope) -> None:
        """Formats all of the variables within a GN scope to a file.

        This function does not write the enclosing braces of the outer scope to
        support use from multiple formatting contexts.
        """
        for key, val in scope.items():
            if isinstance(val, int):
                file.line(f'{key} = {val}')
                continue

            if isinstance(val, str):
                file.line(f'{key} = {_GnBuildFile._gn_string(val)}')
                continue

            if isinstance(val, bool):
                file.line(f'{key} = {str(val).lower()}')
                continue

            if isinstance(val, dict):
                file.line(f'{key} = {{')
                with file.indent():
                    _GnBuildFile._format_gn_scope(file, val)
                file.line('}')
                continue

            # Format a list of strings.
            # TODO(frolv): Lists of other types?
            assert isinstance(val, list)

            if not val:
                file.line(f'{key} = []')
                continue

            if len(val) == 1:
                file.line(f'{key} = [ {_GnBuildFile._gn_string(val[0])} ]')
                continue

            file.line(f'{key} = [')
            with file.indent():
                for string in sorted(val):
                    file.line(f'{_GnBuildFile._gn_string(string)},')
            file.line(']')

    @staticmethod
    def _gn_string(string: str) -> str:
        """Converts a Python string into a string literal within a GN file.

        Accounts for the possibility of variable interpolation within GN,
        removing quotes if unnecessary:

            "string"           ->  "string"
            "string"           ->  "string"
            "$var"             ->  var
            "$var2"            ->  var2
            "$3var"            ->  "$3var"
            "$dir_pw_foo"      ->  dir_pw_foo
            "$dir_pw_foo:bar"  ->  "$dir_pw_foo:bar"
            "$dir_pw_foo/baz"  ->  "$dir_pw_foo/baz"
            "${dir_pw_foo}"    ->  dir_pw_foo

        """

        # Check if the entire string refers to a interpolated variable.
        #
        # Simple case: '$' followed a single word, e.g. "$my_variable".
        # Note that identifiers can't start with a number.
        if re.fullmatch(r'^\$[a-zA-Z_]\w*$', string):
            return string[1:]

        # GN permits wrapping an interpolated variable in braces.
        # Check for strings of the format "${my_variable}".
        if re.fullmatch(r'^\$\{[a-zA-Z_]\w*\}$', string):
            return string[2:-1]

        return f'"{string}"'


class _BazelBuildFile(_BuildFile):
    _DEFAULT_FILENAME = 'BUILD.bazel'

    def __init__(
        self,
        directory: Path,
        ctx: _ModuleContext,
        filename: str = _DEFAULT_FILENAME,
    ):
        super().__init__(directory / filename, ctx)

    def write(self) -> None:
        """Writes the contents of the build file to disk."""
        file = _OutputFile(self._path)
        file.write_template('BUILD.bazel.jinja', build=self, module=self._ctx)

    def _indent_width(self) -> int:
        return 4

    # TODO(tonymd): Remove these functions once all file types are created with
    # templates.
    def _write_preamble(self, file: _OutputFile) -> None:
        pass

    def _write_cc_target(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        pass

    def _write_cc_test(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        pass

    def _write_docs_target(
        self,
        file: _OutputFile,
        docs_sources: list[str],
    ) -> None:
        pass


class _CmakeBuildFile(_BuildFile):
    _DEFAULT_FILENAME = 'CMakeLists.txt'

    def __init__(
        self,
        directory: Path,
        ctx: _ModuleContext,
        filename: str = _DEFAULT_FILENAME,
    ):
        super().__init__(directory / filename, ctx)

    def write(self) -> None:
        """Writes the contents of the build file to disk."""
        file = _OutputFile(self._path)
        file.write_template(
            'CMakeLists.txt.jinja', build=self, module=self._ctx
        )

    def _indent_width(self) -> int:
        return 2

    # TODO(tonymd): Remove these functions once all file types are created with
    # templates.
    def _write_preamble(self, file: _OutputFile) -> None:
        pass

    def _write_cc_target(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        pass

    def _write_cc_test(
        self,
        file: _OutputFile,
        target: _BuildFile.CcTarget,
    ) -> None:
        pass

    def _write_docs_target(
        self,
        file: _OutputFile,
        docs_sources: list[str],
    ) -> None:
        pass


class _LanguageGenerator:
    """Generates files for a programming language in a new Pigweed module."""

    def __init__(self, ctx: _ModuleContext) -> None:
        self._ctx = ctx

    @abc.abstractmethod
    def create_source_files(self) -> None:
        """Creates the boilerplate source files required by the language."""


class _CcLanguageGenerator(_LanguageGenerator):
    """Generates boilerplate source files for a C++ module."""

    def __init__(self, ctx: _ModuleContext) -> None:
        super().__init__(ctx)

        self._public_dir = ctx.dir / 'public'
        self._headers_dir = self._public_dir / ctx.name.full

    def create_source_files(self) -> None:
        self._headers_dir.mkdir(parents=True, exist_ok=True)

        main_header = self._new_header(self._ctx.name.main)
        main_source = self._new_source(self._ctx.name.main)
        test_source = self._new_source(f'{self._ctx.name.main}_test')

        # TODO(frolv): This could be configurable.
        namespace = self._ctx.name.default_namespace

        main_source.line(
            f'#include "{main_header.path.relative_to(self._public_dir)}"\n'
        )
        main_source.line(f'namespace {namespace} {{\n')
        main_source.line('int magic = 42;\n')
        main_source.line(f'}}  // namespace {namespace}')

        main_header.line(f'namespace {namespace} {{\n')
        main_header.line('extern int magic;\n')
        main_header.line(f'}}  // namespace {namespace}')

        test_source.line(
            f'#include "{main_header.path.relative_to(self._public_dir)}"\n'
        )
        test_source.line('#include "pw_unit_test/framework.h"\n')
        test_source.line(f'namespace {namespace} {{')
        test_source.line('namespace {\n')
        test_source.line(
            f'TEST({self._ctx.name.upper_camel_case()}, GeneratesCorrectly) {{'
        )
        with test_source.indent():
            test_source.line('EXPECT_EQ(magic, 42);')
        test_source.line('}\n')
        test_source.line('}  // namespace')
        test_source.line(f'}}  // namespace {namespace}')

        self._ctx.add_cc_target(
            _BuildFile.CcTarget(
                name=self._ctx.name.full,
                sources=[main_source.path],
                headers=[main_header.path],
            )
        )

        self._ctx.add_cc_test(
            _BuildFile.CcTarget(
                name=f'{self._ctx.name.main}_test',
                deps=[f':{self._ctx.name.full}'],
                sources=[test_source.path],
            )
        )

        main_header.write()
        main_source.write()
        test_source.write()

    def _new_source(self, name: str) -> _OutputFile:
        file = _OutputFile(self._ctx.dir / f'{name}.cc')

        if self._ctx.is_upstream:
            file.line(_PIGWEED_LICENSE_CC)
            file.line()

        return file

    def _new_header(self, name: str) -> _OutputFile:
        file = _OutputFile(self._headers_dir / f'{name}.h')

        if self._ctx.is_upstream:
            file.line(_PIGWEED_LICENSE_CC)

        file.line('#pragma once\n')
        return file


_BUILD_FILES: dict[str, Type[_BuildFile]] = {
    'bazel': _BazelBuildFile,
    'cmake': _CmakeBuildFile,
    'gn': _GnBuildFile,
}

_LANGUAGE_GENERATORS: dict[str, Type[_LanguageGenerator]] = {
    'cc': _CcLanguageGenerator,
}


def _check_module_name(
    module: str,
    is_upstream: bool,
) -> _ModuleName | None:
    """Checks whether a module name is valid."""

    name = _ModuleName.parse(module)
    if not name:
        _LOG.error(
            '"%s" does not conform to the Pigweed module name format', module
        )
        return None

    if is_upstream and name.prefix != 'pw':
        _LOG.error('Modules within Pigweed itself must start with "pw_"')
        return None

    return name


def _create_main_docs_file(ctx: _ModuleContext) -> None:
    """Populates the top-level docs.rst file within a new module."""

    template = get_template('docs.rst.jinja')
    rendered_template = template.render(module=ctx)

    docs_file = _OutputFile(ctx.dir / 'docs.rst')
    ctx.add_docs_file(docs_file.path)
    docs_file.write(content=rendered_template)


def _basic_module_setup(
    module_name: _ModuleName,
    module_dir: Path,
    build_systems: Iterable[str],
    is_upstream: bool,
) -> _ModuleContext:
    """Creates the basic layout of a Pigweed module."""
    module_dir.mkdir(parents=True, exist_ok=True)
    public_dir = module_dir / 'public' / module_name.full
    public_dir.mkdir(parents=True, exist_ok=True)

    ctx = _ModuleContext(
        name=module_name,
        dir=module_dir,
        root_build_files=[],
        sub_build_files=[],
        build_systems=list(build_systems),
        is_upstream=is_upstream,
    )

    ctx.root_build_files.extend(
        _BUILD_FILES[build](module_dir, ctx) for build in ctx.build_systems
    )

    _create_main_docs_file(ctx)

    return ctx


def _add_to_module_metadata(
    project_root: Path,
    module_name: _ModuleName,
    languages: Iterable[str] | None = None,
) -> None:
    """Update sphinx module metadata."""
    module_metadata_file = project_root / 'docs/module_metadata.json'
    metadata_dict = json.loads(module_metadata_file.read_text())

    language_tags = []
    if languages:
        for lang in languages:
            if lang == 'cc':
                language_tags.append('C++')

    # Add the new entry if it doesn't exist
    if module_name.full not in metadata_dict:
        metadata_dict[module_name.full] = dict(
            status='experimental',
            languages=language_tags,
        )

    # Sort by module name.
    sorted_metadata = dict(
        sorted(metadata_dict.items(), key=lambda item: item[0])
    )
    output_text = json.dumps(sorted_metadata, sort_keys=False, indent=2)
    output_text += '\n'

    # Write the file.
    if _prompt_overwrite(module_metadata_file, new_contents=output_text):
        _report_write_file(module_metadata_file)
        module_metadata_file.write_text(output_text)


def _add_to_pigweed_modules_file(
    project_root: Path,
    module_name: _ModuleName,
) -> None:
    modules_file = project_root / 'PIGWEED_MODULES'
    if not modules_file.exists():
        _LOG.error(
            'Could not locate PIGWEED_MODULES file; '
            'your repository may be in a bad state.'
        )
        return

    modules_gni_file = (
        project_root / 'pw_build' / 'generated_pigweed_modules_lists.gni'
    )

    # Cut off the extra newline at the end of the file.
    modules_list = modules_file.read_text().splitlines()
    if module_name.path in modules_list:
        _report_unchanged_file(modules_file)
        return
    modules_list.append(module_name.path)
    modules_list.sort()
    modules_list.append('')
    modules_file.write_text('\n'.join(modules_list))
    _report_edited_file(modules_file)

    generate_modules_lists.main(
        root=project_root,
        modules_list=modules_file,
        modules_gni_file=modules_gni_file,
        mode=generate_modules_lists.Mode.UPDATE,
    )
    _report_edited_file(modules_gni_file)


def _add_to_root_cmakelists(
    project_root: Path,
    module_name: _ModuleName,
) -> None:
    new_line = f'add_subdirectory({module_name.path} EXCLUDE_FROM_ALL)\n'

    path = project_root / 'CMakeLists.txt'
    if not path.exists():
        _LOG.error('Could not locate root CMakeLists.txt file.')
        return

    lines = path.read_text().splitlines(keepends=True)
    if new_line in lines:
        _report_unchanged_file(path)
        return

    add_subdir_start = 0
    while add_subdir_start < len(lines):
        if lines[add_subdir_start].startswith('add_subdirectory'):
            break
        add_subdir_start += 1

    insert_point = add_subdir_start
    while (
        lines[insert_point].startswith('add_subdirectory')
        and lines[insert_point] < new_line
    ):
        insert_point += 1

    lines.insert(insert_point, new_line)
    path.write_text(''.join(lines))
    _report_edited_file(path)


def _project_root() -> Path:
    """Returns the path to the root directory of the current project."""
    project_root = _PW_ENV.PW_PROJECT_ROOT
    if not project_root.is_dir():
        _LOG.error(
            'Expected env var $PW_PROJECT_ROOT to point to a directory, but '
            'found `%s` which is not a directory.',
            project_root,
        )
        sys.exit(1)
    return project_root


def _is_upstream() -> bool:
    """Returns whether this command is being run within Pigweed itself."""
    return _PW_ROOT == _project_root()


_COMMENTS = re.compile(r'\w*#.*$')


def _read_root_owners(project_root: Path) -> Iterable[str]:
    for line in (project_root / 'OWNERS').read_text().splitlines():
        line = _COMMENTS.sub('', line).strip()
        if line:
            yield line


def _create_module(
    module: str,
    languages: Iterable[str],
    build_systems: Iterable[str],
    owners: Collection[str] | None = None,
) -> None:
    project_root = _project_root()
    is_upstream = _is_upstream()

    module_name = _check_module_name(module, is_upstream)
    if not module_name:
        sys.exit(1)

    if not is_upstream:
        _LOG.error(
            '`pw module create` is experimental and does '
            'not yet support downstream projects.'
        )
        sys.exit(1)

    module_dir = project_root / module

    if module_dir.is_dir():
        _REPORT.wrn(f'Directory {module} already exists.')
        if _prompt_user('Continue?') == PromptChoice.NO:
            sys.exit(1)

    if module_dir.is_file():
        _LOG.error(
            'Cannot create module %s as a file of that name already exists',
            module,
        )
        sys.exit(1)

    if owners is not None:
        if len(owners) < 2:
            _LOG.error(
                'New modules must have at least two owners, but only `%s` was '
                'provided.',
                owners,
            )
            sys.exit(1)
        for owner in owners:
            if '@' not in owner:
                _LOG.error(
                    'Owners should be email addresses, but found `%s`', owner
                )
                sys.exit(1)
        root_owners = list(_read_root_owners(project_root))
        if not any(owner in root_owners for owner in owners):
            root_owners_str = '\n'.join(root_owners)
            _LOG.error(
                'Module owners must include at least one root owner, but only '
                '`%s` was provided. Root owners include:\n%s',
                owners,
                root_owners_str,
            )
            sys.exit(1)

    ctx = _basic_module_setup(
        module_name, module_dir, build_systems, is_upstream
    )

    if owners is not None:
        owners_file = module_dir / 'OWNERS'
        owners_text = '\n'.join(sorted(owners))
        owners_text += '\n'
        if _prompt_overwrite(owners_file, new_contents=owners_text):
            _report_write_file(owners_file)
            owners_file.write_text(owners_text)

    try:
        generators = list(_LANGUAGE_GENERATORS[lang](ctx) for lang in languages)
    except KeyError as key:
        _LOG.error('Unsupported language: %s', key)
        sys.exit(1)

    for generator in generators:
        generator.create_source_files()

    for build_file in ctx.build_files():
        build_file.write()

    if is_upstream:
        _add_to_pigweed_modules_file(project_root, module_name)
        _add_to_module_metadata(project_root, module_name, languages)
        if 'cmake' in build_systems:
            _add_to_root_cmakelists(project_root, module_name)

    print()
    _REPORT.new(f'{module_name} created at: {module_dir.relative_to(_PW_ROOT)}')


def register_subcommand(parser: argparse.ArgumentParser) -> None:
    """Registers the module `create` subcommand with `parser`."""

    def csv(s):
        return s.split(",")

    def csv_with_choices(choices: list[str], string) -> list[str]:
        chosen_items = list(string.split(','))
        invalid_items = set(chosen_items) - set(choices)
        if invalid_items:
            raise argparse.ArgumentTypeError(
                '\n'
                f'  invalid items: [ {", ".join(invalid_items)} ].\n'
                f'  choose from: [ {", ".join(choices)} ]'
            )

        return chosen_items

    parser.add_argument(
        '--build-systems',
        help=(
            'Comma-separated list of build systems the module supports. '
            f'Options: {", ".join(_BUILD_FILES.keys())}'
        ),
        default=_BUILD_FILES.keys(),
        type=functools.partial(csv_with_choices, list(_BUILD_FILES.keys())),
    )
    parser.add_argument(
        '--languages',
        help=(
            'Comma-separated list of languages the module will use. '
            f'Options: {", ".join(_LANGUAGE_GENERATORS.keys())}'
        ),
        default=[],
        type=functools.partial(
            csv_with_choices, list(_LANGUAGE_GENERATORS.keys())
        ),
    )
    if _is_upstream():
        parser.add_argument(
            '--owners',
            help=(
                'Comma-separated list of emails of the people who will own and '
                'maintain the new module. This list must contain at least two '
                'entries, and at least one user must be a top-level OWNER '
                f'(listed in `{_project_root()}/OWNERS`).'
            ),
            required=True,
            metavar='firstownername@google.com,secondownername@foo.net',
            type=csv,
        )
    parser.add_argument(
        'module', help='Name of the module to create.', metavar='MODULE_NAME'
    )
    parser.set_defaults(func=_create_module)
