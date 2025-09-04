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
"""pw_ide settings."""

import enum
from inspect import cleandoc
import os
from pathlib import Path
from typing import Any, cast, Literal
import yaml

from pw_cli.env import pigweed_environment
from pw_config_loader.yaml_config_loader_mixin import YamlConfigLoaderMixin

env = pigweed_environment()
env_vars = vars(env)

PW_PROJECT_ROOT = Path(
    env.PW_PROJECT_ROOT if env.PW_PROJECT_ROOT is not None else os.getcwd()
)

PW_IDE_DIR_NAME = '.pw_ide'

_DEFAULT_BUILD_DIR_NAME = 'out'
_DEFAULT_BUILD_DIR = PW_PROJECT_ROOT / _DEFAULT_BUILD_DIR_NAME

_DEFAULT_WORKSPACE_ROOT = PW_PROJECT_ROOT

_DEFAULT_TARGET_INFERENCE = '?'

SupportedEditorName = Literal['vscode']


class SupportedEditor(enum.Enum):
    VSCODE = 'vscode'


_DEFAULT_SUPPORTED_EDITORS: dict[SupportedEditorName, bool] = {
    'vscode': True,
}

_DEFAULT_CONFIG: dict[str, Any] = {
    'cascade_targets': False,
    'clangd_alternate_path': None,
    'clangd_additional_query_drivers': [],
    'compdb_gen_cmd': None,
    'compdb_search_paths': [_DEFAULT_BUILD_DIR_NAME],
    'default_target': None,
    'workspace_root': _DEFAULT_WORKSPACE_ROOT,
    'editors': _DEFAULT_SUPPORTED_EDITORS,
    'sync': ['pw --no-banner ide cpp --process'],
    'targets_exclude': [],
    'targets_include': [],
    'target_inference': _DEFAULT_TARGET_INFERENCE,
    'working_dir': _DEFAULT_WORKSPACE_ROOT / PW_IDE_DIR_NAME,
}

_DEFAULT_PROJECT_FILE = PW_PROJECT_ROOT / '.pw_ide.yaml'
_DEFAULT_PROJECT_USER_FILE = PW_PROJECT_ROOT / '.pw_ide.user.yaml'
_DEFAULT_USER_FILE = Path.home() / '.pw_ide.yaml'


def _expand_any_vars(input_path: Path) -> Path:
    """Expand any environment variables in a path.

    Python's ``os.path.expandvars`` will only work on an isolated environment
    variable name. In shell, you can expand variables within a larger command
    or path. We replicate that functionality here.
    """
    outputs = []

    for token in input_path.parts:
        expanded_var = os.path.expandvars(token)

        if expanded_var == token:
            outputs.append(token)
        else:
            outputs.append(expanded_var)

    # pylint: disable=no-value-for-parameter
    return Path(os.path.join(*outputs))
    # pylint: enable=no-value-for-parameter


def _expand_any_vars_str(input_path: str) -> str:
    """`_expand_any_vars`, except takes and returns a string instead of path."""
    return str(_expand_any_vars(Path(input_path)))


def _parse_dir_path(input_path_str: str, workspace_root: Path) -> Path:
    if (path := Path(input_path_str)).is_absolute():
        return path

    return (workspace_root / path).resolve()


def _parse_compdb_search_path(
    input_data: str | tuple[str, str],
    default_inference: str,
    workspace_root: Path,
) -> tuple[Path, str]:
    if isinstance(input_data, (tuple, list)):
        return _parse_dir_path(input_data[0], workspace_root), input_data[1]

    return _parse_dir_path(input_data, workspace_root), default_inference


class PigweedIdeSettings(YamlConfigLoaderMixin):
    """Pigweed IDE features settings storage class."""

    def __init__(
        self,
        project_file: Path | bool = _DEFAULT_PROJECT_FILE,
        project_user_file: Path | bool = _DEFAULT_PROJECT_USER_FILE,
        user_file: Path | bool = _DEFAULT_USER_FILE,
        default_config: dict[str, Any] | None = None,
    ) -> None:
        self.config_init(
            config_section_title='pw_ide',
            project_file=project_file,
            project_user_file=project_user_file,
            user_file=user_file,
            default_config=_DEFAULT_CONFIG
            if default_config is None
            else default_config,
            environment_var='PW_IDE_CONFIG_FILE',
        )

        # YamlConfigLoaderMixin only conditionally assigns this, but we rely
        # on it later, so we just extra assign it here.
        self.project_file = project_file  # type: ignore

    def __repr__(self) -> str:
        return str(
            {
                key: getattr(self, key)
                for key, value in self.__class__.__dict__.items()
                if isinstance(value, property)
            }
        )

    @property
    def working_dir(self) -> Path:
        """Path to the ``pw_ide`` working directory.

        The working directory holds C++ compilation databases and caches, and
        other supporting files. This should not be a directory that's regularly
        deleted or manipulated by other processes (e.g. the GN ``out``
        directory) nor should it be committed to source control.
        """
        return Path(
            _expand_any_vars_str(
                self._config.get(
                    'working_dir', self.workspace_root / PW_IDE_DIR_NAME
                )
            )
        )

    @property
    def compdb_gen_cmd(self) -> str | None:
        """The command that should be run to generate a compilation database.

        Defining this allows ``pw_ide`` to automatically generate a compilation
        database if it suspects one has not been generated yet before a sync.
        """
        return self._config.get('compdb_gen_cmd')

    @property
    def compdb_search_paths(self) -> list[tuple[Path, str]]:
        """Paths to directories to search for compilation databases.

        If you're using a build system to generate compilation databases, this
        may simply be your build output directory. However, you can add
        additional directories to accommodate compilation databases from other
        sources.

        Entries can be just directories, in which case the default target
        inference pattern will be used. Or entries can be tuples of a directory
        and a target inference pattern. See the documentation for
        ``target_inference`` for more information.

        Finally, the directories can be concrete paths, or they can be globs
        that expand to multiple paths.

        Note that relative directory paths will be resolved relative to the
        workspace root.
        """
        return [
            _parse_compdb_search_path(
                search_path, self.target_inference, self.workspace_root
            )
            for search_path in self._config.get(
                'compdb_search_paths', [_DEFAULT_BUILD_DIR]
            )
        ]

    @property
    def targets_exclude(self) -> list[str]:
        """The list of targets that should not be enabled for code analysis.

        In this case, "target" is analogous to a GN target, i.e., a particular
        build configuration. By default, all available targets are enabled. By
        adding targets to this list, you can disable/hide targets that should
        not be available for code analysis.

        Target names need to match the name of the directory that holds the
        build system artifacts for the target. For example, GN outputs build
        artifacts for the ``pw_strict_host_clang_debug`` target in a directory
        with that name in its output directory. So that becomes the canonical
        name for the target.
        """
        return self._config.get('targets_exclude', list())

    @property
    def targets_include(self) -> list[str]:
        """The list of targets that should be enabled for code analysis.

        In this case, "target" is analogous to a GN target, i.e., a particular
        build configuration. By default, all available targets are enabled. By
        adding targets to this list, you can constrain the targets that are
        enabled for code analysis to a subset of those that are available, which
        may be useful if your project has many similar targets that are
        redundant from a code analysis perspective.

        Target names need to match the name of the directory that holds the
        build system artifacts for the target. For example, GN outputs build
        artifacts for the ``pw_strict_host_clang_debug`` target in a directory
        with that name in its output directory. So that becomes the canonical
        name for the target.
        """
        return self._config.get('targets_include', list())

    @property
    def target_inference(self) -> str:
        """A glob-like string for extracting a target name from an output path.

        Build systems and projects have varying ways of organizing their build
        directory structure. For a given compilation unit, we need to know how
        to extract the build's target name from the build artifact path. A
        simple example:

        .. code-block:: none

           clang++ hello.cc -o host/obj/hello.cc.o

        The top-level directory ``host`` is the target name we want. The same
        compilation unit might be used with another build target:

        .. code-block:: none

           gcc-arm-none-eabi hello.cc -o arm_dev_board/obj/hello.cc.o

        In this case, this compile command is associated with the
        ``arm_dev_board`` target.

        When importing and processing a compilation database, we assume by
        default that for each compile command, the corresponding target name is
        the name of the top level directory within the build directory root
        that contains the build artifact. This is the default behavior for most
        build systems. However, if your project is structured differently, you
        can provide a glob-like string that indicates how to extract the target
        name from build artifact path.

        A ``*`` indicates any directory, and ``?`` indicates the directory that
        has the name of the target. The path is resolved from the build
        directory root, and anything deeper than the target directory is
        ignored. For example, a glob indicating that the directory two levels
        down from the build directory root has the target name would be
        expressed with ``*/*/?``.

        Note that the build artifact path is relative to the compilation
        database search path that found the file. For example, for a compilation
        database search path of ``{project dir}/out``, for the purposes of
        target inference, the build artifact path is relative to the ``{project
        dir}/out`` directory. Target inference patterns can be defined for each
        compilation database search path. See the documentation for
        ``compdb_search_paths`` for more information.
        """
        return self._config.get('target_inference', _DEFAULT_TARGET_INFERENCE)

    @property
    def default_target(self) -> str | None:
        """The default target to use when calling ``--set-default``.

        This target will be selected when ``pw ide cpp --set-default`` is
        called. You can define an explicit default target here. If that command
        is invoked without a default target definition, ``pw_ide`` will try to
        infer the best choice of default target. Currently, it selects the
        target with the broadest compilation unit coverage.
        """
        return self._config.get('default_target', None)

    @property
    def sync(self) -> list[str]:
        """A sequence of commands to automate IDE features setup.

        ``pw ide sync`` should do everything necessary to get the project from
        a fresh checkout to a working default IDE experience. This defines the
        list of commands that makes that happen, which will be executed
        sequentially in subprocesses. These commands should be idempotent, so
        that the user can run them at any time to update their IDE features
        configuration without the risk of putting those features in a bad or
        unexpected state.
        """
        return self._config.get('sync', list())

    @property
    def clangd_alternate_path(self) -> Path | None:
        """An alternate path to ``clangd`` to use instead of Pigweed's.

        Pigweed provides the ``clang`` toolchain, including ``clangd``, via
        CIPD, and by default, ``pw_ide`` will look for that toolchain in the
        CIPD directory at ``$PW_PIGWEED_CIPD_INSTALL_DIR`` *or* in an alternate
        CIPD directory specified by ``$PW_{project name}_CIPD_INSTALL_DIR`` if
        it exists.

        If your project needs to use a ``clangd`` located somewhere else not
        covered by the cases described above, you can define the path to that
        ``clangd`` here.
        """
        return self._config.get('clangd_alternate_path', None)

    @property
    def clangd_additional_query_drivers(self) -> list[str] | None:
        """Additional query driver paths that clangd should use.

        By default, ``pw_ide`` supplies driver paths for the toolchains included
        in Pigweed. If you are using toolchains that are not supplied by
        Pigweed, you should include path globs to your toolchains here. These
        paths will be given higher priority than the Pigweed toolchain paths.

        If you want to omit the query drivers argument altogether, set this to
        ``null``.
        """
        return self._config.get('clangd_additional_query_drivers', list())

    def clangd_query_drivers(
        self, host_clang_cc_path: Path
    ) -> list[str] | None:
        if self.clangd_additional_query_drivers is None:
            return None

        drivers = [
            *[
                _expand_any_vars_str(p)
                for p in self.clangd_additional_query_drivers
            ],
        ]

        drivers.append(str(host_clang_cc_path.parent / '*'))

        if (env_var := env_vars.get('PW_ARM_CIPD_INSTALL_DIR')) is not None:
            drivers.append(str(Path(env_var) / 'bin' / '*'))

        return drivers

    def clangd_query_driver_str(self, host_clang_cc_path: Path) -> str | None:
        clangd_query_drivers = self.clangd_query_drivers(host_clang_cc_path)

        if clangd_query_drivers is None:
            return None

        return ','.join(clangd_query_drivers)

    @property
    def workspace_root(self) -> Path:
        """The root directory of the IDE workspace.

        In most cases, the IDE workspace directory is also your Pigweed project
        root; that's the default, and if that's the case, this you can omit this
        configuration.

        If your project has a structure where the directory you open in your
        IDE is *not* the Pigweed project root directory, you can specify the
        workspace root directory here to ensure that IDE support files are
        put in the right place.

        For example, given this directory structure:

        .. code-block::

           my_project
           |
           ├- third_party
           ├- docs
           ├- src
           |  ├- pigweed.json
           |  └- ... all other project source files
           |
           └- pigweed
               └ ... upstream Pigweed source, e.g. a submodule

        In this case ```my_project/src/``` is the Pigweed project root, but
        ```my_project/``` is the workspace root directory you will open in your
        IDE.

        A relative path will be resolved relative to the directory in which the
        ``.pw_ide.yaml`` config file is located.
        """
        workspace_root = Path(
            self._config.get('workspace_root', _DEFAULT_WORKSPACE_ROOT)
        )

        project_file_exists = (
            isinstance(self.project_file, Path) and self.project_file.exists()
        )

        config_file_root = (
            cast(Path, self.project_file)
            if project_file_exists
            else _DEFAULT_PROJECT_FILE
        ).parent

        if workspace_root.is_absolute():
            return workspace_root

        return config_file_root.resolve() / workspace_root

    @property
    def editors(self) -> dict[str, bool]:
        """Enable or disable automated support for editors.

        Automatic support for some editors is provided by ``pw_ide``, which is
        accomplished through generating configuration files in your project
        directory. All supported editors are enabled by default, but you can
        disable editors by adding an ``'<editor>': false`` entry.
        """
        return self._config.get('editors', _DEFAULT_SUPPORTED_EDITORS)

    def editor_enabled(self, editor: SupportedEditorName) -> bool:
        """True if the provided editor is enabled in settings.

        This module will integrate the project with all supported editors by
        default. If the project or user want to disable particular editors,
        they can do so in the appropriate settings file.
        """
        return self._config.get('editors', {}).get(editor, False)

    @property
    def cascade_targets(self) -> bool:
        """Mix compile commands for multiple targets to maximize code coverage.

        By default (with this set to ``False``), the compilation database for
        each target is consistent in the sense that it only contains compile
        commands for one build target, so the code intelligence that database
        provides is related to a single, known compilation artifact. However,
        this means that code intelligence may not be provided for every source
        file in a project, because some source files may be relevant to targets
        other than the one you have currently set. Those source files won't
        have compile commands for the current target, and no code intelligence
        will appear in your editor.

        If this is set to ``True``, compilation databases will still be
        separated by target, but compile commands for *all other targets* will
        be appended to the list of compile commands for *that* target. This
        will maximize code coverage, ensuring that you have code intelligence
        for every file that is built for any target, at the cost of
        consistency—the code intelligence for some files may show information
        that is incorrect or irrelevant to the currently selected build target.

        The currently set target's compile commands will take priority at the
        top of the combined file, then all other targets' commands will come
        after in order of the number of commands they have (i.e. in the order of
        their code coverage). This relies on the fact that ``clangd`` parses the
        compilation database from the top down, using the first compile command
        it encounters for each compilation unit.
        """
        return self._config.get('cascade_targets', False)


def _docstring_set_default(
    obj: Any, default: Any, literal: bool = False
) -> None:
    """Add a default value annotation to a docstring.

    Formatting isn't allowed in docstrings, so by default we can't inject
    variables that we would like to appear in the documentation, like the
    default value of a property. But we can use this function to add it
    separately.
    """
    if obj.__doc__ is not None:
        default = str(default)

        if literal:
            lines = default.splitlines()

            if len(lines) == 0:
                return
            if len(lines) == 1:
                default = f'Default: ``{lines[0]}``'
            else:
                default = 'Default:\n\n.. code-block::\n\n  ' + '\n  '.join(
                    lines
                )

        doc = cast(str, obj.__doc__)
        obj.__doc__ = f'{cleandoc(doc)}\n\n{default}'


_docstring_set_default(
    PigweedIdeSettings.working_dir, PW_IDE_DIR_NAME, literal=True
)
_docstring_set_default(
    PigweedIdeSettings.compdb_gen_cmd,
    _DEFAULT_CONFIG['compdb_gen_cmd'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.compdb_search_paths,
    [_DEFAULT_BUILD_DIR_NAME],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.targets_exclude,
    _DEFAULT_CONFIG['targets_exclude'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.targets_include,
    _DEFAULT_CONFIG['targets_include'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.default_target,
    _DEFAULT_CONFIG['default_target'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.cascade_targets,
    _DEFAULT_CONFIG['cascade_targets'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.target_inference,
    _DEFAULT_CONFIG['target_inference'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.sync, _DEFAULT_CONFIG['sync'], literal=True
)
_docstring_set_default(
    PigweedIdeSettings.clangd_alternate_path,
    _DEFAULT_CONFIG['clangd_alternate_path'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.clangd_additional_query_drivers,
    _DEFAULT_CONFIG['clangd_additional_query_drivers'],
    literal=True,
)
_docstring_set_default(
    PigweedIdeSettings.editors,
    yaml.dump(_DEFAULT_SUPPORTED_EDITORS),
    literal=True,
)
