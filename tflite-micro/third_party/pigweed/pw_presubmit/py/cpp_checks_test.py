#!/usr/bin/env python3
# Copyright 2023 The Pigweed Authors
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
"""Tests for cpp_checks."""

import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, mock_open, patch

from pw_presubmit import cpp_checks

# pylint: disable=attribute-defined-outside-init


def _temproot():
    root = Path(os.environ['_PW_ACTUAL_ENVIRONMENT_ROOT']) / 'temp'
    root.mkdir(exist_ok=True)
    return root


class TestPragmaOnce(unittest.TestCase):
    """Test pragma_once check."""

    def _run(self, contents: str) -> None:
        self.ctx = MagicMock()
        self.ctx.fail = MagicMock()
        self.ctx.format_options.filter_paths = lambda x: x
        path = MagicMock(spec=Path('include/foo.h'))

        def mocked_open_read(*args, **kwargs):
            return mock_open(read_data=contents)(*args, **kwargs)

        with patch.object(path, 'open', mocked_open_read):
            self.ctx.paths = [path]
            cpp_checks.pragma_once(self.ctx)

    def test_empty(self) -> None:
        self._run('')
        self.ctx.fail.assert_called()

    def test_simple(self) -> None:
        self._run('#pragma once')
        self.ctx.fail.assert_not_called()

    def test_not_first(self) -> None:
        self._run('1\n2\n3\n#pragma once')
        self.ctx.fail.assert_not_called()

    def test_different_pragma(self) -> None:
        self._run('#pragma thrice')
        self.ctx.fail.assert_called()


def guard(path: Path) -> str:
    return str(path.name).replace('.', '_').upper()


class TestIncludeGuard(unittest.TestCase):
    """Test pragma_once check."""

    def _run(self, contents: str, **kwargs) -> None:
        self.ctx = MagicMock()
        self.ctx.format_options.filter_paths = lambda x: x
        self.ctx.fail = MagicMock()
        path = MagicMock(spec=Path('abc/def/foo.h'))
        path.name = 'foo.h'

        def mocked_open_read(*args, **kwargs):
            return mock_open(read_data=contents)(*args, **kwargs)

        with patch.object(path, 'open', mocked_open_read):
            self.ctx.paths = [path]
            check = cpp_checks.include_guard_check(**kwargs)
            check(self.ctx)

    def test_empty(self) -> None:
        self._run('')
        self.ctx.fail.assert_called()

    def test_simple(self) -> None:
        self._run('#ifndef FOO_H\n#define FOO_H', guard=guard)
        self.ctx.fail.assert_not_called()

    def test_simple_any(self) -> None:
        self._run('#ifndef BAR_H\n#define BAR_H')
        self.ctx.fail.assert_not_called()

    def test_simple_comments(self) -> None:
        self._run('// abc\n#ifndef BAR_H\n// def\n#define BAR_H\n// ghi')
        self.ctx.fail.assert_not_called()

    def test_simple_whitespace(self) -> None:
        self._run(
            '         // abc\n #ifndef           BAR_H\n// def\n'
            '                  #define BAR_H\n// ghi\n #endif'
        )
        self.ctx.fail.assert_not_called()

    def test_simple_not_expected(self) -> None:
        self._run('#ifnef BAR_H\n#define BAR_H', guard=guard)
        self.ctx.fail.assert_called()

    def test_no_define(self) -> None:
        self._run('#ifndef FOO_H')
        self.ctx.fail.assert_called()

    def test_non_matching_define(self) -> None:
        self._run('#ifndef FOO_H\n#define BAR_H')
        self.ctx.fail.assert_called()


if __name__ == '__main__':
    unittest.main()
