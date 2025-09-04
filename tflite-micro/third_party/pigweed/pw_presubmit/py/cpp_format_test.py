# Copyright 2024 The Pigweed Authors
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
"""Tests for the clang-format formatter."""

import importlib.resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, Sequence
import unittest

from format_testing_utils import CapturingToolRunner
from pw_presubmit.format.cpp import ClangFormatFormatter

_TEST_DATA_FILES = importlib.resources.files('pw_presubmit.format.test_data')

_TEST_SRC_FILE = _TEST_DATA_FILES / 'cpp_test_data.cc'
_TEST_GOLDEN = _TEST_DATA_FILES / 'cpp_test_data_golden.cc'

_CLANG_FORMAT_CONFIG_PATH = _TEST_DATA_FILES / 'clang_format_config'
_CLANG_FORMAT_ARGS: Final[Sequence[str]] = (
    f'--style=file:{_CLANG_FORMAT_CONFIG_PATH}',
)


class TestClangFormatFormatter(unittest.TestCase):
    """Tests for the ClangFormatFormatter."""

    def test_check_file(self):
        tool_runner = CapturingToolRunner()
        formatter = ClangFormatFormatter(_CLANG_FORMAT_ARGS)
        formatter.run_tool = tool_runner

        result = formatter.format_file_in_memory(
            _TEST_SRC_FILE, _TEST_SRC_FILE.read_bytes()
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.error_message, None)

        self.assertEqual(
            tool_runner.command_history.pop(0),
            ' '.join(
                (
                    'clang-format',
                    f'--style=file:{_CLANG_FORMAT_CONFIG_PATH}',
                    str(_TEST_SRC_FILE),
                )
            ),
        )

        self.assertMultiLineEqual(
            result.formatted_file_contents.decode(), _TEST_GOLDEN.read_text()
        )

    def test_fix_file(self):
        tool_runner = CapturingToolRunner()
        formatter = ClangFormatFormatter(_CLANG_FORMAT_ARGS)
        formatter.run_tool = tool_runner

        with TemporaryDirectory() as temp_dir:
            file_to_fix = Path(temp_dir) / _TEST_SRC_FILE.name
            file_to_fix.write_bytes(_TEST_SRC_FILE.read_bytes())

            errors = formatter.format_files([file_to_fix])

            # Error dictionary should be empty.
            self.assertFalse(list(errors))

            self.assertEqual(
                tool_runner.command_history.pop(0),
                ' '.join(
                    (
                        'clang-format',
                        '-i',
                        f'--style=file:{_CLANG_FORMAT_CONFIG_PATH}',
                        str(file_to_fix),
                    )
                ),
            )

            self.assertMultiLineEqual(
                file_to_fix.read_text(), _TEST_GOLDEN.read_text()
            )


if __name__ == '__main__':
    unittest.main()
