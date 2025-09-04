#!/usr/bin/env python3
# Copyright 2021 The Pigweed Authors
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
"""Tests for the thread analyzer."""

import unittest
from pw_thread.thread_analyzer import ThreadInfo, ThreadSnapshotAnalyzer
from pw_thread_protos import thread_pb2
import pw_tokenizer
from pw_tokenizer import tokens


class ThreadInfoTest(unittest.TestCase):
    """Tests that the ThreadInfo class produces expected results."""

    def test_empty_thread(self):
        thread_info = ThreadInfo(thread_pb2.Thread())
        expected = '\n'.join(
            (
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
            )
        )
        self.assertFalse(thread_info.has_stack_size_limit())
        self.assertFalse(thread_info.has_stack_used())
        self.assertEqual(expected, str(thread_info))

    def test_thread_with_cpu_usage(self):
        thread = thread_pb2.Thread(cpu_usage_hundredths=1234)
        thread_info = ThreadInfo(thread)

        expected = '\n'.join(
            (
                'Est CPU usage: 12.34%',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
            )
        )
        self.assertFalse(thread_info.has_stack_size_limit())
        self.assertFalse(thread_info.has_stack_used())
        self.assertEqual(expected, str(thread_info))

    def test_thread_with_stack_pointer(self):
        thread = thread_pb2.Thread(stack_pointer=0x5AC6A86C)
        thread_info = ThreadInfo(thread)

        expected = '\n'.join(
            (
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x5ac6a86c (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
            )
        )
        self.assertFalse(thread_info.has_stack_size_limit())
        self.assertFalse(thread_info.has_stack_used())
        self.assertEqual(expected, str(thread_info))

    def test_thread_with_stack_usage(self):
        thread = thread_pb2.Thread(
            stack_start_pointer=0x5AC6B86C,
            stack_pointer=0x5AC6A86C,
        )
        thread_info = ThreadInfo(thread)

        expected = '\n'.join(
            (
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x5ac6b86c - 0x5ac6a86c (4096 bytes)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x5ac6b86c - 0x???????? (size unknown)',
            )
        )
        self.assertFalse(thread_info.has_stack_size_limit())
        self.assertTrue(thread_info.has_stack_used())
        self.assertEqual(expected, str(thread_info))

    def test_thread_with_zero_size_stack(self):
        thread = thread_pb2.Thread(
            stack_start_pointer=0x5AC6B86C,
            stack_end_pointer=0x5AC6B86C,
            stack_pointer=0x5AC6A86C,
            stack_pointer_est_peak=0x5AC6A86C,
        )
        thread_info = ThreadInfo(thread)

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x5ac6b86c - 0x5ac6a86c (4096 bytes, NaN%)',
                '  Est peak usage:  4096 bytes, NaN%',
                '  Stack limits:    0x5ac6b86c - 0x5ac6b86c (WARNING: total stack size is 0 bytes)',
            )
        )
        # pylint: enable=line-too-long
        self.assertTrue(thread_info.has_stack_size_limit())
        self.assertTrue(thread_info.has_stack_used())
        self.assertEqual(expected, str(thread_info))

    def test_thread_with_all_stack_info(self):
        thread = thread_pb2.Thread(
            stack_start_pointer=0x5AC6B86C,
            stack_end_pointer=0x5AC6986C,
            stack_pointer=0x5AC6A86C,
        )
        thread_info = ThreadInfo(thread)

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x5ac6b86c - 0x5ac6a86c (4096 bytes, 50.00%)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x5ac6b86c - 0x5ac6986c (8192 bytes)',
            )
        )
        # pylint: enable=line-too-long
        self.assertTrue(thread_info.has_stack_size_limit())
        self.assertTrue(thread_info.has_stack_used())
        self.assertEqual(expected, str(thread_info))


class ThreadSnapshotAnalyzerTest(unittest.TestCase):
    """Tests that the ThreadSnapshotAnalyzer class produces expected results."""

    def test_no_threads(self):
        analyzer = ThreadSnapshotAnalyzer(thread_pb2.SnapshotThreadInfo())
        self.assertEqual('', str(analyzer))

    def test_one_empty_thread(self):
        snapshot = thread_pb2.SnapshotThreadInfo(threads=[thread_pb2.Thread()])
        expected = '\n'.join(
            (
                'Thread State',
                '  1 thread running.',
                '',
                'Thread (UNKNOWN): [unnamed thread]',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
                '',
            )
        )
        analyzer = ThreadSnapshotAnalyzer(snapshot)
        self.assertEqual(analyzer.active_thread(), None)
        self.assertEqual(str(ThreadSnapshotAnalyzer(snapshot)), expected)

    def test_one_thread_with_id(self):
        """Ensures threads with names and IDs are printed correctly."""
        snapshot = thread_pb2.SnapshotThreadInfo(
            threads=[
                thread_pb2.Thread(
                    name='Alice'.encode(),
                    id=0x12345,
                    state=thread_pb2.ThreadState.Enum.READY,
                    stack_start_pointer=0x2001AC00,
                    stack_end_pointer=0x2001AA00,
                    stack_pointer=0x2001AB0C,
                    stack_pointer_est_peak=0x2001AA00,
                )
            ]
        )

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Thread State',
                '  1 thread running.',
                '',
                'Thread (READY): Alice (0x12345)',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001ac00 - 0x2001ab0c (244 bytes, 47.66%)',
                '  Est peak usage:  512 bytes, 100.00%',
                '  Stack limits:    0x2001ac00 - 0x2001aa00 (512 bytes)',
                '',
            )
        )
        # pylint: enable=line-too-long
        analyzer = ThreadSnapshotAnalyzer(snapshot)
        self.assertEqual(analyzer.active_thread(), None)
        self.assertEqual(str(ThreadSnapshotAnalyzer(snapshot)), expected)

    def test_two_threads(self):
        """Ensures multiple threads are printed correctly."""
        snapshot = thread_pb2.SnapshotThreadInfo(
            threads=[
                thread_pb2.Thread(
                    name='Idle'.encode(),
                    state=thread_pb2.ThreadState.Enum.READY,
                    stack_start_pointer=0x2001AC00,
                    stack_end_pointer=0x2001AA00,
                    stack_pointer=0x2001AB0C,
                    stack_pointer_est_peak=0x2001AA00,
                ),
                thread_pb2.Thread(
                    name='Alice'.encode(),
                    stack_start_pointer=0x2001B000,
                    stack_pointer=0x2001AE20,
                    state=thread_pb2.ThreadState.Enum.BLOCKED,
                ),
            ]
        )

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Thread State',
                '  2 threads running.',
                '',
                'Thread (READY): Idle',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001ac00 - 0x2001ab0c (244 bytes, 47.66%)',
                '  Est peak usage:  512 bytes, 100.00%',
                '  Stack limits:    0x2001ac00 - 0x2001aa00 (512 bytes)',
                '',
                'Thread (BLOCKED): Alice',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001b000 - 0x2001ae20 (480 bytes)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x2001b000 - 0x???????? (size unknown)',
                '',
            )
        )
        # pylint: enable=line-too-long
        analyzer = ThreadSnapshotAnalyzer(snapshot)
        self.assertEqual(analyzer.active_thread(), None)
        self.assertEqual(str(ThreadSnapshotAnalyzer(snapshot)), expected)

    def test_interrupts_with_thread(self):
        """Ensures interrupts are properly reported as active."""
        snapshot = thread_pb2.SnapshotThreadInfo(
            threads=[
                thread_pb2.Thread(
                    name='Idle'.encode(),
                    state=thread_pb2.ThreadState.Enum.READY,
                    stack_start_pointer=0x2001AC00,
                    stack_end_pointer=0x2001AA00,
                    stack_pointer=0x2001AB0C,
                    stack_pointer_est_peak=0x2001AA00,
                ),
                thread_pb2.Thread(
                    name='Main/Handler'.encode(),
                    stack_start_pointer=0x2001B000,
                    stack_pointer=0x2001AE20,
                    state=thread_pb2.ThreadState.Enum.INTERRUPT_HANDLER,
                ),
            ]
        )

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Thread State',
                '  2 threads running, Main/Handler active at the time of capture.',
                '                     ~~~~~~~~~~~~',
                '',
                # Ensure the active thread is moved to the top of the list.
                'Thread (INTERRUPT_HANDLER): Main/Handler <-- [ACTIVE]',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001b000 - 0x2001ae20 (480 bytes)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x2001b000 - 0x???????? (size unknown)',
                '',
                'Thread (READY): Idle',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001ac00 - 0x2001ab0c (244 bytes, 47.66%)',
                '  Est peak usage:  512 bytes, 100.00%',
                '  Stack limits:    0x2001ac00 - 0x2001aa00 (512 bytes)',
                '',
            )
        )
        # pylint: enable=line-too-long
        analyzer = ThreadSnapshotAnalyzer(snapshot)
        self.assertEqual(analyzer.active_thread(), snapshot.threads[1])
        self.assertEqual(str(ThreadSnapshotAnalyzer(snapshot)), expected)

    def test_active_thread(self):
        """Ensures the 'active' thread is highlighted."""
        snapshot = thread_pb2.SnapshotThreadInfo(
            threads=[
                thread_pb2.Thread(
                    name='Idle'.encode(),
                    state=thread_pb2.ThreadState.Enum.READY,
                    stack_start_pointer=0x2001AC00,
                    stack_end_pointer=0x2001AA00,
                    stack_pointer=0x2001AB0C,
                    stack_pointer_est_peak=0x2001AC00 + 0x100,
                ),
                thread_pb2.Thread(
                    name='Main/Handler'.encode(),
                    active=True,
                    stack_start_pointer=0x2001B000,
                    stack_pointer=0x2001AE20,
                    stack_pointer_est_peak=0x2001B000 + 0x200,
                    state=thread_pb2.ThreadState.Enum.INTERRUPT_HANDLER,
                ),
            ]
        )

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Thread State',
                '  2 threads running, Main/Handler active at the time of capture.',
                '                     ~~~~~~~~~~~~',
                '',
                # Ensure the active thread is moved to the top of the list.
                'Thread (INTERRUPT_HANDLER): Main/Handler <-- [ACTIVE]',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001b000 - 0x2001ae20 (480 bytes)',
                '  Est peak usage:  512 bytes',
                '  Stack limits:    0x2001b000 - 0x???????? (size unknown)',
                '',
                'Thread (READY): Idle',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x2001ac00 - 0x2001ab0c (244 bytes, 47.66%)',
                '  Est peak usage:  256 bytes, 50.00%',
                '  Stack limits:    0x2001ac00 - 0x2001aa00 (512 bytes)',
                '',
            )
        )
        # pylint: enable=line-too-long
        analyzer = ThreadSnapshotAnalyzer(snapshot)

        # Ensure the active thread is found.
        self.assertEqual(analyzer.active_thread(), snapshot.threads[1])
        self.assertEqual(str(ThreadSnapshotAnalyzer(snapshot)), expected)

    def test_tokenized_thread_name(self):
        """Ensures a tokenized thread name is detokenized."""
        snapshot = thread_pb2.SnapshotThreadInfo(
            threads=[
                thread_pb2.Thread(name=b'\x97\x74\xBE\x46'),
                thread_pb2.Thread(name=b'\x5D\xA8\x66\xAE'),
            ]
        )
        detokenizer = pw_tokenizer.Detokenizer(
            tokens.Database(
                [
                    tokens.TokenizedStringEntry(
                        0x46BE7497, 'The thread for Kuzco'
                    ),
                ]
            )
        )

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Thread State',
                '  2 threads running.',
                '',
                'Thread (UNKNOWN): The thread for Kuzco',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
                '',
                'Thread (UNKNOWN): $Xahmrg==',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
                '',
            )
        )
        # pylint: enable=line-too-long
        analyzer = ThreadSnapshotAnalyzer(snapshot, tokenizer_db=detokenizer)

        # Ensure text dump matches expected contents.
        self.assertEqual(str(analyzer), expected)

    def test_no_db_tokenized_thread_name(self):
        """Ensures a tokenized thread name is detokenized."""
        snapshot = thread_pb2.SnapshotThreadInfo(
            threads=[
                thread_pb2.Thread(name=b'\x97\x74\xBE\x46'),
                thread_pb2.Thread(name=b'\x5D\xA8\x66\xAE'),
            ]
        )

        # pylint: disable=line-too-long
        expected = '\n'.join(
            (
                'Thread State',
                '  2 threads running.',
                '',
                'Thread (UNKNOWN): $l3S+Rg==',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
                '',
                'Thread (UNKNOWN): $Xahmrg==',
                'Est CPU usage: unknown',
                'Stack info',
                '  Current usage:   0x???????? - 0x???????? (size unknown)',
                '  Est peak usage:  size unknown',
                '  Stack limits:    0x???????? - 0x???????? (size unknown)',
                '',
            )
        )
        # pylint: enable=line-too-long
        analyzer = ThreadSnapshotAnalyzer(snapshot)

        # Ensure text dump matches expected contents.
        self.assertEqual(str(analyzer), expected)


if __name__ == '__main__':
    unittest.main()
