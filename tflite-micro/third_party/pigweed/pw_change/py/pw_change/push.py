#!/usr/bin/python3
# Copyright 2025 The Pigweed Authors
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
"""Simple command to push changes to Gerrit."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import NoReturn

from pw_change.remote_dest import remote_dest


def _default_at_google_com(x: str) -> str:
    if '@' in x:
        return x
    return f'{x}@google.com'


# This looks like URL-encoding but it's different in which characters need to be
# escaped and which do not.
def _escaped_string(x: str) -> str:
    for ch in r'%^@.,~-+_:/!\'"[](){}\\':
        x = x.replace(ch, f'%{ord(ch):02x}')
    return x.replace(' ', '_')


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('rev', nargs='?', default='HEAD')
    parser.add_argument('-d', '--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-t', '--topic')
    parser.add_argument('--ready', action='store_true')
    parser.add_argument(
        '-q', '--commit-queue', action='append_const', const=True
    )
    parser.add_argument('-p', '--publish', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-r',
        '--reviewer',
        metavar='USERNAME',
        action='append',
        default=[],
        type=_default_at_google_com,
    )
    group.add_argument('-w', '--wip', action='store_true')

    parser.add_argument(
        '-c',
        '--cc',
        metavar='USERNAME',
        action='append',
        default=[],
        type=_default_at_google_com,
    )
    parser.add_argument('-l', '--label', action='append', default=[])
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('-a', '--auto-submit', action='store_true')
    parser.add_argument('--hashtag')
    parser.add_argument('-m', '--message', type=_escaped_string)
    args = parser.parse_args(argv)

    if args.force:
        for name in 'topic wip ready reviewer'.split():
            value = getattr(args, name)
            if value:
                parser.error(f'--force cannot be used with --{name}')

    if args.wip and args.ready:
        parser.error('--wip cannot be used with --ready')

    return args


def _auto_submit_label(host: str | None) -> str:
    return {
        'pigweed': 'Pigweed-Auto-Submit',
        'pigweed-internal': 'Pigweed-Auto-Submit',
        'fuchsia': 'Fuchsia-Auto-Submit',
        'fuchsia-internal': 'Fuchsia-Auto-Submit',
    }.get(host or '', 'Auto-Submit')


def push(args: argparse.Namespace) -> int:
    """Push changes to Gerrit."""
    remote, branch = remote_dest()

    if not args.force:
        branch = f'refs/for/{branch}'

    options: list[str] = []
    push_args: list[str] = []

    if args.topic:
        options.append(f'topic={args.topic}')

    if args.wip:
        options.append('wip')
    elif args.ready or args.reviewer:
        options.append('ready')

    if args.commit_queue:
        options.append(f'l=Commit-Queue+{len(args.commit_queue)}')

    if args.publish:
        options.append('publish-comments')

    for reviewer in args.reviewer:
        options.append(f'r={reviewer}')

    for reviewer in args.cc:
        options.append(f'cc={reviewer}')

    for label in args.label:
        options.append(f'l={label}')

    if args.hashtag:
        options.append(f'hashtag={_escaped_string(args.hashtag)}')

    if args.message:
        options.append(f'message={args.message}')

    if args.auto_submit:
        remote_url = (
            subprocess.run(
                ['git', 'config', '--get', f'remote.{remote}.url'],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        match = re.search(r'^\w+:/+(?P<name>[\w-]+)[./]', remote_url)
        name = match.group('name') if match else None
        options.append(f'l={_auto_submit_label(name)}')

    if args.no_verify:
        push_args.append('--no-verify')

    options_str = ','.join(options)
    branch = f'{branch}%{options_str}'

    cmd = ['git', 'push', remote, f'+{args.rev}:{branch}']
    cmd.extend(push_args)
    print(*cmd)

    if args.dry_run:
        print('dry run, not pushing')
    else:
        subprocess.check_call(cmd)
    return 0


def main(argv: Sequence[str] | None = None) -> int | NoReturn:
    args = parse(argv)
    return push(args)


if __name__ == '__main__':
    sys.exit(main())
