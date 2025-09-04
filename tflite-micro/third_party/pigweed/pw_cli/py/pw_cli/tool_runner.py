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
"""A subprocess wrapper that enables injection of externally-provided tools."""

import abc
from pathlib import Path
from typing import Iterable

import subprocess


class ToolRunner(abc.ABC):
    """A callable interface that runs the requested tool as a subprocess.

    This class is used to support subprocess-like semantics while allowing
    injection of wrappers that enable testing, finer granularity identifying
    where tools fail, and stricter control of which binaries are called.

    By default, all subprocess output is captured.
    """

    def __call__(
        self,
        tool: str,
        args: Iterable[str | Path],
        stdout: int | None = subprocess.PIPE,
        stderr: int | None = subprocess.PIPE,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Calls ``tool`` with the provided ``args``.

        ``**kwargs`` are forwarded to the underlying ``subprocess.run()``
        for the requested tool.

        By default, all subprocess output is captured.

        Returns:
            The ``subprocess.CompletedProcess`` result of running the requested
            tool.
        """
        additional_kwargs = set(self._custom_args())
        allowed_kwargs = {
            key: value
            for key, value in kwargs.items()
            if not key.startswith('pw_') or key in additional_kwargs
        }

        return self._run_tool(
            tool,
            args,
            stderr=stderr,
            stdout=stdout,
            **allowed_kwargs,
        )

    @staticmethod
    def _custom_args() -> Iterable[str]:
        """List of additional keyword arguments accepted by this tool.

        By default, all kwargs passed into a tool are forwarded to
        ``subprocess.run()``. However, some tools have extra arguments custom
        to them, which are not valid for ``subprocess.run()``. Tools requiring
        these custom args should override this method, listing the arguments
        they accept.

        To make filtering custom arguments possible, they must be prefixed
        with  ``pw_``.
        """
        return []

    @abc.abstractmethod
    def _run_tool(
        self, tool: str, args, **kwargs
    ) -> subprocess.CompletedProcess:
        """Implements the subprocess runner logic.

        Calls ``tool`` with the provided ``args``. ``**kwargs`` not listed in
        ``_custom_args`` are forwarded to the underlying ``subprocess.run()``
        for the requested tool.

        Returns:
            The ``subprocess.CompletedProcess`` result of running the requested
            tool.
        """


class BasicSubprocessRunner(ToolRunner):
    """A simple ToolRunner that calls subprocess.run()."""

    def _run_tool(
        self, tool: str, args, **kwargs
    ) -> subprocess.CompletedProcess:
        return subprocess.run([tool, *args], **kwargs)
