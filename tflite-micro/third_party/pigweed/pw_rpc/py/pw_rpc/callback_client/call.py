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
"""Classes for handling ongoing RPC calls."""

from __future__ import annotations

import enum
from collections import deque
import logging
import math
import queue
from typing import (
    Any,
    Callable,
    Deque,
    Iterable,
    Iterator,
    NamedTuple,
    Sequence,
    TypeVar,
)

from pw_protobuf_compiler.python_protos import proto_repr
from pw_status import Status
from google.protobuf.message import Message

from pw_rpc.callback_client.errors import RpcTimeout, RpcError
from pw_rpc.client import PendingRpc, PendingRpcs
from pw_rpc.descriptors import Method

_LOG = logging.getLogger(__package__)


class UseDefault(enum.Enum):
    """Marker for args that should use a default value, when None is valid."""

    VALUE = 0


CallTypeT = TypeVar(
    'CallTypeT',
    'UnaryCall',
    'ServerStreamingCall',
    'ClientStreamingCall',
    'BidirectionalStreamingCall',
)

OnNextCallback = Callable[[CallTypeT, Any], Any]
OnCompletedCallback = Callable[[CallTypeT, Any], Any]
OnErrorCallback = Callable[[CallTypeT, Any], Any]

OptionalTimeout = UseDefault | float | None


class UnaryResponse(NamedTuple):
    """Result from a unary or client streaming RPC: status and response."""

    status: Status
    response: Any

    def unwrap_or_raise(self):
        """Returns the response value or raises `ValueError` if not OK."""
        if not self.status.ok():
            raise ValueError(f'RPC returned non-OK status: {self.status}')
        return self.response

    def __repr__(self) -> str:
        reply = proto_repr(self.response) if self.response else self.response
        return f'({self.status}, {reply})'


class StreamResponse(NamedTuple):
    """Results from a server or bidirectional streaming RPC."""

    status: Status
    responses: Sequence[Any]

    def unwrap_or_raise(self):
        """Returns the response value or raises `ValueError` if not OK."""
        if not self.status.ok():
            raise ValueError(f'RPC returned non-OK status: {self.status}')
        return self.responses

    def __repr__(self) -> str:
        return (
            f'({self.status}, '
            f'[{", ".join(proto_repr(r) for r in self.responses)}])'
        )


class Call:
    """Represents an in-progress or completed RPC call."""

    def __init__(
        self,
        rpcs: PendingRpcs,
        rpc: PendingRpc,
        default_timeout_s: float | None,
        on_next: OnNextCallback | None,
        on_completed: OnCompletedCallback | None,
        on_error: OnErrorCallback | None,
        max_responses: int,
    ) -> None:
        self._rpcs = rpcs
        self._rpc = rpc
        self.default_timeout_s = default_timeout_s

        self.status: Status | None = None
        self.error: Status | None = None
        self._callback_exception: Exception | None = None
        self._responses: Deque = deque(maxlen=max_responses)
        self._response_queue: queue.SimpleQueue = queue.SimpleQueue()

        self.on_next = on_next or Call._default_response
        self.on_completed = on_completed or Call._default_completion
        self.on_error = on_error or Call._default_error

    def _invoke(self, request: Message | None) -> None:
        """Calls the RPC. This must be called immediately after __init__."""
        self._rpcs.send_request(self._rpc, request, self)

    def _open(self) -> None:
        self._rpcs.open(self._rpc, self)

    def _default_response(self, response: Message) -> None:
        _LOG.debug('%s received response: %s', self._rpc, response)

    def _default_completion(self, status: Status) -> None:
        _LOG.info('%s completed: %s', self._rpc, status)

    def _default_error(self, error: Status) -> None:
        _LOG.warning('%s terminated due to an error: %s', self._rpc, error)

    @property
    def call_id(self) -> int:
        return self._rpc.call_id

    @property
    def method(self) -> Method:
        return self._rpc.method

    def completed(self) -> bool:
        """True if the RPC call has completed, successfully or from an error."""
        return self.status is not None or self.error is not None

    def _send_client_stream(
        self, request_proto: Message | None, request_fields: dict
    ) -> None:
        """Sends a client to the server in the client stream.

        Sending a client stream packet on a closed RPC raises an exception.
        """
        self._check_errors()

        if self.status is not None:
            raise RpcError(self._rpc, Status.FAILED_PRECONDITION)

        self._rpcs.send_client_stream(
            self._rpc, self.method.get_request(request_proto, request_fields)
        )

    def _finish_client_stream(self, requests: Iterable[Message]) -> None:
        for request in requests:
            self._send_client_stream(request, {})

        if not self.completed():
            self._rpcs.send_client_stream_end(self._rpc)

    def _unary_wait(self, timeout_s: OptionalTimeout) -> UnaryResponse:
        """Waits until the RPC has completed."""
        for _ in self._get_responses(timeout_s=timeout_s):
            pass

        assert self.status is not None and self._responses
        return UnaryResponse(self.status, self._responses[-1])

    def _stream_wait(self, timeout_s: OptionalTimeout) -> StreamResponse:
        """Waits until the RPC has completed."""
        for _ in self._get_responses(timeout_s=timeout_s):
            pass

        assert self.status is not None
        return StreamResponse(self.status, list(self._responses))

    def _get_responses(
        self, *, count: int | None = None, timeout_s: OptionalTimeout
    ) -> Iterator:
        """Returns an iterator of stream responses.

        Args:
          count: Responses to read before returning; None reads all
          timeout_s: max time in seconds to wait between responses; 0 doesn't
              block, None blocks indefinitely
        """
        self._check_errors()

        if self.completed() and self._response_queue.empty():
            return

        if timeout_s is UseDefault.VALUE:
            timeout_s = self.default_timeout_s

        remaining = math.inf if count is None else count

        try:
            while remaining:
                response = self._response_queue.get(True, timeout_s)

                self._check_errors()

                if response is None:
                    return

                yield response
                remaining -= 1
        except queue.Empty:
            raise RpcTimeout(self._rpc, timeout_s)

    def cancel(self) -> bool:
        """Cancels the RPC; returns whether the RPC was active."""
        if self.completed():
            return False

        self.error = Status.CANCELLED
        return self._rpcs.send_cancel(self._rpc)

    def _check_errors(self) -> None:
        if self._callback_exception:
            raise self._callback_exception

        if self.error:
            raise RpcError(self._rpc, self.error)

    def _handle_response(self, response: Any) -> None:
        self._responses.append(response)
        self._response_queue.put(response)

        self._invoke_callback('on_next', response)

    def _handle_completion(self, status: Status) -> None:
        self.status = status
        self._response_queue.put(None)

        self._invoke_callback('on_completed', status)

    def _handle_error(self, error: Status) -> None:
        self.error = error
        self._response_queue.put(None)

        self._invoke_callback('on_error', error)

    def _invoke_callback(self, callback_name: str, arg: Any) -> None:
        """Invokes a user-provided callback function for an RPC event."""

        # Catch and log any exceptions from the user-provided callback so that
        # exceptions don't terminate the thread handling RPC packets.
        callback: Callable[[Call, Any], None] = getattr(self, callback_name)

        try:
            callback(self, arg)
        except Exception as callback_exception:  # pylint: disable=broad-except
            msg = (
                f'The {callback_name} callback ({callback}) for '
                f'{self._rpc} raised an exception'
            )
            _LOG.exception(msg)

            self._callback_exception = RuntimeError(msg)
            self._callback_exception.__cause__ = callback_exception

    def __enter__(self) -> Call:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cancel()

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.method})'


class UnaryCall(Call):
    """`Call` variant that tracks the a unary RPC call."""

    @property
    def response(self) -> Any:
        return self._responses[-1] if self._responses else None

    def wait(
        self, timeout_s: OptionalTimeout = UseDefault.VALUE
    ) -> UnaryResponse:
        return self._unary_wait(timeout_s)


class ServerStreamingCall(Call):
    """`Call` variant that tracks a server streaming RPC call."""

    @property
    def responses(self) -> Sequence:
        return self._responses

    def wait(
        self, timeout_s: OptionalTimeout = UseDefault.VALUE
    ) -> StreamResponse:
        return self._stream_wait(timeout_s)

    def get_responses(
        self,
        *,
        count: int | None = None,
        timeout_s: OptionalTimeout = UseDefault.VALUE,
    ) -> Iterator:
        return self._get_responses(count=count, timeout_s=timeout_s)

    def request_completion(self) -> None:
        """Sends client completion packet to server."""
        if not self.completed():
            self._rpcs.send_client_stream_end(self._rpc)

    def __iter__(self) -> Iterator:
        return self.get_responses()


class ClientStreamingCall(Call):
    """`Call` variant that tracks a client streaming RPC call."""

    @property
    def response(self) -> Any:
        return self._responses[-1] if self._responses else None

    def send(
        self, request_proto: Message | None = None, /, **request_fields
    ) -> None:
        """Sends client stream request to the server."""
        self._send_client_stream(request_proto, request_fields)

    def finish_and_wait(
        self,
        requests: Iterable[Message] = (),
        *,
        timeout_s: OptionalTimeout = UseDefault.VALUE,
    ) -> UnaryResponse:
        """Ends the client stream and waits for the RPC to complete."""
        self._finish_client_stream(requests)
        return self._unary_wait(timeout_s)


class BidirectionalStreamingCall(Call):
    """`Call` variant that tracks a bidirectional streaming RPC call."""

    @property
    def responses(self) -> Sequence:
        return self._responses

    def send(
        self, request_proto: Message | None = None, /, **request_fields
    ) -> None:
        """Sends a message to the server in the client stream."""
        self._send_client_stream(request_proto, request_fields)

    def finish_and_wait(
        self,
        requests: Iterable[Message] = (),
        *,
        timeout_s: OptionalTimeout = UseDefault.VALUE,
    ) -> StreamResponse:
        """Ends the client stream and waits for the RPC to complete."""
        self._finish_client_stream(requests)
        return self._stream_wait(timeout_s)

    def get_responses(
        self,
        *,
        count: int | None = None,
        timeout_s: OptionalTimeout = UseDefault.VALUE,
    ) -> Iterator:
        return self._get_responses(count=count, timeout_s=timeout_s)

    def __iter__(self) -> Iterator:
        return self.get_responses()
