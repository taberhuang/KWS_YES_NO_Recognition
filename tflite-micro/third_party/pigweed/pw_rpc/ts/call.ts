// Copyright 2021 The Pigweed Authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import { Status } from 'pigweedjs/pw_status';
import { Message } from 'google-protobuf';

import WaitQueue from './queue';

import { PendingCalls, Rpc } from './rpc_classes';

export type Callback = (a: any) => any;

class RpcError extends Error {
  status: Status;

  constructor(rpc: Rpc, status: Status) {
    let message = '';
    if (status === Status.NOT_FOUND) {
      message = ': the RPC server does not support this RPC';
    } else if (status === Status.DATA_LOSS) {
      message = ': an error occurred while decoding the RPC payload';
    }

    super(`${rpc.method.name} failed with error ${Status[status]}${message}`);
    this.status = status;
  }
}

class RpcTimeout extends Error {
  readonly rpc: Rpc;
  readonly timeoutMs: number;

  constructor(rpc: Rpc, timeoutMs: number) {
    super(`${rpc.method.name} timed out after ${timeoutMs} ms`);
    this.rpc = rpc;
    this.timeoutMs = timeoutMs;
  }
}

class Responses {
  private responses: Message[] = [];
  private totalResponses = 0;
  private readonly maxResponses: number;

  constructor(maxResponses: number) {
    this.maxResponses = maxResponses;
  }

  get length(): number {
    return Math.min(this.totalResponses, this.maxResponses);
  }

  push(response: Message): void {
    this.responses[this.totalResponses % this.maxResponses] = response;
    this.totalResponses += 1;
  }

  last(): Message | undefined {
    if (this.totalResponses === 0) {
      return undefined;
    }

    const lastIndex = (this.totalResponses - 1) % this.maxResponses;
    return this.responses[lastIndex];
  }

  getAll(): Message[] {
    if (this.totalResponses < this.maxResponses) {
      return this.responses.slice(0, this.totalResponses);
    }

    const splitIndex = this.totalResponses % this.maxResponses;
    return this.responses
      .slice(splitIndex)
      .concat(this.responses.slice(0, splitIndex));
  }
}

/** Represent an in-progress or completed RPC call. */
export class Call {
  // Responses ordered by arrival time. Undefined signifies stream completion.
  private responseQueue = new WaitQueue<Message | undefined>();
  protected responses: Responses;

  private rpcs: PendingCalls;
  rpc: Rpc;
  readonly callId: number;

  private onNext: Callback;
  private onCompleted: Callback;
  private onError: Callback;

  status?: Status;
  error?: Status;
  callbackException?: Error;

  constructor(
    rpcs: PendingCalls,
    rpc: Rpc,
    onNext: Callback,
    onCompleted: Callback,
    onError: Callback,
    maxResponses: number,
  ) {
    this.rpcs = rpcs;
    this.rpc = rpc;
    this.responses = new Responses(maxResponses);

    this.onNext = onNext;
    this.onCompleted = onCompleted;
    this.onError = onError;
    this.callId = rpcs.allocateCallId();
  }

  /* Calls the RPC. This must be called immediately after construction. */
  invoke(request?: Message, ignoreErrors = false): void {
    const previous = this.rpcs.sendRequest(
      this.rpc,
      this,
      ignoreErrors,
      request,
    );

    if (previous !== undefined && !previous.completed) {
      previous.handleError(Status.CANCELLED);
    }
  }

  get completed(): boolean {
    return this.status !== undefined || this.error !== undefined;
  }

  // eslint-disable-next-line @typescript-eslint/ban-types
  private invokeCallback(func: () => {}) {
    try {
      func();
    } catch (err: unknown) {
      if (err instanceof Error) {
        console.error(
          `An exception was raised while invoking a callback: ${err}`,
        );
        this.callbackException = err;
      }
      console.error(`Unexpected item thrown while invoking callback: ${err}`);
    }
  }

  handleResponse(response: Message): void {
    this.responses.push(response);
    this.responseQueue.push(response);
    this.invokeCallback(() => this.onNext(response));
  }

  handleCompletion(status: Status) {
    this.status = status;
    this.responseQueue.push(undefined);
    this.invokeCallback(() => this.onCompleted(status));
  }

  handleError(error: Status): void {
    this.error = error;
    this.responseQueue.push(undefined);
    this.invokeCallback(() => this.onError(error));
  }

  private async queuePopWithTimeout(
    timeoutMs: number,
  ): Promise<Message | undefined> {
    // eslint-disable-next-line no-async-promise-executor
    return new Promise(async (resolve, reject) => {
      let timeoutExpired = false;
      const timeoutWatcher = setTimeout(() => {
        timeoutExpired = true;
        reject(new RpcTimeout(this.rpc, timeoutMs));
      }, timeoutMs);
      const response = await this.responseQueue.shift();
      if (timeoutExpired) {
        this.responseQueue.unshift(response);
        return;
      }
      clearTimeout(timeoutWatcher);
      resolve(response);
    });
  }

  /**
   * Yields responses up the specified count as they are added.
   *
   * Throws an error as soon as it is received even if there are still
   * responses in the queue.
   *
   * Usage
   * ```
   * for await (const response of call.getResponses(5)) {
   *  console.log(response);
   * }
   * ```
   *
   * @param {number} count The number of responses to read before returning.
   *    If no value is specified, getResponses will block until the stream
   *    either ends or hits an error.
   * @param {number} timeout The number of milliseconds to wait for a response
   *    before throwing an error.
   */
  async *getResponses(
    count?: number,
    timeoutMs?: number,
  ): AsyncGenerator<Message> {
    this.checkErrors();

    if (this.completed && this.responseQueue.length == 0) {
      return;
    }

    let remaining = count ?? Number.POSITIVE_INFINITY;
    while (remaining > 0) {
      const response =
        timeoutMs === undefined
          ? await this.responseQueue.shift()
          : await this.queuePopWithTimeout(timeoutMs!);
      this.checkErrors();
      if (response === undefined) {
        return;
      }
      yield response!;
      remaining -= 1;
    }
  }

  cancel(): boolean {
    if (this.completed) {
      return false;
    }

    this.error = Status.CANCELLED;
    return this.rpcs.sendCancel(this.rpc, this.callId);
  }

  private checkErrors(): void {
    if (this.callbackException !== undefined) {
      throw this.callbackException;
    }
    if (this.error !== undefined) {
      throw new RpcError(this.rpc, this.error);
    }
  }

  protected async unaryWait(timeoutMs?: number): Promise<[Status, Message]> {
    for await (const response of this.getResponses(1, timeoutMs)) {
      // Do nothing.
    }
    if (this.status === undefined) {
      throw Error('Unexpected undefined status at end of stream');
    }
    if (this.responses.length !== 1) {
      throw Error(`Unexpected number of responses: ${this.responses.length}`);
    }
    return [this.status!, this.responses.last()!];
  }

  protected async streamWait(timeoutMs?: number): Promise<[Status, Message[]]> {
    for await (const response of this.getResponses(undefined, timeoutMs)) {
      // Do nothing.
    }
    if (this.status === undefined) {
      throw Error('Unexpected undefined status at end of stream');
    }
    return [this.status!, this.responses.getAll()];
  }

  protected sendClientStream(request: Message) {
    this.checkErrors();
    if (this.status !== undefined) {
      throw new RpcError(this.rpc, Status.FAILED_PRECONDITION);
    }
    this.rpcs.sendClientStream(this.rpc, request, this.callId);
  }

  protected finishClientStream(requests: Message[]) {
    for (const request of requests) {
      this.sendClientStream(request);
    }

    if (!this.completed) {
      this.rpcs.sendClientStreamEnd(this.rpc, this.callId);
    }
  }
}

/** Tracks the state of a unary RPC call. */
export class UnaryCall extends Call {
  /** Awaits the server response */
  async complete(timeoutMs?: number): Promise<[Status, Message]> {
    return await this.unaryWait(timeoutMs);
  }
}

/** Tracks the state of a client streaming RPC call. */
export class ClientStreamingCall extends Call {
  /** Gets the last server message, if it exists */
  get response(): Message | undefined {
    return this.responses.last();
  }

  /** Sends a message from the client. */
  send(request: Message) {
    this.sendClientStream(request);
  }

  /** Ends the client stream and waits for the RPC to complete. */
  async finishAndWait(
    requests: Message[] = [],
    timeoutMs?: number,
  ): Promise<[Status, Message]> {
    this.finishClientStream(requests);
    return await this.unaryWait(timeoutMs);
  }
}

/** Tracks the state of a server streaming RPC call. */
export class ServerStreamingCall extends Call {
  complete(timeoutMs?: number): Promise<[Status, Message[]]> {
    return this.streamWait(timeoutMs);
  }
}

/** Tracks the state of a bidirectional streaming RPC call. */
export class BidirectionalStreamingCall extends Call {
  /** Sends a message from the client. */
  send(request: Message) {
    this.sendClientStream(request);
  }

  /** Ends the client stream and waits for the RPC to complete. */
  async finishAndWait(
    requests: Array<Message> = [],
    timeoutMs?: number,
  ): Promise<[Status, Array<Message>]> {
    this.finishClientStream(requests);
    return await this.streamWait(timeoutMs);
  }
}
