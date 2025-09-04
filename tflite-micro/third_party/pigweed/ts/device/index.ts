// Copyright 2022 The Pigweed Authors
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

import { setPathOnObject } from './object_set';
import { Decoder, Encoder } from 'pigweedjs/pw_hdlc';
import {
  Client,
  Channel,
  ServiceClient,
  UnaryMethodStub,
  MethodStub,
  ServerStreamingMethodStub,
  Call,
} from 'pigweedjs/pw_rpc';
import { WebSerialTransport } from '../transport/web_serial_transport';
import { ProtoCollection } from 'pigweedjs/pw_protobuf_compiler';
import { Message } from 'google-protobuf';

function protoFieldToMethodName(fieldName: string) {
  return fieldName.split('_').map(titleCase).join('');
}
function titleCase(title: string) {
  return title.charAt(0).toUpperCase() + title.slice(1);
}

interface RPCUnderlyingSource extends UnderlyingSource {
  call?: Call;
}

export class RPCReadableStream<R = any> extends ReadableStream<R> {
  constructor(private underlyingSource: RPCUnderlyingSource) {
    super(underlyingSource);
  }

  get call(): Call {
    return this.underlyingSource.call!;
  }

  override cancel(): Promise<void> {
    this.call.cancel();
    return Promise.resolve();
  }
}

export class Device {
  private protoCollection: ProtoCollection;
  private transport: WebSerialTransport;
  private decoder: Decoder;
  private encoder: Encoder;
  private rpcAddress: number;
  private nameToMethodArgumentsMap: any;
  client: Client;
  rpcs: any;

  constructor(
    protoCollection: ProtoCollection,
    transport: WebSerialTransport = new WebSerialTransport(),
    channel = 1,
    rpcAddress = 82,
  ) {
    this.transport = transport;
    this.rpcAddress = rpcAddress;
    this.protoCollection = protoCollection;
    this.decoder = new Decoder();
    this.encoder = new Encoder();
    this.nameToMethodArgumentsMap = {};
    const channels = [
      new Channel(channel, (bytes) => {
        const hdlcBytes = this.encoder.uiFrame(this.rpcAddress, bytes);
        this.transport.sendChunk(hdlcBytes);
      }),
    ];
    this.client = Client.fromProtoSet(channels, this.protoCollection);

    this.setupRpcs();
  }

  async connect() {
    await this.transport.connect();
    this.transport.chunks.subscribe((item) => {
      const decoded = this.decoder.process(item);
      for (const frame of decoded) {
        if (frame.address === this.rpcAddress) {
          this.client.processPacket(frame.data);
        }
      }
    });
  }

  getMethodArguments(fullPath: string) {
    return this.nameToMethodArgumentsMap[fullPath];
  }

  private setupRpcs() {
    const rpcMap = {};
    const channel = this.client.channel()!;
    const servicesKeys = Array.from(channel.services.keys());
    servicesKeys.forEach((serviceKey) => {
      setPathOnObject(
        rpcMap,
        serviceKey,
        this.mapServiceMethods(channel.services.get(serviceKey)!),
      );
    });
    this.rpcs = rpcMap;
  }

  private mapServiceMethods(service: ServiceClient) {
    const methodMap: { [index: string]: any } = {};
    const methodKeys = Array.from(service.methodsByName.keys());
    methodKeys
      .filter(
        (method: any) =>
          service.methodsByName.get(method) instanceof UnaryMethodStub ||
          service.methodsByName.get(method) instanceof
            ServerStreamingMethodStub,
      )
      .forEach((key) => {
        const fn = this.createMethodWrapper(service.methodsByName.get(key)!);
        methodMap[key] = fn;
      });
    return methodMap;
  }

  private createMethodWrapper(realMethod: MethodStub) {
    if (realMethod instanceof UnaryMethodStub) {
      return this.createUnaryMethodWrapper(realMethod);
    } else if (realMethod instanceof ServerStreamingMethodStub) {
      return this.createServerStreamingMethodWrapper(realMethod);
    }
    throw new Error(`Unknown method: ${realMethod}`);
  }

  private createUnaryMethodWrapper(realMethod: UnaryMethodStub) {
    const call = async (request: Message, timeout?: number) => {
      return await realMethod.call(request, timeout);
    };
    const createRequest = () => {
      return new realMethod.method.requestType();
    };
    return { call, createRequest };
  }

  private createServerStreamingMethodWrapper(
    realMethod: ServerStreamingMethodStub,
  ) {
    const call = (request: Message) => {
      const source: RPCUnderlyingSource = {
        start(controller: ReadableStreamDefaultController) {
          this.call = realMethod.invoke(
            request,
            (msg) => {
              controller.enqueue(msg);
            },
            () => {
              controller.close();
            },
          );
        },
        cancel() {
          this.call!.cancel();
        },
      };
      return new RPCReadableStream<Message>(source);
    };
    const createRequest = () => {
      return new realMethod.method.requestType();
    };
    return { call, createRequest };
  }
}
