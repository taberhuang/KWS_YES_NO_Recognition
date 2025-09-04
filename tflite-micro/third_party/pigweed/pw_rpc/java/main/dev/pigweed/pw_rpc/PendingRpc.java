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

package dev.pigweed.pw_rpc;

import com.google.auto.value.AutoValue;
import java.util.Locale;

/** Represents an active RPC invocation: channel + service + method + call id. */
@AutoValue
abstract class PendingRpc {
  static PendingRpc create(Channel channel, Method method, int callId) {
    return new AutoValue_PendingRpc(channel, method, callId);
  }

  /** Creates a copy of the provided rpc with the given {@code callId}. */
  static PendingRpc withCallId(PendingRpc rpc, int callId) {
    return create(rpc.channel(), rpc.method(), callId);
  }

  public abstract Channel channel();

  public final Service service() {
    return method().service();
  }

  public abstract Method method();

  public abstract int callId();

  @Override
  public final String toString() {
    return String.format(
        Locale.ENGLISH, "PendingRpc[%s|channel=%d|callId=%d]", method(), channel().id(), callId());
  }

  public final boolean equalsExceptCallId(PendingRpc other) {
    return channel() == other.channel() && method() == other.method()
        && service() == other.service();
  }
}
