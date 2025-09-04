// Copyright 2024 The Pigweed Authors
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

#pragma once

#include "pw_bluetooth_proxy/gatt_notify_channel.h"

namespace pw::bluetooth::proxy {

class GattNotifyChannelInternal final : public GattNotifyChannel {
 public:
  // Should only be created by `ProxyHost` and tests.
  static pw::Result<GattNotifyChannel> Create(
      L2capChannelManager& l2cap_channel_manager,
      uint16_t connection_handle,
      uint16_t attribute_handle,
      ChannelEventCallback&& event_fn) {
    return GattNotifyChannel::Create(l2cap_channel_manager,
                                     connection_handle,
                                     attribute_handle,
                                     std::move(event_fn));
  }
};

}  // namespace pw::bluetooth::proxy
