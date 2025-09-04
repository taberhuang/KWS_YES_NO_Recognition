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

#include "pw_bluetooth_proxy/internal/l2cap_signaling_channel.h"

namespace pw::bluetooth::proxy {

// Signaling channel for managing L2CAP channels over ACL-U logical links.
class L2capAclUSignalingChannel : public L2capSignalingChannel {
 public:
  explicit L2capAclUSignalingChannel(L2capChannelManager& l2cap_channel_manager,
                                     uint16_t connection_handle);

 private:
  bool OnCFramePayload(Direction direction,
                       pw::span<const uint8_t> cframe_payload) override;
};

}  // namespace pw::bluetooth::proxy
