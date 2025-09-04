// Copyright 2023 The Pigweed Authors
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
#include "pw_bluetooth_sapphire/internal/host/common/byte_buffer.h"
#include "pw_bluetooth_sapphire/internal/host/hci-spec/protocol.h"
#include "pw_bluetooth_sapphire/internal/host/l2cap/frame_headers.h"
#include "pw_bluetooth_sapphire/internal/host/l2cap/l2cap_defs.h"
#include "pw_bluetooth_sapphire/internal/host/l2cap/types.h"

namespace bt::l2cap::testing {

// Signaling Packets

DynamicByteBuffer AclCommandRejectNotUnderstoodRsp(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle handle,
    ChannelId chan_id = kSignalingChannelId);
DynamicByteBuffer AclExtFeaturesInfoReq(l2cap::CommandId id,
                                        hci_spec::ConnectionHandle handle);
DynamicByteBuffer AclExtFeaturesInfoRsp(l2cap::CommandId id,
                                        hci_spec::ConnectionHandle handle,
                                        l2cap::ExtendedFeatures features);
DynamicByteBuffer AclFixedChannelsSupportedInfoReq(
    l2cap::CommandId id, hci_spec::ConnectionHandle handle);
DynamicByteBuffer AclFixedChannelsSupportedInfoRsp(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle handle,
    l2cap::FixedChannelsSupported chan_mask);
DynamicByteBuffer AclNotSupportedInformationResponse(
    l2cap::CommandId id, hci_spec::ConnectionHandle handle);
DynamicByteBuffer AclConfigReq(l2cap::CommandId id,
                               hci_spec::ConnectionHandle handle,
                               l2cap::ChannelId dst_id,
                               l2cap::ChannelParameters params);
DynamicByteBuffer AclConfigRsp(l2cap::CommandId id,
                               hci_spec::ConnectionHandle link_handle,
                               l2cap::ChannelId src_id,
                               l2cap::ChannelParameters params);
DynamicByteBuffer AclConnectionReq(l2cap::CommandId id,
                                   hci_spec::ConnectionHandle link_handle,
                                   l2cap::ChannelId src_id,
                                   l2cap::Psm psm);
DynamicByteBuffer AclConnectionRsp(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle link_handle,
    l2cap::ChannelId src_id,
    l2cap::ChannelId dst_id,
    ConnectionResult result = ConnectionResult::kSuccess);
DynamicByteBuffer AclDisconnectionReq(l2cap::CommandId id,
                                      hci_spec::ConnectionHandle link_handle,
                                      l2cap::ChannelId src_id,
                                      l2cap::ChannelId dst_id);
DynamicByteBuffer AclDisconnectionRsp(l2cap::CommandId id,
                                      hci_spec::ConnectionHandle link_handle,
                                      l2cap::ChannelId src_id,
                                      l2cap::ChannelId dst_id);
DynamicByteBuffer AclConnectionParameterUpdateReq(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle link_handle,
    uint16_t interval_min,
    uint16_t interval_max,
    uint16_t peripheral_latency,
    uint16_t timeout_multiplier);
DynamicByteBuffer AclConnectionParameterUpdateRsp(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle link_handle,
    ConnectionParameterUpdateResult result);

DynamicByteBuffer AclLeCreditBasedConnectionReq(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle link_handle,
    l2cap::Psm psm,
    l2cap::ChannelId cid,
    uint16_t mtu,
    uint16_t mps,
    uint16_t credits);

DynamicByteBuffer AclLeCreditBasedConnectionRsp(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle link_handle,
    l2cap::ChannelId cid,
    uint16_t mtu,
    uint16_t mps,
    uint16_t credits,
    LECreditBasedConnectionResult result);

DynamicByteBuffer AclFlowControlCreditInd(
    l2cap::CommandId id,
    hci_spec::ConnectionHandle link_handle,
    l2cap::ChannelId cid,
    uint16_t credits);

// S-Frame Packets

DynamicByteBuffer AclSFrame(hci_spec::ConnectionHandle link_handle,
                            l2cap::ChannelId channel_id,
                            internal::SupervisoryFunction function,
                            uint8_t receive_seq_num,
                            bool is_poll_request,
                            bool is_poll_response);

inline DynamicByteBuffer AclSFrameReceiverReady(
    hci_spec::ConnectionHandle link_handle,
    l2cap::ChannelId channel_id,
    uint8_t receive_seq_num,
    bool is_poll_request = false,
    bool is_poll_response = false) {
  return AclSFrame(link_handle,
                   channel_id,
                   internal::SupervisoryFunction::ReceiverReady,
                   receive_seq_num,
                   is_poll_request,
                   is_poll_response);
}

inline DynamicByteBuffer AclSFrameReceiverNotReady(
    hci_spec::ConnectionHandle link_handle,
    l2cap::ChannelId channel_id,
    uint8_t receive_seq_num,
    bool is_poll_request,
    bool is_poll_response) {
  return AclSFrame(link_handle,
                   channel_id,
                   internal::SupervisoryFunction::ReceiverNotReady,
                   receive_seq_num,
                   is_poll_request,
                   is_poll_response);
}

// I-Frame Packets

DynamicByteBuffer AclIFrame(hci_spec::ConnectionHandle link_handle,
                            l2cap::ChannelId channel_id,
                            uint8_t receive_seq_num,
                            uint8_t tx_seq,
                            bool is_poll_response,
                            const ByteBuffer& payload);
// K-Frame Packets

DynamicByteBuffer AclKFrame(hci_spec::ConnectionHandle link_handle,
                            l2cap::ChannelId channel_id,
                            const ByteBuffer& payload);

// B-Frame Packets

DynamicByteBuffer AclBFrame(hci_spec::ConnectionHandle link_handle,
                            l2cap::ChannelId channel_id,
                            const ByteBuffer& payload);

}  // namespace bt::l2cap::testing
