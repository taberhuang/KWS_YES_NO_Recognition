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

#include <cstdint>
#include <variant>
#include <vector>

#include "pw_bluetooth/emboss_util.h"
#include "pw_bluetooth/hci_common.emb.h"
#include "pw_bluetooth/hci_data.emb.h"
#include "pw_bluetooth/hci_events.emb.h"
#include "pw_bluetooth/hci_h4.emb.h"
#include "pw_bluetooth/l2cap_frames.emb.h"
#include "pw_bluetooth_proxy/basic_l2cap_channel.h"
#include "pw_bluetooth_proxy/direction.h"
#include "pw_bluetooth_proxy/gatt_notify_channel.h"
#include "pw_bluetooth_proxy/h4_packet.h"
#include "pw_bluetooth_proxy/internal/logical_transport.h"
#include "pw_bluetooth_proxy/l2cap_channel_common.h"
#include "pw_bluetooth_proxy/l2cap_coc.h"
#include "pw_bluetooth_proxy/proxy_host.h"
#include "pw_bluetooth_proxy/rfcomm_channel.h"
#include "pw_containers/flat_map.h"
#include "pw_function/function.h"
#include "pw_multibuf/simple_allocator_for_test.h"
#include "pw_status/status.h"
#include "pw_status/try.h"
#include "pw_unit_test/framework.h"

namespace pw::bluetooth::proxy {

// ########## Util functions

struct AclFrameWithStorage {
  std::vector<uint8_t> storage;
  emboss::AclDataFrameWriter writer;

  static constexpr size_t kH4HeaderSize = 1;
  pw::span<uint8_t> h4_span() { return storage; }
  pw::span<uint8_t> hci_span() {
    return pw::span(storage).subspan(kH4HeaderSize);
  }
};

// Allocate storage and populate an ACL packet header with the given length.
Result<AclFrameWithStorage> SetupAcl(uint16_t handle, uint16_t l2cap_length);

struct BFrameWithStorage {
  AclFrameWithStorage acl;
  emboss::BFrameWriter writer;
};

Result<BFrameWithStorage> SetupBFrame(uint16_t handle,
                                      uint16_t channel_id,
                                      uint16_t bframe_len);

struct CFrameWithStorage {
  AclFrameWithStorage acl;
  emboss::CFrameWriter writer;
};

Result<CFrameWithStorage> SetupCFrame(uint16_t handle,
                                      uint16_t channel_id,
                                      uint16_t cframe_len);

struct KFrameWithStorage {
  AclFrameWithStorage acl;
  std::variant<emboss::FirstKFrameWriter, emboss::SubsequentKFrameWriter>
      writer;
};

// Size of sdu_length field in first K-frames.
inline constexpr uint8_t kSduLengthFieldSize = 2;

// Populate a KFrame that encodes a particular segment of `payload` based on the
// `mps`, or maximum PDU payload size of a segment. `segment_no` is the nth
// segment that would be generated based on the `mps`. The first segment is
// `segment_no == 0` and returns the `FirstKFrameWriter` variant in
// `KFrameWithStorage`.
//
// Returns PW_STATUS_OUT_OF_RANGE if a segment is requested beyond the last
// segment that would be generated based on `mps`.
Result<KFrameWithStorage> SetupKFrame(uint16_t handle,
                                      uint16_t channel_id,
                                      uint16_t mps,
                                      uint16_t segment_no,
                                      span<const uint8_t> payload);

// Populate passed H4 command buffer and return Emboss view on it.
template <typename EmbossT>
Result<EmbossT> CreateAndPopulateToControllerView(H4PacketWithH4& h4_packet,
                                                  emboss::OpCode opcode,
                                                  size_t parameter_total_size) {
  h4_packet.SetH4Type(emboss::H4PacketType::COMMAND);
  PW_TRY_ASSIGN(auto view, MakeEmbossWriter<EmbossT>(h4_packet.GetHciSpan()));
  view.header().opcode().Write(opcode);
  view.header().parameter_total_size().Write(parameter_total_size);
  return view;
}

// Populate passed H4 event buffer and return Emboss writer on it. Suitable for
// use with EmbossT types whose `SizeInBytes()` accurately represents
// the `parameter_total_size` that should be written (minus `EventHeader` size).
template <typename EmbossT>
Result<EmbossT> CreateAndPopulateToHostEventWriter(
    H4PacketWithHci& h4_packet,
    emboss::EventCode event_code,
    size_t parameter_total_size = EmbossT::SizeInBytes() -
                                  emboss::EventHeader::IntrinsicSizeInBytes()) {
  h4_packet.SetH4Type(emboss::H4PacketType::EVENT);
  PW_TRY_ASSIGN(auto view, MakeEmbossWriter<EmbossT>(h4_packet.GetHciSpan()));
  view.header().event_code().Write(event_code);
  view.header().parameter_total_size().Write(parameter_total_size);
  view.status().Write(emboss::StatusCode::SUCCESS);
  EXPECT_TRUE(view.IsComplete());
  return view;
}

// Send an LE_Read_Buffer_Size (V2) CommandComplete event to `proxy` to request
// the reservation of a number of LE ACL send credits.
Status SendLeReadBufferResponseFromController(
    ProxyHost& proxy,
    uint8_t num_credits_to_reserve,
    uint16_t le_acl_data_packet_length = 251);

Status SendReadBufferResponseFromController(ProxyHost& proxy,
                                            uint8_t num_credits_to_reserve);

// Send a Number_of_Completed_Packets event to `proxy` that reports each
// {connection handle, number of completed packets} entry provided.
template <size_t kNumConnections>
Status SendNumberOfCompletedPackets(
    ProxyHost& proxy,
    containers::FlatMap<uint16_t, uint16_t, kNumConnections>
        packets_per_connection) {
  std::array<
      uint8_t,
      emboss::NumberOfCompletedPacketsEvent::MinSizeInBytes() +
          kNumConnections *
              emboss::NumberOfCompletedPacketsEventData::IntrinsicSizeInBytes()>
      hci_arr;
  hci_arr.fill(0);
  H4PacketWithHci nocp_event{emboss::H4PacketType::EVENT, hci_arr};
  PW_TRY_ASSIGN(auto view,
                MakeEmbossWriter<emboss::NumberOfCompletedPacketsEventWriter>(
                    nocp_event.GetHciSpan()));
  view.header().event_code().Write(
      emboss::EventCode::NUMBER_OF_COMPLETED_PACKETS);
  view.header().parameter_total_size().Write(
      nocp_event.GetHciSpan().size() -
      emboss::EventHeader::IntrinsicSizeInBytes());
  view.num_handles().Write(kNumConnections);

  size_t i = 0;
  for (const auto& [handle, num_packets] : packets_per_connection) {
    view.nocp_data()[i].connection_handle().Write(handle);
    view.nocp_data()[i].num_completed_packets().Write(num_packets);
    ++i;
  }

  proxy.HandleH4HciFromController(std::move(nocp_event));
  return OkStatus();
}

// Send a Connection_Complete event to `proxy` indicating the provided
// `handle` has disconnected.
Status SendConnectionCompleteEvent(ProxyHost& proxy,
                                   uint16_t handle,
                                   emboss::StatusCode status);

// Send a LE_Connection_Complete event to `proxy` indicating the provided
// `handle` has disconnected.
Status SendLeConnectionCompleteEvent(ProxyHost& proxy,
                                     uint16_t handle,
                                     emboss::StatusCode status);

// Send a Disconnection_Complete event to `proxy` indicating the provided
// `handle` has disconnected.
Status SendDisconnectionCompleteEvent(
    ProxyHost& proxy,
    uint16_t handle,
    Direction direction = Direction::kFromController,
    bool successful = true);

struct L2capOptions {
  std::optional<MtuOption> mtu;
};

Status SendL2capConnectionReq(ProxyHost& proxy,
                              Direction direction,
                              uint16_t handle,
                              uint16_t source_cid,
                              uint16_t psm);

Status SendL2capConfigureReq(ProxyHost& proxy,
                             Direction direction,
                             uint16_t handle,
                             uint16_t destination_cid,
                             L2capOptions& l2cap_options);

Status SendL2capConfigureRsp(ProxyHost& proxy,
                             Direction direction,
                             uint16_t handle,
                             uint16_t local_cid,
                             emboss::L2capConfigurationResult result);

Status SendL2capConnectionRsp(ProxyHost& proxy,
                              Direction direction,
                              uint16_t handle,
                              uint16_t source_cid,
                              uint16_t destination_cid,

                              emboss::L2capConnectionRspResultCode result_code);

Status SendL2capDisconnectRsp(ProxyHost& proxy,
                              Direction direction,
                              AclTransportType transport,
                              uint16_t handle,
                              uint16_t source_cid,
                              uint16_t destination_cid);

/// Sends an L2CAP B-Frame.
///
/// This can be either a complete PDU (pdu_length == payload.size()) or an
/// initial fragment (pdu_length > payload.size()).
void SendL2capBFrame(ProxyHost& proxy,
                     uint16_t handle,
                     pw::span<const uint8_t> payload,
                     size_t pdu_length,
                     uint16_t channel_id);

/// Sends an ACL frame with CONTINUING_FRAGMENT boundary flag.
///
/// No L2CAP header is included.
void SendAclContinuingFrag(ProxyHost& proxy,
                           uint16_t handle,
                           pw::span<const uint8_t> payload);

// TODO: https://pwbug.dev/382783733 - Migrate to L2capChannelEvent callback.
struct CocParameters {
  uint16_t handle = 123;
  uint16_t local_cid = 234;
  uint16_t remote_cid = 456;
  uint16_t rx_mtu = 100;
  uint16_t rx_mps = 100;
  uint16_t rx_credits = 1;
  uint16_t tx_mtu = 100;
  uint16_t tx_mps = 100;
  uint16_t tx_credits = 1;
  Function<void(multibuf::MultiBuf&& payload)>&& receive_fn = nullptr;
  ChannelEventCallback&& event_fn = nullptr;
};

struct BasicL2capParameters {
  multibuf::MultiBufAllocator* rx_multibuf_allocator = nullptr;
  uint16_t handle = 123;
  uint16_t local_cid = 234;
  uint16_t remote_cid = 456;
  AclTransportType transport = AclTransportType::kLe;
  OptionalPayloadReceiveCallback&& payload_from_controller_fn = nullptr;
  OptionalPayloadReceiveCallback&& payload_from_host_fn = nullptr;
  ChannelEventCallback&& event_fn = nullptr;
};

struct GattNotifyChannelParameters {
  uint16_t handle = 0xAB;
  uint16_t attribute_handle = 0xBC;
  ChannelEventCallback&& event_fn = nullptr;
};

struct RfcommConfigParameters {
  uint16_t cid = 123;
  uint16_t max_information_length = 900;
  uint16_t credits = 10;
};

struct RfcommParameters {
  uint16_t handle = 123;
  RfcommConfigParameters rx_config = {
      .cid = 234, .max_information_length = 900, .credits = 10};
  RfcommConfigParameters tx_config = {
      .cid = 456, .max_information_length = 900, .credits = 10};
  uint8_t rfcomm_channel = 3;
};

// ########## Test Suites

class ProxyHostTest : public testing::Test {
 protected:
  pw::Result<L2capCoc> BuildCocWithResult(ProxyHost& proxy,
                                          CocParameters params);

  L2capCoc BuildCoc(ProxyHost& proxy, CocParameters params);

  Result<BasicL2capChannel> BuildBasicL2capChannelWithResult(
      ProxyHost& proxy, BasicL2capParameters params);

  BasicL2capChannel BuildBasicL2capChannel(ProxyHost& proxy,
                                           BasicL2capParameters params);

  Result<GattNotifyChannel> BuildGattNotifyChannelWithResult(
      ProxyHost& proxy, GattNotifyChannelParameters params);

  GattNotifyChannel BuildGattNotifyChannel(ProxyHost& proxy,
                                           GattNotifyChannelParameters params);

  RfcommChannel BuildRfcomm(
      ProxyHost& proxy,
      RfcommParameters params = {},
      Function<void(multibuf::MultiBuf&& payload)>&& receive_fn = nullptr,
      ChannelEventCallback&& event_fn = nullptr);

  template <typename T, size_t N>
  pw::multibuf::MultiBuf MultiBufFromSpan(span<T, N> buf) {
    std::optional<pw::multibuf::MultiBuf> multibuf =
        test_multibuf_allocator_.AllocateContiguous(buf.size());
    PW_ASSERT(multibuf.has_value());
    std::optional<ConstByteSpan> multibuf_span = multibuf->ContiguousSpan();
    PW_ASSERT(multibuf_span);
    PW_TEST_EXPECT_OK(multibuf->CopyFrom(as_bytes(buf)));
    return std::move(*multibuf);
  }

  template <typename T, size_t N>
  pw::multibuf::MultiBuf MultiBufFromArray(const std::array<T, N>& arr) {
    return MultiBufFromSpan(pw::span{arr});
  }

 private:
  // MultiBuf allocator for creating objects to pass to the system under
  // test (e.g. creating test packets to send to proxy host).
  pw::multibuf::test::SimpleAllocatorForTest</*kDataSizeBytes=*/2 * 1024,
                                             /*kMetaSizeBytes=*/2 * 1024>
      test_multibuf_allocator_{};

  // Default MultiBuf allocator to be passed to system under test (e.g.
  // to pass to AcquireL2capCoc).
  pw::multibuf::test::SimpleAllocatorForTest</*kDataSizeBytes=*/1024,
                                             /*kMetaSizeBytes=*/2 * 1024>
      sut_multibuf_allocator_{};
};

}  // namespace pw::bluetooth::proxy
