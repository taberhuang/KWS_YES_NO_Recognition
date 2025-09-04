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

#include "pw_bluetooth_proxy/internal/l2cap_channel.h"

#include <mutex>
#include <optional>

#include "lib/stdcompat/utility.h"
#include "pw_bluetooth/emboss_util.h"
#include "pw_bluetooth/hci_data.emb.h"
#include "pw_bluetooth/hci_h4.emb.h"
#include "pw_bluetooth/l2cap_frames.emb.h"
#include "pw_bluetooth_proxy/internal/l2cap_channel_manager.h"
#include "pw_bluetooth_proxy/l2cap_channel_common.h"
#include "pw_log/log.h"
#include "pw_status/status.h"
#include "pw_status/try.h"

namespace pw::bluetooth::proxy {

void L2capChannel::MoveFields(L2capChannel& other) {
  // TODO: https://pwbug.dev/380504851 - Add tests for move operators.

  holder_ = other.holder_;
  other.holder_ = nullptr;
  if (holder_) {
    holder_->SetUnderlyingChannel(this);
  }

  state_ = other.state();
  connection_handle_ = other.connection_handle();
  transport_ = other.transport();
  local_cid_ = other.local_cid();
  remote_cid_ = other.remote_cid();
  payload_from_controller_fn_ = std::move(other.payload_from_controller_fn_);
  payload_from_host_fn_ = std::move(other.payload_from_host_fn_);
  rx_multibuf_allocator_ = other.rx_multibuf_allocator_;
  {
    std::lock_guard lock(tx_mutex_);
    std::lock_guard other_lock(other.tx_mutex_);
    payload_queue_ = std::move(other.payload_queue_);
    notify_on_dequeue_ = other.notify_on_dequeue_;
    l2cap_channel_manager_.DeregisterChannel(other);
    l2cap_channel_manager_.RegisterChannel(*this);
  }
  other.Undefine();
}

L2capChannel::L2capChannel(L2capChannel&& other)
    : l2cap_channel_manager_(other.l2cap_channel_manager_) {
  MoveFields(other);
}

L2capChannel& L2capChannel::operator=(L2capChannel&& other) {
  if (this != &other) {
    l2cap_channel_manager_.DeregisterChannel(*this);
    MoveFields(other);
  }
  return *this;
}

L2capChannel::~L2capChannel() {
  // Don't log dtor of moved-from channels.
  if (state_ != State::kUndefined) {
    PW_LOG_INFO(
        "btproxy: L2capChannel dtor - transport_: %u, connection_handle_ : "
        "%#x, local_cid_: %#x, remote_cid_: %#x, state_: %u",
        cpp23::to_underlying(transport_),
        connection_handle_,
        local_cid_,
        remote_cid_,
        cpp23::to_underlying(state_));
  }

  // Channel objects may outlive `ProxyHost`, but they are closed on
  // `ProxyHost` dtor, so this check will prevent a crash from trying to access
  // a destructed `L2capChannelManager`.
  if (state_ != State::kClosed) {
    // Note, DeregisterChannel locks channels_mutex_. This is used to block
    // channels being destroyed during Tx.
    // TODO: https://pwbug.dev/402454277 - Update comment after we no longer
    // use channels_mutex_ to block ChannelProxy dtor.
    l2cap_channel_manager_.DeregisterChannel(*this);
    ClearQueue();
  }
}

void L2capChannel::Stop() {
  PW_LOG_INFO(
      "btproxy: L2capChannel::Stop - transport_: %u, connection_handle_: %#x, "
      "local_cid_: %#x, remote_cid_: %#x, previous state_: %u",
      cpp23::to_underlying(transport_),
      connection_handle_,
      local_cid_,
      remote_cid_,
      cpp23::to_underlying(state_));

  PW_CHECK(state_ != State::kUndefined && state_ != State::kClosed);

  state_ = State::kStopped;
  ClearQueue();
}

void L2capChannel::Close() {
  l2cap_channel_manager_.DeregisterChannel(*this);
  InternalClose();
}

void L2capChannel::InternalClose(L2capChannelEvent event) {
  PW_LOG_INFO(
      "btproxy: L2capChannel::Close - transport_: %u, "
      "connection_handle_: %#x, local_cid_: %#x, remote_cid_: %#x, previous "
      "state_: %u",
      cpp23::to_underlying(transport_),
      connection_handle_,
      local_cid_,
      remote_cid_,
      cpp23::to_underlying(state_));

  PW_CHECK(state_ != State::kUndefined);
  if (state_ == State::kClosed) {
    return;
  }
  state_ = State::kClosed;

  ClearQueue();
  DoClose();
  SendEvent(event);
}

void L2capChannel::Undefine() { state_ = State::kUndefined; }

StatusWithMultiBuf L2capChannel::Write(pw::multibuf::MultiBuf&& payload) {
  Status status = DoCheckWriteParameter(payload);
  if (!status.ok()) {
    return {status, std::move(payload)};
  }
  StatusWithMultiBuf result = WriteLocked(std::move(payload));
  l2cap_channel_manager_.DrainChannelQueuesIfNewTx();
  return result;
}

StatusWithMultiBuf L2capChannel::WriteLocked(pw::multibuf::MultiBuf&& payload) {
  if (!payload.IsContiguous()) {
    return {Status::InvalidArgument(), std::move(payload)};
  }

  if (state() != State::kRunning) {
    return {Status::FailedPrecondition(), std::move(payload)};
  }

  return QueuePayload(std::move(payload));
}

Status L2capChannel::IsWriteAvailable() {
  if (state() != State::kRunning) {
    return Status::FailedPrecondition();
  }

  std::lock_guard lock(tx_mutex_);

  if (payload_queue_.full()) {
    notify_on_dequeue_ = true;
    return Status::Unavailable();
  }

  notify_on_dequeue_ = false;
  return OkStatus();
}

std::optional<H4PacketWithH4> L2capChannel::DequeuePacket() {
  std::optional<H4PacketWithH4> packet;
  bool should_notify = false;
  {
    std::lock_guard lock(tx_mutex_);
    packet = GenerateNextTxPacket();
    if (packet) {
      should_notify = notify_on_dequeue_;
      notify_on_dequeue_ = false;
    }
  }

  if (should_notify) {
    SendEvent(L2capChannelEvent::kWriteAvailable);
  }

  return packet;
}

StatusWithMultiBuf L2capChannel::QueuePayload(multibuf::MultiBuf&& buf) {
  PW_CHECK(state() == State::kRunning);
  PW_CHECK(buf.IsContiguous());

  {
    std::lock_guard lock(tx_mutex_);
    if (payload_queue_.full()) {
      notify_on_dequeue_ = true;
      return {Status::Unavailable(), std::move(buf)};
    }
    payload_queue_.push(std::move(buf));
  }

  ReportNewTxPacketsOrCredits();
  return {OkStatus(), std::nullopt};
}

void L2capChannel::PopFrontPayload() {
  PW_CHECK(!payload_queue_.empty());
  payload_queue_.pop();
}

ConstByteSpan L2capChannel::GetFrontPayloadSpan() const {
  PW_CHECK(!payload_queue_.empty());
  const multibuf::MultiBuf& buf = payload_queue_.front();
  std::optional<ConstByteSpan> span = buf.ContiguousSpan();
  PW_CHECK(span);
  return *span;
}

bool L2capChannel::PayloadQueueEmpty() const { return payload_queue_.empty(); }

bool L2capChannel::HandlePduFromController(pw::span<uint8_t> l2cap_pdu) {
  if (state() != State::kRunning) {
    PW_LOG_ERROR(
        "btproxy: L2capChannel::OnPduReceivedFromController on non-running "
        "channel. local_cid: %#x, remote_cid: %#x, state: %u",
        local_cid(),
        remote_cid(),
        cpp23::to_underlying(state()));
    SendEvent(L2capChannelEvent::kRxWhileStopped);
    return true;
  }
  return DoHandlePduFromController(l2cap_pdu);
}

L2capChannel::L2capChannel(
    L2capChannelManager& l2cap_channel_manager,
    multibuf::MultiBufAllocator* rx_multibuf_allocator,
    uint16_t connection_handle,
    AclTransportType transport,
    uint16_t local_cid,
    uint16_t remote_cid,
    OptionalPayloadReceiveCallback&& payload_from_controller_fn,
    OptionalPayloadReceiveCallback&& payload_from_host_fn)
    : l2cap_channel_manager_(l2cap_channel_manager),
      state_(State::kRunning),
      connection_handle_(connection_handle),
      transport_(transport),
      local_cid_(local_cid),
      remote_cid_(remote_cid),
      rx_multibuf_allocator_(rx_multibuf_allocator),
      payload_from_controller_fn_(std::move(payload_from_controller_fn)),
      payload_from_host_fn_(std::move(payload_from_host_fn)) {
  PW_LOG_INFO(
      "btproxy: L2capChannel ctor - transport_: %u, connection_handle_ : %u, "
      "local_cid_ : %#x, remote_cid_: %#x",
      cpp23::to_underlying(transport_),
      connection_handle_,
      local_cid_,
      remote_cid_);

  l2cap_channel_manager_.RegisterChannel(*this);
}

bool L2capChannel::AreValidParameters(uint16_t connection_handle,
                                      uint16_t local_cid,
                                      uint16_t remote_cid) {
  if (connection_handle > kMaxValidConnectionHandle) {
    PW_LOG_ERROR(
        "Invalid connection handle %#x. Maximum connection handle is 0x0EFF.",
        connection_handle);
    return false;
  }
  if (local_cid == 0 || remote_cid == 0) {
    PW_LOG_ERROR("L2CAP channel identifier 0 is not valid.");
    return false;
  }
  return true;
}

pw::Result<H4PacketWithH4> L2capChannel::PopulateTxL2capPacket(
    uint16_t data_length) {
  return PopulateL2capPacket(data_length);
}

namespace {

constexpr size_t H4SizeForL2capData(uint16_t data_length) {
  const size_t l2cap_packet_size =
      emboss::BasicL2capHeader::IntrinsicSizeInBytes() + data_length;
  const size_t acl_packet_size =
      emboss::AclDataFrameHeader::IntrinsicSizeInBytes() + l2cap_packet_size;
  return sizeof(emboss::H4PacketType) + acl_packet_size;
}

}  // namespace

bool L2capChannel::IsOkL2capDataLength(uint16_t data_length) {
  return H4SizeForL2capData(data_length) <=
         l2cap_channel_manager_.GetH4BuffSize();
}

pw::Result<H4PacketWithH4> L2capChannel::PopulateL2capPacket(
    uint16_t data_length) {
  const size_t l2cap_packet_size =
      emboss::BasicL2capHeader::IntrinsicSizeInBytes() + data_length;
  const size_t h4_packet_size = H4SizeForL2capData(data_length);

  pw::Result<H4PacketWithH4> h4_packet_res =
      l2cap_channel_manager_.GetAclH4Packet(h4_packet_size);
  if (!h4_packet_res.ok()) {
    return h4_packet_res.status();
  }
  H4PacketWithH4 h4_packet = std::move(h4_packet_res.value());
  h4_packet.SetH4Type(emboss::H4PacketType::ACL_DATA);

  PW_TRY_ASSIGN(
      auto acl,
      MakeEmbossWriter<emboss::AclDataFrameWriter>(h4_packet.GetHciSpan()));
  acl.header().handle().Write(connection_handle_);
  // TODO: https://pwbug.dev/360932103 - Support packet segmentation, so this
  // value will not always be FIRST_NON_FLUSHABLE.
  acl.header().packet_boundary_flag().Write(
      emboss::AclDataPacketBoundaryFlag::FIRST_NON_FLUSHABLE);
  acl.header().broadcast_flag().Write(
      emboss::AclDataPacketBroadcastFlag::POINT_TO_POINT);
  acl.data_total_length().Write(l2cap_packet_size);

  PW_TRY_ASSIGN(auto l2cap_header,
                MakeEmbossWriter<emboss::BasicL2capHeaderWriter>(
                    acl.payload().BackingStorage().data(),
                    emboss::BasicL2capHeader::IntrinsicSizeInBytes()));
  l2cap_header.pdu_length().Write(data_length);
  l2cap_header.channel_id().Write(remote_cid_);

  return h4_packet;
}

std::optional<uint16_t> L2capChannel::MaxL2capPayloadSize() const {
  std::optional<uint16_t> le_acl_data_packet_length =
      l2cap_channel_manager_.le_acl_data_packet_length();
  if (!le_acl_data_packet_length) {
    return std::nullopt;
  }

  uint16_t max_acl_data_size_based_on_h4_buffer =
      l2cap_channel_manager_.GetH4BuffSize() - sizeof(emboss::H4PacketType) -
      emboss::AclDataFrameHeader::IntrinsicSizeInBytes();
  uint16_t max_acl_data_size = std::min(max_acl_data_size_based_on_h4_buffer,
                                        *le_acl_data_packet_length);
  return max_acl_data_size - emboss::BasicL2capHeader::IntrinsicSizeInBytes();
}

void L2capChannel::ReportNewTxPacketsOrCredits() {
  l2cap_channel_manager_.ReportNewTxPacketsOrCredits();
}

void L2capChannel::DrainChannelQueuesIfNewTx() PW_LOCKS_EXCLUDED(tx_mutex_) {
  l2cap_channel_manager_.DrainChannelQueuesIfNewTx();
}

void L2capChannel::ClearQueue() {
  std::lock_guard lock(tx_mutex_);
  payload_queue_.clear();
}

//-------
//  Rx (protected)
//-------

bool L2capChannel::SendPayloadFromHostToClient(pw::span<uint8_t> payload) {
  return SendPayloadToClient(payload, payload_from_host_fn_);
}

bool L2capChannel::SendPayloadFromControllerToClient(
    pw::span<uint8_t> payload) {
  return SendPayloadToClient(payload, payload_from_controller_fn_);
}

bool L2capChannel::SendPayloadToClient(
    pw::span<uint8_t> payload, OptionalPayloadReceiveCallback& callback) {
  if (!callback) {
    return false;
  }

  if (!rx_multibuf_allocator_) {
    PW_LOG_ERROR(
        "btproxy: rx_multibuf_allocator_ is null so unable to create multibuf "
        "to pass to client. Will passthrough instead. "
        "connection: %#x, local_cid: %#x ",
        connection_handle(),
        local_cid());
    return false;
  }

  std::optional<multibuf::MultiBuf> buffer =
      rx_multibuf_allocator()->AllocateContiguous(payload.size());

  if (!buffer) {
    PW_LOG_ERROR(
        "btproxy: rx_multibuf_allocator_ is out of memory. So stopping "
        "channel and reporting it needs to be closed."
        "connection: %#x, local_cid: %#x ",
        connection_handle(),
        local_cid());
    StopAndSendEvent(L2capChannelEvent::kRxOutOfMemory);
    return true;
  }

  StatusWithSize status = buffer->CopyFrom(/*source=*/as_bytes(payload),
                                           /*position=*/0);
  PW_CHECK_OK(status);

  std::optional<multibuf::MultiBuf> client_multibuf =
      callback(std::move(*buffer));
  // If client returned multibuf to us, we drop it and indicate to caller that
  // packet should be forwarded. In the future when whole path is operating
  // with multibuf's, we could pass it back up to container to be forwarded.
  return !client_multibuf.has_value();
}

pw::Status L2capChannel::StartRecombinationBuf(Direction direction,
                                               size_t payload_size,
                                               size_t extra_header_size) {
  std::optional<multibuf::MultiBuf>& buf_optref =
      GetRecombinationBufOptRef(direction);
  PW_CHECK(!buf_optref.has_value());

  if (!rx_multibuf_allocator_) {
    // TODO: https://pwbug.dev/423695410 - Should eventually recombine for these
    // cases to allow channel to make handle/unhandle decision.
    PW_LOG_WARN(
        "Cannot start recombination without an rx_multibuf_allocator."
        "connection: %#x, local_cid: %#x ",
        connection_handle(),
        local_cid());
    return Status::FailedPrecondition();
  }

  buf_optref = rx_multibuf_allocator_->AllocateContiguous(payload_size +
                                                          extra_header_size);
  if (!buf_optref.has_value()) {
    return Status::ResourceExhausted();
  }

  buf_optref->DiscardPrefix(extra_header_size);

  return pw::OkStatus();
}

void L2capChannel::EndRecombinationBuf(Direction direction) {
  GetRecombinationBufOptRef(direction) = std::nullopt;
}

}  // namespace pw::bluetooth::proxy
