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

#include <atomic>
#include <cstddef>

#include "pw_bytes/span.h"
#include "pw_chrono/system_clock.h"
#include "pw_result/result.h"
#include "pw_rpc/channel.h"
#include "pw_rpc_transport/internal/packet_buffer_queue.h"
#include "pw_rpc_transport/rpc_transport.h"
#include "pw_status/status.h"
#include "pw_sync/thread_notification.h"
#include "pw_thread/thread_core.h"

namespace pw::rpc {

// Override and provide to LocalRpcEgress to be notified of events.
class LocalRpcEgressTracker {
 public:
  virtual ~LocalRpcEgressTracker() = default;
  virtual void NoRpcServiceRegistryError() {}
  virtual void PacketSizeTooLarge([[maybe_unused]] size_t packet_size,
                                  [[maybe_unused]] size_t max_packet_size) {}
  virtual void EgressThreadNotRunningError() {}
  virtual void FailedToProcessPacket([[maybe_unused]] Status status) {}
  virtual void FailedToAccessPacket([[maybe_unused]] Status status) {}
  virtual void NoPacketAvailable([[maybe_unused]] Status status) {}
  virtual void PacketProcessed(
      [[maybe_unused]] ConstByteSpan packet,
      [[maybe_unused]] chrono::SystemClock::duration processing_duration) {}
};

// Handles RPC packets destined for the local receiver.
template <size_t kPacketQueueSize, size_t kMaxPacketSize>
class LocalRpcEgress : public RpcEgressHandler,
                       public ChannelOutput,
                       public thread::ThreadCore {
  using PacketBuffer =
      typename internal::PacketBufferQueue<kMaxPacketSize>::PacketBuffer;

 public:
  LocalRpcEgress(LocalRpcEgressTracker* tracker = nullptr)
      : ChannelOutput("RPC local egress"), tracker_(tracker) {}
  ~LocalRpcEgress() override { Stop(); }

  // Packet processor cannot be passed as a construction dependency as it would
  // create a circular dependency in the RPC transport configuration.
  void set_packet_processor(RpcPacketProcessor& packet_processor) {
    packet_processor_ = &packet_processor;
  }

  // Adds the packet to the transmit queue. The queue is continuously processed
  // by another thread. Implements RpcEgressHandler.
  Status SendRpcPacket(ConstByteSpan rpc_packet) override;

  // Implements ChannelOutput.
  Status Send(ConstByteSpan buffer) override { return SendRpcPacket(buffer); }

  // Once stopped, LocalRpcEgress will no longer process data and
  // will report errors on SendPacket().
  void Stop() {
    if (stopped_) {
      return;
    }
    stopped_ = true;
    // Unblock the processing thread and let it finish gracefully.
    process_queue_.release();
  }

 private:
  void Run() override;
  virtual void PacketQueued() {}
  virtual void PacketProcessed() {}

  LocalRpcEgressTracker* tracker_;
  sync::ThreadNotification process_queue_;
  RpcPacketProcessor* packet_processor_ = nullptr;
  std::array<PacketBuffer, kPacketQueueSize> packet_storage_;
  internal::PacketBufferQueue<kMaxPacketSize> packet_queue_{packet_storage_};
  internal::PacketBufferQueue<kMaxPacketSize> transmit_queue_ = {};
  std::atomic<bool> stopped_ = false;
};

template <size_t kPacketQueueSize, size_t kMaxPacketSize>
Status LocalRpcEgress<kPacketQueueSize, kMaxPacketSize>::SendRpcPacket(
    ConstByteSpan packet) {
  if (!packet_processor_) {
    if (tracker_) {
      tracker_->NoRpcServiceRegistryError();
    }
    return Status::FailedPrecondition();
  }
  if (packet.size() > kMaxPacketSize) {
    if (tracker_) {
      tracker_->PacketSizeTooLarge(packet.size(), kMaxPacketSize);
    }
    return Status::InvalidArgument();
  }
  if (stopped_) {
    if (tracker_) {
      tracker_->EgressThreadNotRunningError();
    }
    return Status::FailedPrecondition();
  }

  // Grab a free packet from the egress' pool, copy incoming frame and
  // push it into the queue for processing.
  auto packet_buffer = packet_queue_.Pop();
  if (!packet_buffer.ok()) {
    if (tracker_) {
      tracker_->NoPacketAvailable(packet_buffer.status());
    }
    return packet_buffer.status();
  }

  PW_TRY(packet_buffer.value()->CopyPacket(packet));

  transmit_queue_.Push(**packet_buffer);
  PacketQueued();

  process_queue_.release();

  if (stopped_) {
    if (tracker_) {
      tracker_->EgressThreadNotRunningError();
    }
    return Status::DataLoss();
  }

  return OkStatus();
}

template <size_t kPacketQueueSize, size_t kMaxPacketSize>
void LocalRpcEgress<kPacketQueueSize, kMaxPacketSize>::Run() {
  while (!stopped_) {
    // Wait until a client has signaled that there is data in the packet queue.
    process_queue_.acquire();

    while (true) {
      Result<PacketBuffer*> packet_buffer = transmit_queue_.Pop();
      if (!packet_buffer.ok()) {
        break;
      }
      Result<ConstByteSpan> packet = (*packet_buffer)->GetPacket();
      if (packet.ok()) {
        auto before = chrono::SystemClock::now();
        if (const auto status = packet_processor_->ProcessRpcPacket(*packet);
            !status.ok()) {
          if (tracker_) {
            tracker_->FailedToProcessPacket(status);
          }
        }
        if (tracker_) {
          tracker_->PacketProcessed(*packet,
                                    pw::chrono::SystemClock::now() - before);
        }
      } else {
        if (tracker_) {
          tracker_->FailedToAccessPacket(packet.status());
        }
      }
      packet_queue_.Push(**packet_buffer);
      PacketProcessed();
    }
  }
}

}  // namespace pw::rpc
