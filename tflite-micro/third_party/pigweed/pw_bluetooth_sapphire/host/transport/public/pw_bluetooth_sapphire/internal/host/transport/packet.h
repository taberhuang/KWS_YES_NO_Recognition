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
#include <memory>

#include "pw_bluetooth_sapphire/internal/host/common/byte_buffer.h"
#include "pw_bluetooth_sapphire/internal/host/common/macros.h"
#include "pw_bluetooth_sapphire/internal/host/common/packet_view.h"

namespace bt::hci {

// A Packet is a move-only object that can be used to hold sent and received HCI
// packets. The Packet template is parameterized over the protocol packet header
// type.
//
// Instances of Packet cannot be created directly as the template does not
// specify the backing buffer, which should be provided by a subclass.
//
// Header-type-specific functionality can be provided in specializations of the
// Packet template.
//
// USAGE:
//
//   Each Packet consists of a PacketView into a buffer that actually stores the
//   data. A buffer should be provided in a subclass implementation. While the
//   buffer must be sufficiently large to store the packet, the packet contents
//   can be much smaller.
//
//     template <typename HeaderType, size_t BufferSize>
//     class FixedBufferPacket : public Packet<HeaderType> {
//      public:
//       void Init(size_t payload_size) {
//         this->init_view(MutablePacketView<HeaderType>(&buffer_,
//         payload_size));
//       }
//
//      private:
//       StaticByteBuffer<BufferSize> buffer_;
//     };
//
//     std::unique_ptr<Packet<MyHeaderType>> packet =
//         std::make_unique<FixedBufferPacket<MyHeaderType, 255>>(payload_size);
//
//   Use Packet::view() to obtain a read-only view into the packet contents:
//
//     auto foo = packet->view().header().some_header_field;
//
//   Use Packet::mutable_view() to obtain a mutable view into the packet, which
//   allows the packet contents and the size of the packet to be modified:
//
//     packet->mutable_view()->mutable_header()->some_header_field = foo;
//     packet->mutable_view()->set_payload_size(my_new_size);
//
//     // Copy data directly into the buffer.
//     auto mutable_bytes = packet->mutable_view()->mutable_bytes();
//     std::memcpy(mutable_bytes.mutable_data(), data, mutable_bytes.size());
//
// SPECIALIZATIONS:
//
//   Additional functionality that is specific to a protocol header type can be
//   provided in a specialization of the Packet template.
//
//     using MagicPacket = Packet<MagicHeader>;
//
//     template <>
//     class Packet<MagicHeader> : public PacketBase<MagicHeader, MagicPacket> {
//      public:
//       // Initializes packet with pancakes.
//       void InitPancakes();
//     };
//
//     // Create an instance of FixedBufferPacket declared above.
//     std::unique_ptr<MagicPacket> packet =
//         std::make_unique<FixedBufferPacket<MagicHeader, 255>>();
//     packet->InitPancakes();
//
//   This pattern is used by EventPacket, ACLDataPacket, and ScoDataPacket
//
// THREAD-SAFETY:
//
//   Packet is NOT thread-safe without external locking.

// PacketBase provides basic view functionality. Intended to be inherited by the
// Packet template and all of its specializations.
template <typename HeaderType, typename T>
class PacketBase {
 public:
  virtual ~PacketBase() = default;

  const PacketView<HeaderType>& view() const { return view_; }
  MutablePacketView<HeaderType>* mutable_view() { return &view_; }

 protected:
  explicit PacketBase(const MutablePacketView<HeaderType>& view)
      : view_(view) {}

 private:
  MutablePacketView<HeaderType> view_;

  BT_DISALLOW_COPY_AND_ASSIGN_ALLOW_MOVE(PacketBase);
};

// The basic Packet template. See control_packets.h and acl_data_packet.h
// for specializations that add functionality beyond that of PacketBase.
template <typename HeaderType>
class Packet : public PacketBase<HeaderType, Packet<HeaderType>> {
 protected:
  using PacketBase<HeaderType, Packet<HeaderType>>::PacketBase;
};

}  // namespace bt::hci
