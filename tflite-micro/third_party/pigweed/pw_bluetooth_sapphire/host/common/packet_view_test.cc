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

#include "pw_bluetooth_sapphire/internal/host/common/packet_view.h"

#include <string>

#include "pw_bluetooth_sapphire/internal/host/common/byte_buffer.h"
#include "pw_bluetooth_sapphire/internal/host/common/macros.h"
#include "pw_bluetooth_sapphire/internal/host/testing/test_helpers.h"
#include "pw_unit_test/framework.h"

namespace bt {
namespace {

struct TestHeader {
  uint16_t field16;
  uint8_t field8;
} __attribute__((packed));

PW_MODIFY_DIAGNOSTICS_PUSH();
PW_MODIFY_DIAGNOSTIC_CLANG(ignored, "-Wzero-length-array");
struct TestPayload {
  uint8_t arg0;
  uint16_t arg1;
  uint8_t arg2[2];
  uint8_t arg3[0];
} __attribute__((packed));
PW_MODIFY_DIAGNOSTICS_POP();

TEST(PacketViewTest, EmptyPayload) {
  constexpr size_t kBufferSize = sizeof(TestHeader);

  StaticByteBuffer<kBufferSize> buffer;

  // Assign some values to the header portion.
  *reinterpret_cast<uint16_t*>(buffer.mutable_data()) = 512;
  buffer[2] = 255;

  PacketView<TestHeader> packet(&buffer);
  EXPECT_EQ(kBufferSize, packet.size());
  EXPECT_EQ(0u, packet.payload_size());
  EXPECT_EQ(0u, packet.payload_data().size());

  uint16_t field16_value = packet.header().field16;
  EXPECT_EQ(512, field16_value);
  EXPECT_EQ(255, packet.header().field8);

  // Verify the buffer contents.
  // TODO(armansito): This assumes that the packet is encoded in Bluetooth
  // network byte-order which is little-endian. For now we rely on the fact that
  // both ARM64 and x86-64 have little-endian encoding schemes to get away with
  // not explicitly encoding the entries. This is obviously wrong on other
  // architectures and will need to be addressed.
  constexpr std::array<uint8_t, kBufferSize> kExpected{{0x00, 0x02, 0xFF}};
  EXPECT_TRUE(ContainersEqual(kExpected, buffer));
}

TEST(PacketViewTest, NonEmptyPayload) {
  constexpr size_t kPayloadPadding = 4;
  constexpr size_t kPayloadSize = sizeof(TestPayload) + kPayloadPadding;
  constexpr size_t kBufferSize = sizeof(TestHeader) + kPayloadSize;

  StaticByteBuffer<kBufferSize> buffer;
  buffer.SetToZeros();

  MutablePacketView<TestHeader> packet(&buffer, kPayloadSize);
  EXPECT_EQ(kBufferSize, packet.size());
  EXPECT_EQ(kPayloadSize, packet.payload_size());
  EXPECT_NE(nullptr, packet.payload_data().data());

  auto payload = packet.mutable_payload<TestPayload>();
  EXPECT_NE(nullptr, payload);

  // Modify the payload.
  payload->arg0 = 127;
  payload->arg2[0] = 1;
  payload->arg2[1] = 2;
  memcpy(payload->arg3, "Test", 4);

  constexpr std::array<uint8_t, kBufferSize> kExpected{{
      0x00,
      0x00,
      0x00,  // header
      0x7F,  // arg0
      0x00,
      0x00,  // arg1
      0x01,
      0x02,  // arg2
      'T',
      'e',
      's',
      't'  // arg3
  }};
  EXPECT_TRUE(ContainersEqual(kExpected, buffer));
}

}  // namespace
}  // namespace bt
