// Copyright 2025 The Pigweed Authors
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

#include "pw_bluetooth_proxy/internal/recombiner.h"

#include <cstdint>
#include <mutex>
#include <optional>

#include "pw_bluetooth_proxy/internal/locked_l2cap_channel.h"
#include "pw_bluetooth_proxy/proxy_host.h"
#include "pw_bluetooth_proxy_private/test_utils.h"
#include "pw_containers/to_array.h"
#include "pw_span/cast.h"
#include "pw_status/status.h"
#include "pw_unit_test/framework.h"

namespace pw::bluetooth::proxy {
namespace {

using pw::containers::to_array;

class RecombinerTest : public ProxyHostTest {};

TEST_F(RecombinerTest, InactiveAtCreation) {
  Recombiner recombiner{Direction::kFromHost};

  EXPECT_FALSE(recombiner.IsActive());
}

TEST_F(RecombinerTest, Start) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
  pw::sync::Mutex mutex;
  LockedL2capChannel locked_channel{channel, std::unique_lock(mutex)};

  Recombiner recombiner{Direction::kFromHost};

  PW_TEST_EXPECT_OK(recombiner.StartRecombination(locked_channel, 8u, 0u));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());
}

TEST_F(RecombinerTest, GetLocalCid) {
  constexpr uint16_t kLocalCid = 0x20;
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  BasicL2capChannel channel =
      BuildBasicL2capChannel(proxy_, {.local_cid = kLocalCid});
  pw::sync::Mutex mutex;
  LockedL2capChannel locked_channel{channel, std::unique_lock(mutex)};

  Recombiner recombiner{Direction::kFromController};

  PW_TEST_EXPECT_OK(recombiner.StartRecombination(locked_channel, 8u, 0u));

  EXPECT_EQ(recombiner.local_cid(), kLocalCid);
}

TEST_F(RecombinerTest, EndWithChannel) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
  pw::sync::Mutex mutex;
  std::optional<LockedL2capChannel> locked_channel{
      LockedL2capChannel{channel, std::unique_lock(mutex)}};

  Recombiner recombiner{Direction::kFromController};

  PW_TEST_EXPECT_OK(recombiner.StartRecombination(*locked_channel, 8u, 0u));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  recombiner.EndRecombination(locked_channel);

  EXPECT_FALSE(recombiner.IsActive());
}

TEST_F(RecombinerTest, EndWithoutChannel) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  Recombiner recombiner{Direction::kFromController};

  {
    BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
    pw::sync::Mutex mutex;
    std::optional<LockedL2capChannel> locked_channel{
        LockedL2capChannel{channel, std::unique_lock(mutex)}};

    PW_TEST_EXPECT_OK(recombiner.StartRecombination(*locked_channel, 8u, 0u));
  }

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  std::optional<LockedL2capChannel> null_channel = std::nullopt;

  recombiner.EndRecombination(null_channel);

  EXPECT_FALSE(recombiner.IsActive());
}

TEST_F(RecombinerTest, WriteThenTake) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  Direction kDirection = Direction::kFromController;
  Recombiner recombiner{kDirection};
  BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
  pw::sync::Mutex mutex;
  std::optional<LockedL2capChannel> locked_channel{
      LockedL2capChannel{channel, std::unique_lock(mutex)}};

  PW_TEST_EXPECT_OK(recombiner.StartRecombination(*locked_channel, 8u, 0u));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  static constexpr std::array<uint8_t, 8> kExpectedData = {
      0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};

  // Write first chunk
  PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
      locked_channel, to_array<uint8_t>({0x11, 0x22, 0x33, 0x44})));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  // Write second chunk
  PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
      locked_channel, to_array<uint8_t>({0x55, 0x66, 0x77, 0x88})));

  // We have read all expected bytes.
  EXPECT_TRUE(recombiner.IsComplete());
  // We are no longer recombining.
  EXPECT_FALSE(recombiner.IsActive());

  multibuf::MultiBuf mbuf = Recombiner::TakeBuf(locked_channel, kDirection);
  EXPECT_TRUE(mbuf.IsContiguous());

  pw::span<uint8_t> span =
      pw::span_cast<uint8_t>(mbuf.ContiguousSpan().value());

  EXPECT_TRUE(std::equal(
      span.begin(), span.end(), kExpectedData.begin(), kExpectedData.end()));
}

TEST_F(RecombinerTest, WriteCompleteWithoutChannel) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  Recombiner recombiner{Direction::kFromController};

  {
    BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
    pw::sync::Mutex mutex;
    std::optional<LockedL2capChannel> locked_channel{
        LockedL2capChannel{channel, std::unique_lock(mutex)}};

    PW_TEST_EXPECT_OK(recombiner.StartRecombination(*locked_channel, 8u, 0u));

    EXPECT_TRUE(recombiner.IsActive());
    EXPECT_FALSE(recombiner.IsComplete());

    // Write first chunk
    PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
        locked_channel, to_array<uint8_t>({0x11, 0x22, 0x33, 0x44})));

    EXPECT_TRUE(recombiner.IsActive());
    EXPECT_FALSE(recombiner.IsComplete());
  }

  std::optional<LockedL2capChannel> null_channel = std::nullopt;

  // Write second chunk
  PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
      null_channel, to_array<uint8_t>({0x55, 0x66, 0x77, 0x88})));

  // We have read all expected bytes.
  EXPECT_TRUE(recombiner.IsComplete());
  // We are no longer recombining.
  EXPECT_FALSE(recombiner.IsActive());

  recombiner.EndRecombination(null_channel);

  EXPECT_FALSE(recombiner.IsActive());
}

TEST_F(RecombinerTest, RecombinedPduIsLargerThanSpecified) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  Recombiner recombiner{Direction::kFromController};

  BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
  pw::sync::Mutex mutex;
  std::optional<LockedL2capChannel> locked_channel{
      LockedL2capChannel{channel, std::unique_lock(mutex)}};

  PW_TEST_EXPECT_OK(recombiner.StartRecombination(*locked_channel, 8u, 0u));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  // Write first chunk
  PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
      locked_channel, to_array<uint8_t>({0x11, 0x22, 0x33, 0x44})));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  // Try to write too large a second chunk.
  EXPECT_EQ(
      recombiner.RecombineFragment(
          locked_channel, to_array<uint8_t>({0x55, 0x66, 0x77, 0x88, 0x99})),
      pw::Status::ResourceExhausted());

  // Should still not be complete.
  EXPECT_FALSE(recombiner.IsComplete());
  // Still waiting for a packet of the right size.
  EXPECT_TRUE(recombiner.IsActive());

  // Client ends recombination at this point.
  recombiner.EndRecombination(locked_channel);

  EXPECT_FALSE(recombiner.IsActive());
}
// Verify ability to have recombiner allocate extra header and to then claim
// it from the results of TakeBuf.
TEST_F(RecombinerTest, CanClaimExtraHeader) {
  ProxyHost proxy_{[]([[maybe_unused]] H4PacketWithHci&& packet) {},
                   []([[maybe_unused]] H4PacketWithH4&& packet) {},
                   0,
                   0};
  Direction kDirection = Direction::kFromController;
  Recombiner recombiner{kDirection};
  BasicL2capChannel channel = BuildBasicL2capChannel(proxy_, {});
  pw::sync::Mutex mutex;
  std::optional<LockedL2capChannel> locked_channel{
      LockedL2capChannel{channel, std::unique_lock(mutex)}};

  constexpr size_t kExtraHeaderSize = 4;
  PW_TEST_EXPECT_OK(
      recombiner.StartRecombination(*locked_channel, 8u, kExtraHeaderSize));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  static constexpr std::array<uint8_t, 8> kExpectedData = {
      0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};

  // Write first chunk
  PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
      locked_channel, to_array<uint8_t>({0x11, 0x22, 0x33, 0x44})));

  EXPECT_TRUE(recombiner.IsActive());
  EXPECT_FALSE(recombiner.IsComplete());

  // Write second chunk
  PW_TEST_EXPECT_OK(recombiner.RecombineFragment(
      locked_channel, to_array<uint8_t>({0x55, 0x66, 0x77, 0x88})));

  // We have read all expected bytes.
  EXPECT_TRUE(recombiner.IsComplete());
  // We are no longer recombining.
  EXPECT_FALSE(recombiner.IsActive());

  multibuf::MultiBuf mbuf = Recombiner::TakeBuf(locked_channel, kDirection);
  EXPECT_TRUE(mbuf.IsContiguous());

  pw::span<uint8_t> span =
      pw::span_cast<uint8_t>(mbuf.ContiguousSpan().value());

  EXPECT_EQ(span.size(), kExpectedData.size());
  EXPECT_TRUE(std::equal(
      span.begin(), span.end(), kExpectedData.begin(), kExpectedData.end()));

  // Verify that the extra header is there by claiming it.
  EXPECT_TRUE(mbuf.ClaimPrefix(kExtraHeaderSize));
  EXPECT_EQ(mbuf.size(), kExpectedData.size() + kExtraHeaderSize);
}

}  // namespace
}  // namespace pw::bluetooth::proxy
