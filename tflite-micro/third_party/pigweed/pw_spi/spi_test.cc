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
#include "pw_spi/chip_selector.h"
#include "pw_spi/device.h"
#include "pw_spi/initiator.h"
#include "pw_spi/responder.h"
#include "pw_status/status.h"
#include "pw_sync/borrow.h"
#include "pw_sync/mutex.h"
#include "pw_unit_test/framework.h"

namespace pw::spi {
namespace {

constexpr pw::spi::Config kConfig = {
    .polarity = ClockPolarity::kActiveHigh,
    .phase = ClockPhase::kFallingEdge,
    .bits_per_word = BitsPerWord(8),
    .bit_order = BitOrder::kMsbFirst,
};

class SpiTestDevice : public ::testing::Test {
 public:
  SpiTestDevice()
      : initiator_(),
        chip_selector_(),
        initiator_lock_(),
        borrowable_initiator_(initiator_, initiator_lock_),
        device_(borrowable_initiator_, kConfig, chip_selector_) {}

 private:
  // Stub SPI Initiator/ChipSelect objects, used to exercise public API surface.
  class TestInitiator : public Initiator {
   private:
    Status DoConfigure(const Config& /*config */) override {
      return OkStatus();
    }
    Status DoWriteRead(ConstByteSpan /* write_buffer */,
                       ByteSpan /* read_buffer */) override {
      return OkStatus();
    }
  };

  class TestChipSelector : public ChipSelector {
   public:
    Status SetActive(bool /*active*/) override { return OkStatus(); }
  };

  TestInitiator initiator_;
  TestChipSelector chip_selector_;
  sync::VirtualMutex initiator_lock_;
  sync::Borrowable<Initiator> borrowable_initiator_;
  Device device_;
};

class SpiResponderTestDevice : public ::testing::Test {
 public:
  SpiResponderTestDevice() : responder_() {}

 private:
  // Stub SPI Responder, used to exercise public API surface.
  class TestResponder : public Responder {
   private:
    void DoSetCompletionHandler(
        Function<void(ByteSpan, Status)> /* callback */) override {}
    Status DoWriteReadAsync(ConstByteSpan /* tx_data */,
                            ByteSpan /* rx_data */) override {
      return OkStatus();
    }
    void DoCancel() override {}
  };

  TestResponder responder_;
};

// Simple test ensuring the SPI HAL compiles
TEST_F(SpiTestDevice, CompilationSucceeds) {
  // arrange
  // act
  // assert
  EXPECT_TRUE(true);
}

// Simple test ensuring the SPI Responder HAL compiles
TEST_F(SpiResponderTestDevice, CompilationSucceeds) { EXPECT_TRUE(true); }

// Config tests
static_assert(kConfig == kConfig);
static_assert(!(kConfig != kConfig));

TEST(Config, Equals) {
  Config lhs = kConfig;
  Config rhs = kConfig;
  EXPECT_EQ(lhs, rhs);
  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(Config, NotEquals) {
  Config lhs = kConfig;
  Config rhs = {
      .polarity = ClockPolarity::kActiveLow,  // different
      .phase = ClockPhase::kFallingEdge,
      .bits_per_word = BitsPerWord(8),
      .bit_order = BitOrder::kMsbFirst,
  };
  EXPECT_NE(lhs, rhs);
  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

}  // namespace
}  // namespace pw::spi
