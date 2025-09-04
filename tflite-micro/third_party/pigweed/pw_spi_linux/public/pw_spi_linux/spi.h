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

#pragma once

#include <cstdint>
#include <optional>

#include "pw_bytes/span.h"
#include "pw_spi/chip_selector.h"
#include "pw_spi/initiator.h"
#include "pw_status/status.h"

namespace pw::spi {

// Linux userspace implementation of the SPI Initiator
class LinuxInitiator : public Initiator {
 public:
  // Configure the Linux Initiator object for use with a bus file descriptor,
  // and maximum bus-speed (in hz).
  constexpr LinuxInitiator(int fd, uint32_t max_speed_hz)
      : max_speed_hz_(max_speed_hz), fd_(fd) {}
  ~LinuxInitiator() override;

 private:
  uint32_t max_speed_hz_;
  int fd_;
  std::optional<Config> current_config_;

  // Implements pw::spi::Initiator
  Status DoConfigure(const Config& config) override;
  Status DoWriteRead(ConstByteSpan write_buffer, ByteSpan read_buffer) override;
};

// Linux userspace implementation of SPI ChipSelector
class LinuxChipSelector : public ChipSelector {
 public:
  Status SetActive(bool active) override;
};

}  // namespace pw::spi
