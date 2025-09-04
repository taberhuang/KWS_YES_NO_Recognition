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

#include <cstdint>

#include "hardware/spi.h"
#include "pw_spi/initiator.h"

namespace pw::spi {

// Pico SDK implementation of the SPI Initiator
class Rp2040Initiator final : public Initiator {
 public:
  Rp2040Initiator(spi_inst_t* spi) : spi_(spi) {}

 private:
  // Implements pw::spi::Initiator:
  Status DoConfigure(const Config& config) override;
  Status DoWriteRead(ConstByteSpan write_buffer, ByteSpan read_buffer) override;

  spi_inst_t* spi_;
};

}  // namespace pw::spi
