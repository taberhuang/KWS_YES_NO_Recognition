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

#include "pw_bluetooth/rfcomm_frames.emb.h"

namespace pw::bluetooth::proxy {

// Calculates the Frame Check Sequence for an RFCOMM frame.
// See: ETSI TS 101 369 V7.1.0 (1999-11)
uint8_t RfcommFcs(const emboss::RfcommFrameView& rfcomm);

}  // namespace pw::bluetooth::proxy
