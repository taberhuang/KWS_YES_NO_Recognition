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
#include <lib/fit/function.h>

#include "pw_bluetooth_sapphire/internal/host/common/uint256.h"
#include "pw_bluetooth_sapphire/internal/host/hci/connection.h"
#include "pw_bluetooth_sapphire/internal/host/sm/error.h"
#include "pw_bluetooth_sapphire/internal/host/sm/smp.h"
#include "pw_bluetooth_sapphire/internal/host/sm/types.h"

namespace bt::sm {

// Pure abstract interface to be implemented by classes that execute
// Authentication Stage 1 of Phase 2 of SMP Secure Connections. The owning class
// will use the abstract OnPairingConfirm and OnPairingRandom methods of this
// class to handle any inbound Pairing Confirm / Random values. Concrete Stage 1
// classes are responsible for tracking the state associated with their Pairing
// Method. See spec V5.0 Vol. 3 Part H 2.3.5.6.2-4 (aka "the spec section") for
// more details.
class ScStage1 {
 public:
  // This object contains the values generated/exchanged during Stage 1 which
  // are needed to finish SC pairing. An ScStage1 returns an |Output| through
  // its callback upon successful completion.
  struct Output {
    bool operator==(const Output& other) const {
      return initiator_r == other.initiator_r &&
             responder_r == other.responder_r &&
             initiator_rand == other.initiator_rand &&
             responder_rand == other.responder_rand;
    }

    // `ra` in the spec, associated with the initiator. Used to generate DHKey
    // check E in SC Phase 2 Stage 2. Meaning depends on the pairing method
    // used, see "the spec section" for details.
    UInt128 initiator_r;

    // `rb` in the spec, associated with the responder. Used to generate DHKey
    // check E in SC Phase 2 Stage 2. Meaning depends on the pairing method
    // used, see "the spec section" for details.
    UInt128 responder_r;

    // 'Na' in the spec; the Pairing Random value sent by the initiator. Used
    // during SC Phase 2 Stage 2 to generate the MacKey/LTK and the DHKey check
    // E values.
    UInt128 initiator_rand;

    // 'Na' in the spec; the Pairing Random value sent by the initiator. Used
    // during SC Phase 2 Stage 2 to generate the MacKey/LTK and the DHKey check
    // E values.
    UInt128 responder_rand;
  };

  // Used by Stage 1 classes to notify their owning class that they have
  // finished. A successful Stage 1 notifies its owner with `Output`, or that it
  // has failed due to `ErrorCode`.
  using Stage1CompleteCallback =
      fit::function<void(fit::result<ErrorCode, Output>)>;

  virtual void Run() = 0;
  virtual void OnPairingConfirm(PairingConfirmValue) = 0;
  virtual void OnPairingRandom(PairingRandomValue) = 0;
  virtual ~ScStage1() = default;
};

}  // namespace bt::sm
