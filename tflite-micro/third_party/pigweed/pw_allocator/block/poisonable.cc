// Copyright 2020 The Pigweed Authors
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

#include "pw_allocator/block/poisonable.h"

#include "pw_assert/check.h"

namespace pw::allocator::internal {

// TODO: b/234875269 - Add stack tracing to locate which call to the heap
// operation caused the corruption in the methods below.

void CheckPoisonedWhileInUse(const void* block, bool is_free) {
  if constexpr (Hardening::kIncludesDebugChecks) {
    PW_CHECK(is_free,
             "A block (%p) is corrupted: it is marked as poisoned while in use",
             block);
  }
}

void CheckPoisonCorrupted(const void* block, bool pattern_is_intact) {
  if constexpr (Hardening::kIncludesDebugChecks) {
    PW_CHECK(pattern_is_intact,
             "A block (%p) is corrupted: its has been modified while free",
             block);
  }
}

}  // namespace pw::allocator::internal
