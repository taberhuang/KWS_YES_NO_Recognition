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

namespace chre {

template <typename AtomicType>
class AtomicBase {
 protected:
  std::atomic<AtomicType> atomic_;
};

using AtomicBoolBase = AtomicBase<bool>;
using AtomicUint32Base = AtomicBase<uint32_t>;

}  // namespace chre
