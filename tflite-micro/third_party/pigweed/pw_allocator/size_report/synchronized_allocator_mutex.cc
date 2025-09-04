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

#include "pw_allocator/best_fit.h"
#include "pw_allocator/size_report/size_report.h"
#include "pw_allocator/synchronized_allocator.h"
#include "pw_bloat/bloat_this_binary.h"
#include "pw_sync/mutex.h"

namespace pw::allocator::size_report {

int Measure() {
  volatile uint32_t mask = bloat::kDefaultMask;
  static BestFitAllocator<BlockType> base(GetBuffer());
  static SynchronizedAllocator<sync::Mutex> alloc(base);
  return MeasureAllocator(alloc, mask);
}

}  // namespace pw::allocator::size_report

int main() { return pw::allocator::size_report::Measure(); }
