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

#include "pw_malloc/config.h"
#include "pw_malloc/malloc.h"

namespace pw::malloc {

using BlockType = ::pw::allocator::BestFitBlock<PW_MALLOC_BLOCK_OFFSET_TYPE>;
using BestFitAllocator = ::pw::allocator::BestFitAllocator<BlockType>;

void InitSystemAllocator(ByteSpan heap) {
  InitSystemAllocator<BestFitAllocator>(heap);
}

Allocator* GetSystemAllocator() {
  static BestFitAllocator allocator;
  return &allocator;
}

}  // namespace pw::malloc
