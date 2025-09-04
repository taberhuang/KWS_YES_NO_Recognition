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

#include "pw_allocator/bucket_allocator.h"

#include "pw_malloc/config.h"
#include "pw_malloc/malloc.h"

namespace pw::malloc {

using BlockType = ::pw::allocator::BucketBlock<PW_MALLOC_BLOCK_OFFSET_TYPE>;
using BucketAllocator =
    ::pw::allocator::BucketAllocator<BlockType,
                                     PW_MALLOC_MIN_BUCKET_SIZE,
                                     PW_MALLOC_NUM_BUCKETS>;

void InitSystemAllocator(ByteSpan heap) {
  InitSystemAllocator<BucketAllocator>(heap);
}

Allocator* GetSystemAllocator() {
  static BucketAllocator allocator;
  return &allocator;
}

}  // namespace pw::malloc
