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

#include "pw_allocator/buddy_allocator.h"

#include <array>
#include <cstddef>

#include "pw_allocator/fuzzing.h"
#include "pw_unit_test/framework.h"

namespace {

// Test fixtures.

using BuddyAllocator = ::pw::allocator::BuddyAllocator<>;
using ::pw::allocator::Layout;

static constexpr size_t kBufferSize = 0x400;

// Unit tests.

TEST(BuddyAllocatorTest, ExplicitlyInit) {
  std::array<std::byte, kBufferSize> buffer;
  BuddyAllocator allocator;
  allocator.Init(buffer);
}

TEST(BuddyAllocatorTest, AllocateSmall) {
  std::array<std::byte, kBufferSize> buffer;
  BuddyAllocator allocator(buffer);
  void* ptr = allocator.Allocate(Layout(BuddyAllocator::kMinOuterSize / 2, 1));
  ASSERT_NE(ptr, nullptr);
  allocator.Deallocate(ptr);
}

TEST(BuddyAllocatorTest, AllocateAllBlocks) {
  std::array<std::byte, kBufferSize> buffer;
  BuddyAllocator allocator(buffer);
  pw::Vector<void*, kBufferSize / BuddyAllocator::kMinOuterSize> ptrs;
  while (true) {
    void* ptr = allocator.Allocate(Layout(1, 1));
    if (ptr == nullptr) {
      break;
    }
    ptrs.push_back(ptr);
  }
  while (!ptrs.empty()) {
    allocator.Deallocate(ptrs.back());
    ptrs.pop_back();
  }
}

TEST(BuddyAllocatorTest, AllocateLarge) {
  std::array<std::byte, kBufferSize> buffer;
  BuddyAllocator allocator(buffer);
  void* ptr = allocator.Allocate(Layout(48, 1));
  ASSERT_NE(ptr, nullptr);
  allocator.Deallocate(ptr);
}

TEST(BuddyAllocatorTest, AllocateExcessiveSize) {
  std::array<std::byte, kBufferSize> buffer;
  BuddyAllocator allocator(buffer);
  void* ptr = allocator.Allocate(Layout(786, 1));
  EXPECT_EQ(ptr, nullptr);
}

TEST(BuddyAllocatorTest, AllocateExcessiveAlignment) {
  std::array<std::byte, kBufferSize> buffer;
  BuddyAllocator allocator(buffer);
  void* ptr = allocator.Allocate(Layout(48, 32));
  EXPECT_EQ(ptr, nullptr);
}

// Fuzz tests.

using ::pw::allocator::test::DefaultArbitraryRequests;
using ::pw::allocator::test::Request;
using ::pw::allocator::test::TestHarness;

void NeverCrashes(const pw::Vector<Request>& requests) {
  static std::array<std::byte, kBufferSize> buffer;
  static BuddyAllocator allocator(buffer);
  static TestHarness fuzzer(allocator);
  fuzzer.HandleRequests(requests);
}

FUZZ_TEST(BucketBlockAllocatorFuzzTest, NeverCrashes)
    .WithDomains(DefaultArbitraryRequests());

}  // namespace
