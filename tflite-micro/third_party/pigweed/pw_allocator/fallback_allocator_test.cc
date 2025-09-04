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

#include "pw_allocator/fallback_allocator.h"

#include "pw_allocator/testing.h"
#include "pw_status/status.h"
#include "pw_unit_test/framework.h"

namespace {

using ::pw::allocator::FallbackAllocator;
using ::pw::allocator::Layout;
using ::pw::allocator::test::AllocatorForTest;

// Test fixtures.

class FallbackAllocatorTest : public ::testing::Test {
 protected:
  constexpr static size_t kCapacity = 256;
  static_assert(sizeof(uintptr_t) >= AllocatorForTest<kCapacity>::kMinSize);

  FallbackAllocatorTest() : allocator_(primary_, secondary_) {}

  AllocatorForTest<kCapacity> primary_;
  AllocatorForTest<kCapacity> secondary_;
  FallbackAllocator allocator_;
};

// Unit tests.

TEST_F(FallbackAllocatorTest, GetCapacity) {
  pw::StatusWithSize capacity = allocator_.GetCapacity();
  EXPECT_EQ(capacity.status(), pw::OkStatus());
  EXPECT_EQ(capacity.size(), kCapacity);
}

TEST_F(FallbackAllocatorTest, AllocateFromPrimary) {
  Layout layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(layout);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(primary_.allocate_size(), layout.size());
  EXPECT_EQ(secondary_.allocate_size(), 0U);
}

TEST_F(FallbackAllocatorTest, AllocateFromSecondary) {
  primary_.Exhaust();
  Layout layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(layout);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(primary_.allocate_size(), layout.size());
  EXPECT_EQ(secondary_.allocate_size(), layout.size());
}

TEST_F(FallbackAllocatorTest, AllocateFailure) {
  Layout layout = Layout::Of<uintptr_t[0x10000]>();
  void* ptr = allocator_.Allocate(layout);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_EQ(primary_.allocate_size(), layout.size());
  EXPECT_EQ(secondary_.allocate_size(), layout.size());
}

TEST_F(FallbackAllocatorTest, DeallocateUsingPrimary) {
  Layout layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(layout);
  ASSERT_NE(ptr, nullptr);
  allocator_.Deallocate(ptr);
  EXPECT_EQ(primary_.deallocate_ptr(), ptr);
  EXPECT_EQ(primary_.deallocate_size(), layout.size());
  EXPECT_EQ(secondary_.deallocate_ptr(), nullptr);
  EXPECT_EQ(secondary_.deallocate_size(), 0U);
}

TEST_F(FallbackAllocatorTest, DeallocateUsingSecondary) {
  primary_.Exhaust();
  Layout layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(layout);
  ASSERT_NE(ptr, nullptr);
  allocator_.Deallocate(ptr);
  EXPECT_EQ(primary_.deallocate_ptr(), nullptr);
  EXPECT_EQ(primary_.deallocate_size(), 0U);
  EXPECT_EQ(secondary_.deallocate_ptr(), ptr);
  EXPECT_EQ(secondary_.deallocate_size(), layout.size());
}

TEST_F(FallbackAllocatorTest, ResizePrimary) {
  Layout old_layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(old_layout);
  ASSERT_NE(ptr, nullptr);

  size_t new_size = sizeof(uintptr_t[3]);
  EXPECT_TRUE(allocator_.Resize(ptr, new_size));
  EXPECT_EQ(primary_.resize_ptr(), ptr);
  EXPECT_EQ(primary_.resize_old_size(), old_layout.size());
  EXPECT_EQ(primary_.resize_new_size(), new_size);

  // Secondary should not be touched.
  EXPECT_EQ(secondary_.resize_ptr(), nullptr);
  EXPECT_EQ(secondary_.resize_old_size(), 0U);
  EXPECT_EQ(secondary_.resize_new_size(), 0U);
}

TEST_F(FallbackAllocatorTest, ResizePrimaryFailure) {
  Layout old_layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(old_layout);
  ASSERT_NE(ptr, nullptr);
  primary_.Exhaust();

  size_t new_size = sizeof(uintptr_t[3]);
  EXPECT_FALSE(allocator_.Resize(ptr, new_size));
  EXPECT_EQ(primary_.resize_ptr(), ptr);
  EXPECT_EQ(primary_.resize_old_size(), old_layout.size());
  EXPECT_EQ(primary_.resize_new_size(), new_size);

  // Secondary should not be touched.
  EXPECT_EQ(secondary_.resize_ptr(), nullptr);
  EXPECT_EQ(secondary_.resize_old_size(), 0U);
  EXPECT_EQ(secondary_.resize_new_size(), 0U);
}

TEST_F(FallbackAllocatorTest, ResizeSecondary) {
  primary_.Exhaust();
  Layout old_layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(old_layout);
  ASSERT_NE(ptr, nullptr);

  size_t new_size = sizeof(uintptr_t[3]);
  EXPECT_TRUE(allocator_.Resize(ptr, new_size));
  EXPECT_EQ(secondary_.resize_ptr(), ptr);
  EXPECT_EQ(secondary_.resize_old_size(), old_layout.size());
  EXPECT_EQ(secondary_.resize_new_size(), new_size);

  // Primary should not be touched.
  EXPECT_EQ(primary_.resize_ptr(), nullptr);
  EXPECT_EQ(primary_.resize_old_size(), 0U);
  EXPECT_EQ(primary_.resize_new_size(), 0U);
}

TEST_F(FallbackAllocatorTest, ResizeSecondaryFailure) {
  primary_.Exhaust();
  Layout old_layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(old_layout);
  ASSERT_NE(ptr, nullptr);
  secondary_.Exhaust();

  size_t new_size = sizeof(uintptr_t[3]);
  EXPECT_FALSE(allocator_.Resize(ptr, new_size));
  EXPECT_EQ(secondary_.resize_ptr(), ptr);
  EXPECT_EQ(secondary_.resize_old_size(), old_layout.size());
  EXPECT_EQ(secondary_.resize_new_size(), new_size);

  // Primary should not be touched.
  EXPECT_EQ(primary_.resize_ptr(), nullptr);
  EXPECT_EQ(primary_.resize_old_size(), 0U);
  EXPECT_EQ(primary_.resize_new_size(), 0U);
}

TEST_F(FallbackAllocatorTest, ReallocateSameAllocator) {
  Layout old_layout = Layout::Of<uintptr_t>();
  void* ptr1 = allocator_.Allocate(old_layout);
  ASSERT_NE(ptr1, nullptr);

  // Claim subsequent memeory to force reallocation.
  void* ptr2 = allocator_.Allocate(old_layout);
  ASSERT_NE(ptr2, nullptr);

  Layout new_layout = Layout::Of<uintptr_t[3]>();
  void* new_ptr = allocator_.Reallocate(ptr1, new_layout);
  EXPECT_NE(new_ptr, nullptr);
  EXPECT_EQ(primary_.deallocate_ptr(), ptr1);
  EXPECT_EQ(primary_.deallocate_size(), old_layout.size());
  EXPECT_EQ(primary_.allocate_size(), new_layout.size());
}

TEST_F(FallbackAllocatorTest, ReallocateDifferentAllocator) {
  Layout old_layout = Layout::Of<uintptr_t>();
  void* ptr = allocator_.Allocate(old_layout);
  primary_.Exhaust();

  Layout new_layout = Layout::Of<uintptr_t[3]>();
  void* new_ptr = allocator_.Reallocate(ptr, new_layout);
  EXPECT_NE(new_ptr, nullptr);
  EXPECT_EQ(primary_.deallocate_ptr(), ptr);
  EXPECT_EQ(primary_.deallocate_size(), old_layout.size());
  EXPECT_EQ(secondary_.allocate_size(), new_layout.size());
}

}  // namespace
