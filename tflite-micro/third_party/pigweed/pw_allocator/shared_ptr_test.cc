// Copyright 2025 The Pigweed Authors
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

#include "pw_allocator/shared_ptr.h"

// TODO(b/402489948): Remove when portable atomics are provided by `pw_atomic`.
#if PW_ALLOCATOR_HAS_ATOMICS

#include <cstddef>

#include "pw_allocator/allocator.h"
#include "pw_allocator/internal/counter.h"
#include "pw_allocator/testing.h"
#include "pw_unit_test/framework.h"

namespace {

using pw::allocator::test::Counter;
using pw::allocator::test::CounterSink;
using pw::allocator::test::CounterWithBuffer;

class SharedPtrTest : public pw::allocator::test::TestWithCounters {
 protected:
  pw::allocator::test::AllocatorForTest<256> allocator_;
};

TEST_F(SharedPtrTest, DefaultInitializationIsNullptr) {
  pw::SharedPtr<int> empty;
  EXPECT_EQ(empty.get(), nullptr);
}

TEST_F(SharedPtrTest, OperatorEqNullptrOnEmptySharedPtrSucceeds) {
  pw::SharedPtr<int> empty;
  EXPECT_TRUE(empty == nullptr);
  EXPECT_FALSE(empty != nullptr);
}

TEST_F(SharedPtrTest, OperatorEqNullptrAfterMakeSharedFails) {
  auto ptr = allocator_.MakeShared<Counter>(5u);
  EXPECT_NE(ptr.get(), nullptr);
  EXPECT_TRUE(ptr != nullptr);
  EXPECT_FALSE(ptr == nullptr);
}

TEST_F(SharedPtrTest, OperatorEqNullptrAfterMakeSharedNullptrTypeFails) {
  auto ptr = allocator_.MakeShared<std::nullptr_t>(nullptr);
  EXPECT_TRUE(ptr != nullptr);
  EXPECT_FALSE(ptr == nullptr);
  EXPECT_TRUE(*ptr == nullptr);
  EXPECT_FALSE(*ptr != nullptr);
}

TEST_F(SharedPtrTest, CopyConstructionIncreasesUseCount) {
  auto ptr1 = allocator_.MakeShared<Counter>(42u);
  pw::SharedPtr<Counter> ptr2(ptr1);
  EXPECT_EQ(ptr1.get(), ptr2.get());
  EXPECT_EQ(ptr1->value(), 42u);
  EXPECT_EQ(ptr2->value(), 42u);
  EXPECT_EQ(ptr1.use_count(), 2);
  EXPECT_EQ(ptr2.use_count(), 2);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 1U);
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
}

TEST_F(SharedPtrTest, CopyAssignmentIncreasesUseCount) {
  auto ptr1 = allocator_.MakeShared<Counter>(42u);
  pw::SharedPtr<Counter> ptr2;
  ptr2 = ptr1;
  EXPECT_EQ(ptr1.get(), ptr2.get());
  EXPECT_EQ(ptr1->value(), 42u);
  EXPECT_EQ(ptr2->value(), 42u);
  EXPECT_EQ(ptr1.use_count(), 2);
  EXPECT_EQ(ptr2.use_count(), 2);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 1U);
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
}

TEST_F(SharedPtrTest, MakeSharedForwardsConstructorArguments) {
  Counter counter(6);
  auto ptr = allocator_.MakeShared<CounterSink>(std::move(counter));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr->value(), 6u);
}

TEST_F(SharedPtrTest, MoveConstructsFromSubClassAndFreesTotalSize) {
  auto ptr = allocator_.MakeShared<CounterWithBuffer>();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 1U);

  size_t allocated = allocator_.allocate_size();
  EXPECT_GE(allocated, sizeof(CounterWithBuffer));

  pw::SharedPtr<Counter> base_ptr(std::move(ptr));
  EXPECT_EQ(base_ptr.use_count(), 1);
  EXPECT_EQ(allocator_.deallocate_size(), 0ul);

  // The size that is deallocated here should be the size of the larger
  // subclass, not the size of the smaller base class.
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
  base_ptr.reset();
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 1U);
  EXPECT_EQ(allocator_.deallocate_size(), allocated);
}

TEST_F(SharedPtrTest, MoveAssignsFromSubClassAndFreesTotalSize) {
  auto ptr = allocator_.MakeShared<CounterWithBuffer>();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 1U);

  size_t allocated = allocator_.allocate_size();
  EXPECT_GE(allocated, sizeof(CounterWithBuffer));

  pw::SharedPtr<Counter> base_ptr = std::move(ptr);
  EXPECT_EQ(base_ptr.use_count(), 1);
  EXPECT_EQ(allocator_.deallocate_size(), 0ul);

  // The size that is deallocated here should be the size of the larger
  // subclass, not the size of the smaller base class.
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
  base_ptr.reset();
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 1U);
  EXPECT_EQ(allocator_.deallocate_size(), allocated);
}

TEST_F(SharedPtrTest, ArrayConstruction) {
  auto ptr = allocator_.MakeShared<Counter[]>(5);
  EXPECT_NE(ptr.get(), nullptr);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 5u);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(ptr[i].value(), i);
  }
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
}

TEST_F(SharedPtrTest, SizeReturnsCorrectSize) {
  auto ptr_array = allocator_.MakeShared<int[]>(5);
  EXPECT_EQ(ptr_array.size(), 5U);
}

TEST_F(SharedPtrTest, ArrayConstructionWithAlignment) {
  auto ptr = allocator_.MakeShared<Counter[]>(5, 32);
  EXPECT_NE(ptr.get(), nullptr);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 5u);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(ptr[i].value(), i);
  }
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
  auto addr = reinterpret_cast<uintptr_t>(ptr.get());
  EXPECT_EQ(addr % 32, 0u);
}

TEST_F(SharedPtrTest, SizeReturnsCorrectSizeWhenAligned) {
  auto ptr_array = allocator_.MakeShared<int[]>(5, 32);
  EXPECT_EQ(ptr_array.size(), 5U);
}

TEST_F(SharedPtrTest, FreedExactlyOnce) {
  auto ptr1 = allocator_.MakeShared<Counter>(42u);
  EXPECT_EQ(ptr1.use_count(), 1);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 1U);

  pw::SharedPtr<Counter> ptr2 = ptr1;
  EXPECT_EQ(ptr1.use_count(), 2);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 0U);
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);

  {
    pw::SharedPtr<Counter> ptr3(std::move(ptr1));
    EXPECT_EQ(ptr3.use_count(), 2);
    EXPECT_EQ(Counter::TakeNumCtorCalls(), 0U);
    EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);

    ptr2 = nullptr;
    EXPECT_EQ(ptr3.use_count(), 1);
    EXPECT_EQ(Counter::TakeNumCtorCalls(), 0U);
    EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
  }

  EXPECT_EQ(Counter::TakeNumDtorCalls(), 1U);
}

TEST_F(SharedPtrTest, ArrayFreedExactlyOnce) {
  auto ptr1 = allocator_.MakeShared<Counter[]>(5);
  EXPECT_EQ(ptr1.use_count(), 1);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 5U);
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);

  pw::SharedPtr<Counter[]> ptr2 = ptr1;
  EXPECT_EQ(ptr1.use_count(), 2);
  EXPECT_EQ(Counter::TakeNumCtorCalls(), 0U);
  EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);

  {
    pw::SharedPtr<Counter[]> ptr3(std::move(ptr1));
    EXPECT_EQ(ptr3.use_count(), 2);
    EXPECT_EQ(Counter::TakeNumCtorCalls(), 0U);
    EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);

    ptr2 = nullptr;
    EXPECT_EQ(ptr3.use_count(), 1);
    EXPECT_EQ(Counter::TakeNumCtorCalls(), 0U);
    EXPECT_EQ(Counter::TakeNumDtorCalls(), 0U);
  }

  EXPECT_EQ(Counter::TakeNumDtorCalls(), 5U);
}

TEST_F(SharedPtrTest, OwnerBeforeProvidesPartialOrder) {
  auto ptr1 = allocator_.MakeShared<int>(111);
  auto ptr2 = allocator_.MakeShared<int>(222);
  auto ptr3 = ptr2;
  auto ptr4 = allocator_.MakeShared<int>(444);

  // Remain agnostic to allocation order.
  bool ascending = ptr1.owner_before(ptr2);

  // Reflexive.
  EXPECT_FALSE(ptr1.owner_before(ptr1));
  EXPECT_FALSE(ptr2.owner_before(ptr3));
  EXPECT_FALSE(ptr3.owner_before(ptr2));

  // Symmetric.
  EXPECT_NE(ptr2.owner_before(ptr1), ascending);

  // Transitive.
  EXPECT_EQ(ptr2.owner_before(ptr4), ascending);
  EXPECT_EQ(ptr1.owner_before(ptr4), ascending);
}

TEST_F(SharedPtrTest, CanswapWhenNeitherAreEmpty) {
  auto ptr1 = allocator_.MakeShared<Counter>(111u);
  auto ptr2 = allocator_.MakeShared<Counter>(222u);
  ptr1.swap(ptr2);
  EXPECT_EQ(ptr1->value(), 222u);
  EXPECT_EQ(ptr2->value(), 111u);
}

TEST_F(SharedPtrTest, CanswapWhenOneIsEmpty) {
  auto ptr1 = allocator_.MakeShared<Counter>(111u);
  pw::SharedPtr<Counter> ptr2;

  // ptr2 is empty.
  ptr1.swap(ptr2);
  EXPECT_EQ(ptr2->value(), 111u);
  EXPECT_EQ(ptr1, nullptr);

  // ptr1 is empty.
  ptr1.swap(ptr2);
  EXPECT_EQ(ptr1->value(), 111u);
  EXPECT_EQ(ptr2, nullptr);
}

TEST_F(SharedPtrTest, CanswapWhenBothAreEmpty) {
  pw::SharedPtr<Counter> ptr1;
  pw::SharedPtr<Counter> ptr2;
  ptr1.swap(ptr2);
  EXPECT_EQ(ptr1, nullptr);
  EXPECT_EQ(ptr2, nullptr);
}

}  // namespace

// TODO(b/402489948): Remove when portable atomics are provided by `pw_atomic`.
#endif  // PW_ALLOCATOR_HAS_ATOMICS
