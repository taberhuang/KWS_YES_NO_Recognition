// Copyright 2021 The Pigweed Authors
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

#include "pw_sync/borrow.h"

#include "pw_sync/test/lock_testing.h"
#include "pw_sync/test/timed_borrow_testing.h"
#include "pw_unit_test/framework.h"

using namespace std::chrono_literals;

using pw::chrono::SystemClock;
using pw::sync::Borrowable;
using pw::sync::BorrowedPointer;
using pw::sync::VirtualBasicLockable;
using pw::sync::test::BaseObj;
using pw::sync::test::BorrowTest;
using pw::sync::test::Derived;
using pw::sync::test::FakeBasicLockable;
using pw::sync::test::FakeLockable;
using pw::sync::test::FakeTimedLockable;
using pw::sync::test::TimedBorrowTest;

namespace {

// We can't control the SystemClock's period configuration, so just in case
// duration cannot be accurately expressed in integer ticks, round the
// duration up.
constexpr SystemClock::duration kRoundedArbitraryDuration =
    SystemClock::for_at_least(42ms);

TEST(BorrowedPointerTest, MoveConstruct) {
  Derived derived(1);
  FakeBasicLockable lock;
  Borrowable<Derived, FakeBasicLockable> borrowable(derived, lock);
  BorrowedPointer<BaseObj, VirtualBasicLockable> borrowed(borrowable.acquire());
  EXPECT_EQ(borrowed->value(), 1);
}

TEST(BorrowedPointerTest, MoveAssign) {
  Derived derived(2);
  FakeBasicLockable lock;
  Borrowable<Derived, FakeBasicLockable> borrowable(derived, lock);
  BorrowedPointer<BaseObj, VirtualBasicLockable> borrowed =
      borrowable.acquire();
  EXPECT_EQ(borrowed->value(), 2);
}

// Unit tests for a `Borrowable`that uses a `FakeBasicLockable` as its lock.
using FakeBasicLockableBorrowTest = BorrowTest<FakeBasicLockable>;

TEST_F(FakeBasicLockableBorrowTest, Acquire) { TestAcquire(); }

TEST_F(FakeBasicLockableBorrowTest, ConstAcquire) { TestConstAcquire(); }

TEST_F(FakeBasicLockableBorrowTest, RepeatedAcquire) { TestRepeatedAcquire(); }

TEST_F(FakeBasicLockableBorrowTest, Moveable) { TestMoveable(); }

TEST_F(FakeBasicLockableBorrowTest, Copyable) { TestCopyable(); }

TEST_F(FakeBasicLockableBorrowTest, CopyableCovariant) {
  TestCopyableCovariant();
}

// Unit tests for a `Borrowable`that uses a `FakeLockable` as its lock.
using FakeLockableBorrowTest = BorrowTest<FakeLockable>;

TEST_F(FakeLockableBorrowTest, Acquire) { TestAcquire(); }

TEST_F(FakeLockableBorrowTest, ConstAcquire) { TestConstAcquire(); }

TEST_F(FakeLockableBorrowTest, RepeatedAcquire) { TestRepeatedAcquire(); }

TEST_F(FakeLockableBorrowTest, Moveable) { TestMoveable(); }

TEST_F(FakeLockableBorrowTest, Copyable) { TestCopyable(); }

TEST_F(FakeLockableBorrowTest, CopyableCovariant) { TestCopyableCovariant(); }

TEST_F(FakeLockableBorrowTest, TryAcquireSuccess) { TestTryAcquireSuccess(); }

TEST_F(FakeLockableBorrowTest, TryAcquireFailure) { TestTryAcquireFailure(); }

// Unit tests for a `Borrowable`that uses a `FakeTimedLockable` as its lock.
using FakeTimedLockableBorrowTest = TimedBorrowTest<FakeTimedLockable>;

TEST_F(FakeTimedLockableBorrowTest, Acquire) { TestAcquire(); }

TEST_F(FakeTimedLockableBorrowTest, ConstAcquire) { TestConstAcquire(); }

TEST_F(FakeTimedLockableBorrowTest, RepeatedAcquire) { TestRepeatedAcquire(); }

TEST_F(FakeTimedLockableBorrowTest, Moveable) { TestMoveable(); }

TEST_F(FakeTimedLockableBorrowTest, Copyable) { TestCopyable(); }

TEST_F(FakeTimedLockableBorrowTest, CopyableCovariant) {
  TestCopyableCovariant();
}

TEST_F(FakeTimedLockableBorrowTest, TryAcquireSuccess) {
  TestTryAcquireSuccess();
}

TEST_F(FakeTimedLockableBorrowTest, TryAcquireFailure) {
  TestTryAcquireFailure();
}

TEST_F(FakeTimedLockableBorrowTest, TryAcquireForSuccess) {
  TestTryAcquireForSuccess(kRoundedArbitraryDuration);
}

TEST_F(FakeTimedLockableBorrowTest, TryAcquireForFailure) {
  TestTryAcquireForFailure(kRoundedArbitraryDuration);
}

TEST_F(FakeTimedLockableBorrowTest, TryAcquireUntilSuccess) {
  TestTryAcquireUntilSuccess(kRoundedArbitraryDuration);
}

TEST_F(FakeTimedLockableBorrowTest, TryAcquireUntilFailure) {
  TestTryAcquireUntilFailure(kRoundedArbitraryDuration);
}

}  // namespace
