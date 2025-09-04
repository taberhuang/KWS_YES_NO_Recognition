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

#include "pw_sync/interrupt_spin_lock.h"
#include "pw_sync/test/borrow_testing.h"
#include "pw_unit_test/framework.h"

using pw::sync::InterruptSpinLock;
using pw::sync::VirtualInterruptSpinLock;
using pw::sync::test::BorrowTest;

namespace {

extern "C" {

// Functions defined in interrupt_spin_lock_facade_test_c.c which call the API
// from C.
void pw_sync_InterruptSpinLock_CallLock(
    pw_sync_InterruptSpinLock* interrupt_spin_lock);
bool pw_sync_InterruptSpinLock_CallTryLock(
    pw_sync_InterruptSpinLock* interrupt_spin_lock);
void pw_sync_InterruptSpinLock_CallUnlock(
    pw_sync_InterruptSpinLock* interrupt_spin_lock);

}  // extern "C"

TEST(InterruptSpinLockTest, LockUnlock) {
  InterruptSpinLock interrupt_spin_lock;
  interrupt_spin_lock.lock();
  interrupt_spin_lock.unlock();
}

// TODO: b/235284163 - Add real concurrency tests once we have pw::thread on SMP
// systems given that uniprocessor systems cannot fail to acquire an ISL.

InterruptSpinLock static_interrupt_spin_lock;
TEST(InterruptSpinLockTest, LockUnlockStatic) {
  static_interrupt_spin_lock.lock();
  // TODO: b/235284163 - Ensure other cores fail to lock when its locked.
  // EXPECT_FALSE(static_interrupt_spin_lock.try_lock());
  static_interrupt_spin_lock.unlock();
}

TEST(InterruptSpinLockTest, TryLockUnlock) {
  InterruptSpinLock interrupt_spin_lock;
  const bool locked = interrupt_spin_lock.try_lock();
  EXPECT_TRUE(locked);
  if (locked) {
    // TODO: b/235284163 - Ensure other cores fail to lock when its locked.
    // EXPECT_FALSE(interrupt_spin_lock.try_lock());
    interrupt_spin_lock.unlock();
  }
}

// Unit tests for a `Borrowable`that uses a `InterruptSpinLock` as its lock.
using InterruptSpinLockBorrowTest = BorrowTest<InterruptSpinLock>;

TEST_F(InterruptSpinLockBorrowTest, Acquire) { TestAcquire(); }

TEST_F(InterruptSpinLockBorrowTest, ConstAcquire) { TestConstAcquire(); }

TEST_F(InterruptSpinLockBorrowTest, RepeatedAcquire) { TestRepeatedAcquire(); }

TEST_F(InterruptSpinLockBorrowTest, Moveable) { TestMoveable(); }

TEST_F(InterruptSpinLockBorrowTest, Copyable) { TestCopyable(); }

TEST_F(InterruptSpinLockBorrowTest, CopyableCovariant) {
  TestCopyableCovariant();
}

TEST_F(InterruptSpinLockBorrowTest, TryAcquireSuccess) {
  TestTryAcquireSuccess();
}

TEST_F(InterruptSpinLockBorrowTest, TryAcquireFailure) {
  TestTryAcquireFailure();
}

TEST(VirtualInterruptSpinLockTest, LockUnlock) {
  VirtualInterruptSpinLock interrupt_spin_lock;
  interrupt_spin_lock.lock();
  // TODO: b/235284163 - Ensure other cores fail to lock when its locked.
  // EXPECT_FALSE(interrupt_spin_lock.try_lock());
  interrupt_spin_lock.unlock();
}

VirtualInterruptSpinLock static_virtual_interrupt_spin_lock;
TEST(VirtualInterruptSpinLockTest, LockUnlockStatic) {
  static_virtual_interrupt_spin_lock.lock();
  // TODO: b/235284163 - Ensure other cores fail to lock when its locked.
  // EXPECT_FALSE(static_virtual_interrupt_spin_lock.try_lock());
  static_virtual_interrupt_spin_lock.unlock();
}

// Unit tests for a `Borrowable`that uses a `VirtualInterruptSpinLock` as its
// lock.
using VirtualInterruptSpinLockBorrowTest = BorrowTest<VirtualInterruptSpinLock>;

TEST_F(VirtualInterruptSpinLockBorrowTest, Acquire) { TestAcquire(); }

TEST_F(VirtualInterruptSpinLockBorrowTest, ConstAcquire) { TestConstAcquire(); }

TEST_F(VirtualInterruptSpinLockBorrowTest, RepeatedAcquire) {
  TestRepeatedAcquire();
}

TEST_F(VirtualInterruptSpinLockBorrowTest, Moveable) { TestMoveable(); }

TEST_F(VirtualInterruptSpinLockBorrowTest, Copyable) { TestCopyable(); }

TEST_F(VirtualInterruptSpinLockBorrowTest, CopyableCovariant) {
  TestCopyableCovariant();
}

TEST_F(VirtualInterruptSpinLockBorrowTest, TryAcquireSuccess) {
  TestTryAcquireSuccess();
}

TEST_F(VirtualInterruptSpinLockBorrowTest, TryAcquireFailure) {
  TestTryAcquireFailure();
}

TEST(InterruptSpinLockTest, LockUnlockInC) {
  InterruptSpinLock interrupt_spin_lock;
  pw_sync_InterruptSpinLock_CallLock(&interrupt_spin_lock);
  pw_sync_InterruptSpinLock_CallUnlock(&interrupt_spin_lock);
}

TEST(InterruptSpinLockTest, TryLockUnlockInC) {
  InterruptSpinLock interrupt_spin_lock;
  ASSERT_TRUE(pw_sync_InterruptSpinLock_CallTryLock(&interrupt_spin_lock));
  // TODO: b/235284163 - Ensure other cores fail to lock when its locked.
  // EXPECT_FALSE(pw_sync_InterruptSpinLock_CallTryLock(&interrupt_spin_lock));
  pw_sync_InterruptSpinLock_CallUnlock(&interrupt_spin_lock);
}

}  // namespace
