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

#include "pw_async2/coro.h"

#include "pw_allocator/null_allocator.h"
#include "pw_allocator/testing.h"
#include "pw_async2/dispatcher_base.h"
#include "pw_status/status.h"

namespace {

using ::pw::OkStatus;
using ::pw::Result;
using ::pw::Status;
using ::pw::allocator::Allocator;
using ::pw::allocator::GetNullAllocator;
using ::pw::allocator::test::AllocatorForTest;
using ::pw::async2::Context;
using ::pw::async2::Coro;
using ::pw::async2::CoroContext;
using ::pw::async2::Dispatcher;
using ::pw::async2::Pending;
using ::pw::async2::Poll;
using ::pw::async2::Ready;
using ::pw::async2::Task;
using ::pw::async2::Waker;

class ExpectCoroTask final : public Task {
 public:
  ExpectCoroTask(Coro<pw::Status>&& coro) : coro_(std::move(coro)) {}

 private:
  Poll<> DoPend(Context& cx) final {
    Poll<Status> result = coro_.Pend(cx);
    if (result.IsPending()) {
      return Pending();
    }
    EXPECT_EQ(*result, OkStatus());
    return Ready();
  }
  Coro<pw::Status> coro_;
};

Coro<Result<int>> ImmediatelyReturnsFive(CoroContext&) { co_return 5; }

Coro<Status> StoresFiveThenReturns(CoroContext& coro_cx, int& out) {
  PW_CO_TRY_ASSIGN(out, co_await ImmediatelyReturnsFive(coro_cx));
  co_return OkStatus();
}

class ObjectWithCoroMethod {
 public:
  ObjectWithCoroMethod(int x) : x_(x) {}
  Coro<Status> CoroMethodStoresField(CoroContext&, int& out) {
    out = x_;
    co_return OkStatus();
  }

 private:
  int x_;
};

TEST(CoroTest, BasicFunctionsWithoutYieldingRun) {
  AllocatorForTest<256> alloc;
  CoroContext coro_cx(alloc);
  int output = 0;
  ExpectCoroTask task = StoresFiveThenReturns(coro_cx, output);
  Dispatcher dispatcher;
  dispatcher.Post(task);
  EXPECT_TRUE(dispatcher.RunUntilStalled().IsReady());
  EXPECT_EQ(output, 5);
}

TEST(CoroTest, AllocationFailureProducesInvalidCoro) {
  CoroContext coro_cx(GetNullAllocator());
  EXPECT_FALSE(ImmediatelyReturnsFive(coro_cx).IsValid());
  int x = 0;
  EXPECT_FALSE(StoresFiveThenReturns(coro_cx, x).IsValid());
}

TEST(CoroTest, ObjectWithCoroMethodIsCallable) {
  AllocatorForTest<256> alloc;
  CoroContext coro_cx(alloc);
  ObjectWithCoroMethod obj(4);
  int out = 22;
  ExpectCoroTask task = obj.CoroMethodStoresField(coro_cx, out);
  Dispatcher dispatcher;
  dispatcher.Post(task);
  EXPECT_TRUE(dispatcher.RunUntilStalled().IsReady());
  EXPECT_EQ(out, 4);
}

struct MockPendable {
  MockPendable() : poll_count(0), return_value(Pending()), last_waker() {}
  MockPendable(const MockPendable&) = delete;
  MockPendable& operator=(const MockPendable&) = delete;
  MockPendable(MockPendable&&) = delete;
  MockPendable& operator=(MockPendable&&) = delete;

  Poll<int> Pend(Context& cx) {
    ++poll_count;
    PW_ASYNC_STORE_WAKER(
        cx, last_waker, "MockPendable is waiting for last_waker");
    return return_value;
  }

  int poll_count;
  Poll<int> return_value;
  Waker last_waker;
};

Coro<Result<int>> AddTwo(CoroContext&, MockPendable& a, MockPendable& b) {
  co_return co_await a + co_await b;
}

Coro<Status> AddTwoThenStore(CoroContext& alloc,
                             MockPendable& a,
                             MockPendable& b,
                             int& out) {
  PW_CO_TRY_ASSIGN(out, co_await AddTwo(alloc, a, b));
  co_return OkStatus();
}

TEST(CoroTest, AwaitMultipleAndAwakenRuns) {
  AllocatorForTest<512> alloc;
  CoroContext coro_cx(alloc);
  MockPendable a;
  MockPendable b;
  int output = 0;
  ExpectCoroTask task = AddTwoThenStore(coro_cx, a, b, output);
  Dispatcher dispatcher;
  dispatcher.Post(task);

  EXPECT_TRUE(dispatcher.RunUntilStalled().IsPending());
  EXPECT_EQ(a.poll_count, 1);
  EXPECT_EQ(b.poll_count, 0);

  EXPECT_TRUE(dispatcher.RunUntilStalled().IsPending());
  EXPECT_EQ(a.poll_count, 1);
  EXPECT_EQ(b.poll_count, 0);

  int a_value = 4;
  a.return_value = a_value;
  std::move(a.last_waker).Wake();
  EXPECT_TRUE(dispatcher.RunUntilStalled().IsPending());
  EXPECT_EQ(a.poll_count, 2);
  EXPECT_EQ(b.poll_count, 1);

  int b_value = 5;
  b.return_value = b_value;
  std::move(b.last_waker).Wake();
  EXPECT_TRUE(dispatcher.RunUntilStalled().IsReady());
  EXPECT_EQ(a.poll_count, 2);
  EXPECT_EQ(b.poll_count, 2);
  EXPECT_EQ(output, a_value + b_value);
}

}  // namespace
