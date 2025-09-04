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

// DOCSTAG[pw_async2-minimal-test]
#include "pw_async2/context.h"
#include "pw_async2/dispatcher.h"
#include "pw_async2/pend_func_task.h"
#include "pw_unit_test/framework.h"

using ::pw::async2::Context;
using ::pw::async2::Ready;

namespace examples {

TEST(Async2UnitTest, MinimalExample) {
  pw::async2::Dispatcher dispatcher;

  // Create a test task to run the pw_async2 code under test.
  pw::async2::PendFuncTask task([](Context&) { return Ready(); });

  // Post and run the task on the dispatcher.
  dispatcher.Post(task);
  EXPECT_EQ(dispatcher.RunUntilStalled(), Ready());
}

}  // namespace examples
// DOCSTAG[pw_async2-minimal-test]

// DOCSTAG[pw_async2-multi-step-test]
#include <utility>

#include "pw_async2/context.h"
#include "pw_async2/dispatcher.h"
#include "pw_async2/pend_func_task.h"
#include "pw_async2/try.h"
#include "pw_unit_test/framework.h"

using ::pw::async2::Context;
using ::pw::async2::Pending;
using ::pw::async2::Poll;
using ::pw::async2::Ready;

namespace examples {

// The class being tested.
class FortuneTeller {
 public:
  // Gets a fortune from the fortune teller.
  Poll<const char*> PendFortune(Context& context) {
    if (next_fortune_ == nullptr) {
      PW_ASYNC_STORE_WAKER(context, waker_, "divining the future");
      return Pending();
    }
    return std::exchange(next_fortune_, nullptr);
  }

  // Sets the next fortune to use and wakes a task waiting for one, if any.
  void SetFortune(const char* fortune) {
    next_fortune_ = fortune;

    // Wake any task waiting for a fortune. If no tasks are waiting, this is a
    // no-op.
    std::move(waker_).Wake();
  }

 private:
  pw::async2::Waker waker_;
  const char* next_fortune_ = nullptr;
};

TEST(Async2UnitTest, MultiStepExample) {
  pw::async2::Dispatcher dispatcher;

  FortuneTeller oracle;
  const char* fortune = "";

  // This task gets a fortune and checks that it matches the expected value.
  // The task may need to execute multiple times if the fortune is not ready.
  pw::async2::PendFuncTask task([&](Context& context) -> Poll<> {
    PW_TRY_READY_ASSIGN(fortune, oracle.PendFortune(context));
    return Ready();
  });

  dispatcher.Post(task);

  // The fortune hasn't been set, so the task should be pending.
  ASSERT_EQ(dispatcher.RunUntilStalled(), Pending());

  // Set the fortune, which wakes the pending task.
  oracle.SetFortune("you will bring balance to the force");

  // The task runs, gets the fortune, then returns Ready.
  ASSERT_EQ(dispatcher.RunUntilStalled(), Ready());

  // Ensure the fortune was set as expected.
  EXPECT_STREQ(fortune, "you will bring balance to the force");
}

}  // namespace examples
// DOCSTAG[pw_async2-multi-step-test]
