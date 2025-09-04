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

#include "pw_async2/select.h"

#include <variant>

#include "pw_async2/dispatcher.h"
#include "pw_async2/pendable.h"
#include "pw_async2/poll.h"
#include "pw_unit_test/framework.h"

namespace {

using ::pw::async2::AllPendablesCompleted;
using ::pw::async2::Context;
using ::pw::async2::Dispatcher;
using ::pw::async2::PendableFor;
using ::pw::async2::Pending;
using ::pw::async2::Poll;
using ::pw::async2::Selector;
using ::pw::async2::VisitSelectResult;
using ::pw::async2::Waker;

// Windows GCC emits a bogus uninitialized error for the
// move constructor below.
PW_MODIFY_DIAGNOSTICS_PUSH();
PW_MODIFY_DIAGNOSTIC_GCC(ignored, "-Wmaybe-uninitialized");
struct SomeMoveOnlyValue {
  explicit SomeMoveOnlyValue(int result) : result_(result), move_count_(0) {}
  SomeMoveOnlyValue(const SomeMoveOnlyValue&) = delete;
  SomeMoveOnlyValue& operator=(const SomeMoveOnlyValue&) = delete;
  SomeMoveOnlyValue(SomeMoveOnlyValue&& other)
      : result_(other.result_), move_count_(other.move_count_ + 1) {
    other.result_ = -1;
  }
  SomeMoveOnlyValue& operator=(SomeMoveOnlyValue&& other) {
    result_ = other.result_;
    other.result_ = -1;
    move_count_ = other.move_count_ + 1;
    return *this;
  }
  ~SomeMoveOnlyValue() = default;
  int result_;
  int move_count_;
};
PW_MODIFY_DIAGNOSTICS_POP();

struct PendableValue {
  PendableValue(int value)
      : result_(value), poll_count_(0), allow_completion_(false), waker_() {}
  PendableValue(PendableValue&) = delete;
  PendableValue(PendableValue&&) = delete;
  PendableValue& operator=(const PendableValue&) = delete;
  PendableValue& operator=(PendableValue&&) = delete;
  ~PendableValue() = default;

  Poll<SomeMoveOnlyValue> Get(Context& cx) {
    ++poll_count_;
    if (allow_completion_) {
      return Poll<SomeMoveOnlyValue>(result_);
    }
    PW_ASYNC_STORE_WAKER(cx, waker_, "PendableValue is unavailable");
    return Pending();
  }

  int result_;
  int poll_count_;
  bool allow_completion_;
  Waker waker_;
};

template <size_t kIndex, typename ResultVariant>
void ExpectVariantIs(ResultVariant& result_variant, int value) {
  std::visit(
      [value](auto& val) {
        using Type = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<Type, AllPendablesCompleted>) {
          FAIL();
        } else if constexpr (Type::kIndex == kIndex) {
          EXPECT_EQ(val.value.result_, value);
        } else {
          FAIL();
        }
      },
      result_variant);
}

template <typename ResultVariant>
void ExpectAllPendablesCompleted(ResultVariant& result_variant) {
  EXPECT_TRUE(std::holds_alternative<AllPendablesCompleted>(result_variant));
}

TEST(Selector, OnePendable) {
  Dispatcher dispatcher;

  PendableValue value(47);
  Selector selector(PendableFor<&PendableValue::Get>(value));

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());
  value.allow_completion_ = true;
  auto result = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result.IsReady());
  auto& result_variant = *result;
  ExpectVariantIs<0>(result_variant, 47);
}

TEST(Selector, DoesNotRepollAfterReady) {
  Dispatcher dispatcher;
  PendableValue value(47);
  Selector selector(PendableFor<&PendableValue::Get>(value));

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());
  EXPECT_EQ(value.poll_count_, 1);

  value.allow_completion_ = true;
  EXPECT_TRUE(dispatcher.RunPendableUntilStalled(selector).IsReady());
  EXPECT_EQ(value.poll_count_, 2);

  // After the selector returns Ready, it should not poll the pendable again.
  auto result = dispatcher.RunPendableUntilStalled(selector);
  EXPECT_EQ(value.poll_count_, 2);
  ASSERT_TRUE(result.IsReady());
  ExpectAllPendablesCompleted(*result);
}

TEST(Selector, MultiplePendables_OneReady) {
  Dispatcher dispatcher;

  PendableValue value_1(47);
  PendableValue value_2(52);
  Selector selector(PendableFor<&PendableValue::Get>(value_1),
                    PendableFor<&PendableValue::Get>(value_2));

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());
  value_2.allow_completion_ = true;
  auto result = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result.IsReady());
  auto& result_variant = *result;
  ExpectVariantIs<1>(result_variant, 52);
}

TEST(Selector, MultiplePendables_PollLoop) {
  Dispatcher dispatcher;

  PendableValue value_1(47);
  PendableValue value_2(52);
  Selector selector(PendableFor<&PendableValue::Get>(value_1),
                    PendableFor<&PendableValue::Get>(value_2));

  value_1.allow_completion_ = true;
  value_2.allow_completion_ = true;

  int iterations = 0;
  bool completed = false;

  while (!completed) {
    iterations++;

    auto result = dispatcher.RunPendableUntilStalled(selector);
    if (result.IsPending()) {
      FAIL();
      break;
    }

    VisitSelectResult(
        *result,
        [&completed](AllPendablesCompleted) { completed = true; },
        [&](SomeMoveOnlyValue value) {
          EXPECT_EQ(iterations, 1);
          EXPECT_EQ(value.result_, 47);
        },
        [&](SomeMoveOnlyValue value) {
          EXPECT_EQ(iterations, 2);
          EXPECT_EQ(value.result_, 52);
        });
  }

  EXPECT_EQ(value_1.poll_count_, 1);
  EXPECT_EQ(value_2.poll_count_, 1);
  EXPECT_EQ(iterations, 3);
}

TEST(Selector, MultiplePendables_AllReady) {
  Dispatcher dispatcher;

  PendableValue value_1(47);
  PendableValue value_2(52);
  Selector selector(PendableFor<&PendableValue::Get>(value_1),
                    PendableFor<&PendableValue::Get>(value_2));

  value_1.allow_completion_ = true;
  value_2.allow_completion_ = true;

  auto result = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result.IsReady());
  ExpectVariantIs<0>(*result, 47);

  auto result_2 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_2.IsReady());
  ExpectVariantIs<1>(*result_2, 52);
}

TEST(Selector, MultiplePendables_SequentialCompletion) {
  Dispatcher dispatcher;

  PendableValue value_1(47);
  PendableValue value_2(52);
  PendableValue value_3(57);
  Selector selector(PendableFor<&PendableValue::Get>(value_1),
                    PendableFor<&PendableValue::Get>(value_2),
                    PendableFor<&PendableValue::Get>(value_3));

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());

  value_1.allow_completion_ = true;
  auto result_1 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_1.IsReady());
  ExpectVariantIs<0>(*result_1, 47);

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());

  value_2.allow_completion_ = true;
  auto result_2 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_2.IsReady());
  ExpectVariantIs<1>(*result_2, 52);

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());

  value_3.allow_completion_ = true;
  auto result_3 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_3.IsReady());
  ExpectVariantIs<2>(*result_3, 57);
}

TEST(Selector, MultiplePendables_OutOfOrderCompletion) {
  Dispatcher dispatcher;

  PendableValue value_1(47);
  PendableValue value_2(52);
  PendableValue value_3(57);
  Selector selector(PendableFor<&PendableValue::Get>(value_1),
                    PendableFor<&PendableValue::Get>(value_2),
                    PendableFor<&PendableValue::Get>(value_3));

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());

  value_2.allow_completion_ = true;
  auto result_2 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_2.IsReady());
  ExpectVariantIs<1>(*result_2, 52);

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());

  value_1.allow_completion_ = true;
  auto result_1 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_1.IsReady());
  ExpectVariantIs<0>(*result_1, 47);

  EXPECT_EQ(dispatcher.RunPendableUntilStalled(selector), Pending());

  value_3.allow_completion_ = true;
  auto result_3 = dispatcher.RunPendableUntilStalled(selector);
  ASSERT_TRUE(result_3.IsReady());
  auto& result_3_variant = *result_3;
  ExpectVariantIs<2>(result_3_variant, 57);
}

}  // namespace
