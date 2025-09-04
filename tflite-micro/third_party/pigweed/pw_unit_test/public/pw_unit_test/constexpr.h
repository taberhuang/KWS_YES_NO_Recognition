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
#pragma once

// block-submission: disable
/// @file pw_unit_test/constexpr.h
///
/// The @c_macro{PW_CONSTEXPR_TEST} macro defines a test that is executed both
/// at compile time in a `static_assert` and as a regular GoogleTest-style
/// `TEST()`. This offers the advantages of compile-time testing in a
/// structured, familiar API, without sacrificing anything from GoogleTest-style
/// tests. The framework uses the standard GoogleTest macros at run time, and is
/// compatible with GoogleTest or Pigweed's `pw_unit_test:light` framework.
///
/// To create a `constexpr` test:
/// - Include `"pw_unit_test/constexpr.h"` alongside the test framework
///   (`"pw_unit_test/framework.h"` or `"gtest/gtest.h"`).
/// - Use the macro @c_macro{PW_CONSTEXPR_TEST} instead of `TEST`. Note that the
///   function body passed as the third argument to the macro.
/// - Use the familiar GoogleTest macros, but with a `PW_TEST_` prefix. For
///   example:
///   - `EXPECT_TRUE` → @c_macro{PW_TEST_EXPECT_TRUE}
///   - `EXPECT_EQ` → @c_macro{PW_TEST_EXPECT_EQ}
///   - `ASSERT_STREQ` → @c_macro{PW_TEST_ASSERT_STREQ}
///   - etc.
///
/// The result is a familiar-looking unit test that executes both at compile
/// time and run time.
///
/// @rst
/// .. literalinclude:: constexpr_test.cc
///    :language: cpp
///    :start-after: [pw_unit_test-constexpr]
///    :end-before: [pw_unit_test-constexpr]
///
/// @endrst
///
/// **Why should I run tests at compile time?**
///
/// - Cross compile and execute tests without having to flash them to a device.
/// - Ensure `constexpr` functions can actually be evaluated at compile time.
///   For example, function templates may be marked as `constexpr`, even if they
///   do not support constant evaluation when instantiated.
/// - Catch undefined behavior, out-of-bounds access, and other issues during
///   compilation on any platform, without needing to run sanitizers.
///
/// **If compile-time testing is so great, why execute the tests are run time at
/// all?**
///
/// - Code may run differently at compile time and execution, particularly when
///   `std::is_constant_evaluated` or `if consteval` are used.
/// - Error messages are much better at runtime. @c_macro{PW_CONSTEXPR_TEST}
///   makes it simple to temporarily disable compile-time tests and see the rich
///   GoogleTest-like output (see @c_macro{SKIP_CONSTEXPR_TESTS_DONT_SUBMIT}).
/// - Tools like code coverage only work for code that is executed normally.
///
/// @c_macro{PW_CONSTEXPR_TEST} uses `stdcompat`'s
/// `cpp20::is_constant_evaluated()`. If the compiler does not support
/// `is_constant_evaluated`, only the regular GoogleTest version will run. Note
/// that compiler support is independent of the C++ standard in use.
// block-submission: enable

#include <limits>

#include "lib/stdcompat/type_traits.h"
#include "pw_preprocessor/boolean.h"
#include "pw_preprocessor/concat.h"

/// Defines a test that is executed both at compile time in a `static_assert`
/// and as a regular GoogleTest-style `TEST()`.
///
/// `PW_CONSTEXPR_TEST` works similarly to the GoogleTest `TEST()` macro, but
/// has some differences.
///
/// - All tested code must be `constexpr`.
/// - Requires the `PW_TEST_*` prefixed versions of GoogleTest's
///   `EXPECT_*` and `ASSERT_*` macros.
/// - The function body is a macro argument. This has two implications:
///   - The function body cannot contain preprocessor directives, such as
///    `#define`. If these are needed, move them to a separate function that is
///    called from the test.
///   - The `PW_CONSTEXPR_TEST` macro must be terminated with `);` after the
///     test body argument.
///
/// @param test_suite GoogleTest test suite name
/// @param test_name GoogleTest test name
/// @param ... test function body surrounded by `{` `}`.
#define PW_CONSTEXPR_TEST(test_suite, test_name, ...)                    \
  _PW_IF_CONSTEXPR_TEST(constexpr)                                       \
  void PwConstexprTest_##test_suite##_##test_name() __VA_ARGS__          \
                                                                         \
  TEST(test_suite, test_name) {                                          \
    PwConstexprTest_##test_suite##_##test_name();                        \
  }                                                                      \
                                                                         \
  static_assert([] {                                                     \
    _PW_IF_CONSTEXPR_TEST(PwConstexprTest_##test_suite##_##test_name();) \
    return true;                                                         \
  }())

// GoogleTest-style test macros for PW_CONSTEXPR_TEST.

#define PW_TEST_EXPECT_TRUE(expr) _PW_CEXPECT(TRUE, expr)
#define PW_TEST_EXPECT_FALSE(expr) _PW_CEXPECT(FALSE, expr)

#define PW_TEST_EXPECT_EQ(lhs, rhs) _PW_CEXPECT(EQ, lhs, rhs)
#define PW_TEST_EXPECT_NE(lhs, rhs) _PW_CEXPECT(NE, lhs, rhs)
#define PW_TEST_EXPECT_GT(lhs, rhs) _PW_CEXPECT(GT, lhs, rhs)
#define PW_TEST_EXPECT_GE(lhs, rhs) _PW_CEXPECT(GE, lhs, rhs)
#define PW_TEST_EXPECT_LT(lhs, rhs) _PW_CEXPECT(LT, lhs, rhs)
#define PW_TEST_EXPECT_LE(lhs, rhs) _PW_CEXPECT(LE, lhs, rhs)

#define PW_TEST_EXPECT_NEAR(lhs, rhs, error) _PW_CEXPECT(NEAR, lhs, rhs, error)
#define PW_TEST_EXPECT_FLOAT_EQ(lhs, rhs) _PW_CEXPECT(FLOAT_EQ, lhs, rhs)
#define PW_TEST_EXPECT_DOUBLE_EQ(lhs, rhs) _PW_CEXPECT(DOUBLE_EQ, lhs, rhs)

#define PW_TEST_EXPECT_STREQ(lhs, rhs) _PW_CEXPECT(STREQ, lhs, rhs)
#define PW_TEST_EXPECT_STRNE(lhs, rhs) _PW_CEXPECT(STRNE, lhs, rhs)

#define PW_TEST_ASSERT_TRUE(expr) _PW_CASSERT(TRUE, expr)
#define PW_TEST_ASSERT_FALSE(expr) _PW_CASSERT(FALSE, expr)

#define PW_TEST_ASSERT_EQ(lhs, rhs) _PW_CASSERT(EQ, lhs, rhs)
#define PW_TEST_ASSERT_NE(lhs, rhs) _PW_CASSERT(NE, lhs, rhs)
#define PW_TEST_ASSERT_GT(lhs, rhs) _PW_CASSERT(GT, lhs, rhs)
#define PW_TEST_ASSERT_GE(lhs, rhs) _PW_CASSERT(GE, lhs, rhs)
#define PW_TEST_ASSERT_LT(lhs, rhs) _PW_CASSERT(LT, lhs, rhs)
#define PW_TEST_ASSERT_LE(lhs, rhs) _PW_CASSERT(LE, lhs, rhs)

#define PW_TEST_ASSERT_NEAR(lhs, rhs, error) _PW_CASSERT(NEAR, lhs, rhs, error)
#define PW_TEST_ASSERT_FLOAT_EQ(lhs, rhs) _PW_CASSERT(FLOAT_EQ, lhs, rhs)
#define PW_TEST_ASSERT_DOUBLE_EQ(lhs, rhs) _PW_CASSERT(DOUBLE_EQ, lhs, rhs)

#define PW_TEST_ASSERT_STREQ(lhs, rhs) _PW_CASSERT(STREQ, lhs, rhs)
#define PW_TEST_ASSERT_STRNE(lhs, rhs) _PW_CASSERT(STRNE, lhs, rhs)

// Implementation details

#define _PW_CEXPECT(macro, ...)                                       \
  if (cpp20::is_constant_evaluated()) {                               \
    ::pw::unit_test::internal::Constexpr_EXPECT_##macro(__VA_ARGS__); \
  } else                                                              \
    EXPECT_##macro(__VA_ARGS__)

#define _PW_CASSERT(macro, ...)                                              \
  if (cpp20::is_constant_evaluated()) {                                      \
    if (!::pw::unit_test::internal::Constexpr_EXPECT_##macro(__VA_ARGS__)) { \
      return;                                                                \
    }                                                                        \
  } else                                                                     \
    ASSERT_##macro(__VA_ARGS__)

// Expands to the provided expression if constexpr tests are enabled.
// block-submission: disable
#define _PW_IF_CONSTEXPR_TEST(a)              \
  PW_CONCAT(                                  \
      _PW_IF_CONSTEXPR_TEST_,                 \
      PW_AND(LIB_STDCOMPAT_CONSTEVAL_SUPPORT, \
             _PW_CONSTEXPR_TEST_ENABLED(SKIP_CONSTEXPR_TESTS_DONT_SUBMIT)))(a)

#define _PW_IF_CONSTEXPR_TEST_0(a)
#define _PW_IF_CONSTEXPR_TEST_1(a) a

#define _PW_CONSTEXPR_TEST_ENABLED(a) _PW_CONSTEXPR_TEST_ENABLED2(a)
#define _PW_CONSTEXPR_TEST_ENABLED2(a) _PW_CONSTEXPR_TEST_##a

// Check if the SKIP_CONSTEXPR_TESTS_DONT_SUBMIT macro is defined, which can
// be used to temporarily disable the constexpr part of tests.
#define _PW_CONSTEXPR_TEST_SKIP_CONSTEXPR_TESTS_DONT_SUBMIT 1
#define _PW_CONSTEXPR_TEST_ 0
#define _PW_CONSTEXPR_TEST_1 0
#define _PW_CONSTEXPR_TEST_0 \
  _Do_not_define_SKIP_CONSTEXPR_TESTS_DONT_SUBMIT_to_0_remove_it_instead
// block-submission: enable

namespace pw::unit_test::internal {

// These undefined, non-constexpr functions are called when a constexpr test
// assertion fails. They appear in the error message to inform the user.
bool EXPECT_TRUE_FAILED();
bool EXPECT_FALSE_FAILED();
bool EXPECT_EQ_FAILED();
bool EXPECT_NE_FAILED();
bool EXPECT_GT_FAILED();
bool EXPECT_GE_FAILED();
bool EXPECT_LT_FAILED();
bool EXPECT_LE_FAILED();
bool EXPECT_NEAR_FAILED();
bool EXPECT_STREQ_FAILED();
bool EXPECT_STRNE_FAILED();

template <typename T>
constexpr bool Constexpr_EXPECT_TRUE(const T& expr) {
  return (expr) ? true : EXPECT_TRUE_FAILED();
}

template <typename T>
constexpr bool Constexpr_EXPECT_FALSE(const T& expr) {
  return !(expr) ? true : EXPECT_FALSE_FAILED();
}

template <typename Lhs, typename Rhs>
constexpr bool Constexpr_EXPECT_EQ(const Lhs& lhs, const Rhs& rhs) {
  return (lhs == rhs) ? true : EXPECT_EQ_FAILED();
}

template <typename Lhs, typename Rhs>
constexpr bool Constexpr_EXPECT_NE(const Lhs& lhs, const Rhs& rhs) {
  return (lhs != rhs) ? true : EXPECT_NE_FAILED();
}

template <typename Lhs, typename Rhs>
constexpr bool Constexpr_EXPECT_GT(const Lhs& lhs, const Rhs& rhs) {
  return (lhs > rhs) ? true : EXPECT_GT_FAILED();
}

template <typename Lhs, typename Rhs>
constexpr bool Constexpr_EXPECT_GE(const Lhs& lhs, const Rhs& rhs) {
  return (lhs >= rhs) ? true : EXPECT_GE_FAILED();
}

template <typename Lhs, typename Rhs>
constexpr bool Constexpr_EXPECT_LT(const Lhs& lhs, const Rhs& rhs) {
  return (lhs < rhs) ? true : EXPECT_LT_FAILED();
}

template <typename Lhs, typename Rhs>
constexpr bool Constexpr_EXPECT_LE(const Lhs& lhs, const Rhs& rhs) {
  return (lhs <= rhs) ? true : EXPECT_LE_FAILED();
}

template <typename T>
constexpr bool Constexpr_EXPECT_NEAR(T lhs, T rhs, T error) {
  T diff = lhs - rhs;
  if (diff < 0) {
    diff *= -1;
  }
  return (diff <= error) ? true : EXPECT_NEAR_FAILED();
}

constexpr bool Constexpr_EXPECT_FLOAT_EQ(float lhs, float rhs) {
  // Compare within 4 ULPs, for consistency with GoogleTest's EXPECT_FLOAT_EQ.
  return Constexpr_EXPECT_NEAR(
      lhs, rhs, 4 * std::numeric_limits<float>::epsilon());
}

constexpr bool Constexpr_EXPECT_DOUBLE_EQ(double lhs, double rhs) {
  // Compare within 4 ULPs, for consistency with GoogleTest's EXPECT_DOUBLE_EQ.
  return Constexpr_EXPECT_NEAR(
      lhs, rhs, 4 * std::numeric_limits<double>::epsilon());
}

[[nodiscard]] constexpr bool StringsAreEqual(const char* lhs, const char* rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return lhs == rhs;
  }

  while (*lhs == *rhs && *lhs != '\0') {
    lhs += 1;
    rhs += 1;
  }

  return *lhs == '\0' && *rhs == '\0';
}

constexpr bool Constexpr_EXPECT_STREQ(const char* lhs, const char* rhs) {
  return StringsAreEqual(lhs, rhs) ? true : EXPECT_STREQ_FAILED();
}

constexpr bool Constexpr_EXPECT_STRNE(const char* lhs, const char* rhs) {
  return !StringsAreEqual(lhs, rhs) ? true : EXPECT_STRNE_FAILED();
}

}  // namespace pw::unit_test::internal
