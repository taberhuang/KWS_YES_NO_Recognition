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
#pragma once

#include "pw_unit_test/framework_backend.h"  // IWYU pragma: export
#include "pw_unit_test/status_macros.h"      // IWYU pragma: export

// Check that the backend defined the following.

#ifndef GTEST_TEST
#error \
    "The pw_unit_test framework backend must define " \
    "GTEST_TEST(test_suite_name, test_name)"
#endif  // GTEST_TEST

#ifndef TEST
#error \
    "The pw_unit_test framework backend must define " \
    "TEST(test_suite_name, test_name)"
#endif  // TEST

#ifndef TEST_F
#error \
    "The pw_unit_test framework backend must define " \
    "TEST_F(test_fixture, test_name)"
#endif  // TEST_F

#ifndef FRIEND_TEST
#error \
    "The pw_unit_test framework backend must define " \
    "FRIEND_TEST(test_suite_name, test_name)"
#endif  // FRIEND_TEST

#ifndef EXPECT_TRUE
#error "The pw_unit_test framework backend must define EXPECT_TRUE(expr)"
#endif  // EXPECT_TRUE

#ifndef EXPECT_FALSE
#error "The pw_unit_test framework backend must define EXPECT_FALSE(expr)"
#endif  // EXPECT_FALSE

#ifndef EXPECT_EQ
#error "The pw_unit_test framework backend must define EXPECT_EQ(lhs, rhs)"
#endif  // EXPECT_EQ

#ifndef EXPECT_NE
#error "The pw_unit_test framework backend must define EXPECT_NE(lhs, rhs)"
#endif  // EXPECT_EQ

#ifndef EXPECT_GT
#error "The pw_unit_test framework backend must define EXPECT_GT(lhs, rhs)"
#endif  // EXPECT_GT

#ifndef EXPECT_GE
#error "The pw_unit_test framework backend must define EXPECT_GE(lhs, rhs)"
#endif  // EXPECT_GE

#ifndef EXPECT_LT
#error "The pw_unit_test framework backend must define EXPECT_LT(lhs, rhs)"
#endif  // EXPECT_LT

#ifndef EXPECT_LE
#error "The pw_unit_test framework backend must define EXPECT_LE(lhs, rhs)"
#endif  // EXPECT_LE

#ifndef EXPECT_NEAR
#error \
    "The pw_unit_test framework backend must define " \
    "EXPECT_NEAR(lhs, rhs, epsilon)"
#endif  // EXPECT_NEAR

#ifndef EXPECT_FLOAT_EQ
#error \
    "The pw_unit_test framework backend must define EXPECT_FLOAT_EQ(lhs, rhs)"
#endif  // EXPECT_FLOAT_EQ

#ifndef EXPECT_DOUBLE_EQ
#error \
    "The pw_unit_test framework backend must define EXPECT_DOUBLE_EQ(lhs, rhs)"
#endif  // EXPECT_DOUBLE_EQ

#ifndef EXPECT_STREQ
#error "The pw_unit_test framework backend must define EXPECT_STREQ(lhs, rhs)"
#endif  // EXPECT_STREQ

#ifndef EXPECT_STRNE
#error "The pw_unit_test framework backend must define EXPECT_STRNE(lhs, rhs)"
#endif  // EXPECT_STRNE

#ifndef ASSERT_TRUE
#error "The pw_unit_test framework backend must define ASSERT_TRUE(expr)"
#endif  // ASSERT_TRUE

#ifndef ASSERT_FALSE
#error "The pw_unit_test framework backend must define ASSERT_FALSE(expr)"
#endif  // ASSERT_FALSE

#ifndef ASSERT_EQ
#error "The pw_unit_test framework backend must define ASSERT_EQ(lhs, rhs)"
#endif  // ASSERT_EQ

#ifndef ASSERT_NE
#error "The pw_unit_test framework backend must define ASSERT_NE(lhs, rhs)"
#endif  // ASSERT_NE

#ifndef ASSERT_GT
#error "The pw_unit_test framework backend must define ASSERT_GT(lhs, rhs)"
#endif  // ASSERT_GT

#ifndef ASSERT_GE
#error "The pw_unit_test framework backend must define ASSERT_GE(lhs, rhs)"
#endif  // ASSERT_GE

#ifndef ASSERT_LT
#error "The pw_unit_test framework backend must define ASSERT_LT(lhs, rhs)"
#endif  // ASSERT_LT

#ifndef ASSERT_LE
#error "The pw_unit_test framework backend must define ASSERT_LE(lhs, rhs)"
#endif  // ASSERT_LE

#ifndef ASSERT_NEAR
#error \
    "The pw_unit_test framework backend must define "\
    "ASSERT_NEAR(lhs, rhs, epsilon)"
#endif  // ASSERT_NEAR

#ifndef ASSERT_FLOAT_EQ
#error \
    "The pw_unit_test framework backend must define ASSERT_FLOAT_EQ(lhs, rhs)"
#endif  // ASSERT_FLOAT_EQ

#ifndef ASSERT_DOUBLE_EQ
#error \
    "The pw_unit_test framework backend must define ASSERT_DOUBLE_EQ(lhs, rhs)"
#endif  // ASSERT_DOUBLE_EQ

#ifndef ASSERT_STREQ
#error "The pw_unit_test framework backend must define ASSERT_STREQ(lhs, rhs)"
#endif  // ASSERT_STREQ

#ifndef ASSERT_STRNE
#error "The pw_unit_test framework backend must define ASSERT_STRNE(lhs, rhs)"
#endif  // ASSERT_STRNE

#ifndef ADD_FAILURE
#error "The pw_unit_test framework backend must define ADD_FAILURE()"
#endif  // ADD_FAILURE

#ifndef GTEST_FAIL
#error "The pw_unit_test framework backend must define GTEST_FAIL()"
#endif  // GTEST_FAIL

#ifndef GTEST_SKIP
#error "The pw_unit_test framework backend must define GTEST_SKIP()"
#endif  // GTEST_SKIP

#ifndef FAIL
#error "The pw_unit_test framework backend must define FAIL()"
#endif  // FAIL

#ifndef GTEST_SUCCEED
#error "The pw_unit_test framework backend must define GTEST_SUCCEED()"
#endif  // GTEST_SUCCEED

#ifndef SUCCEED
#error "The pw_unit_test framework backend must define SUCCEED()"
#endif  // SUCCEED

static_assert(std::is_same_v<decltype(RUN_ALL_TESTS()), int>,
              "The pw_unit_test framework backend must define the "
              "int RUN_ALL_TESTS() function");

#ifndef GTEST_HAS_DEATH_TEST
#error "The pw_unit_test framework backend must define GTEST_HAS_DEATH_TEST"
#endif  // GTEST_HAS_DEATH_TEST

#ifndef EXPECT_DEATH_IF_SUPPORTED
#error \
    "The pw_unit_test framework backend must define "\
    "EXPECT_DEATH_IF_SUPPORTED(statement, regex)"
#endif  // EXPECT_DEATH_IF_SUPPORTED

#ifndef ASSERT_DEATH_IF_SUPPORTED
#error \
    "The pw_unit_test framework backend must define " \
    "ASSERT_DEATH_IF_SUPPORTED(statement, regex)"
#endif  // ASSERT_DEATH_IF_SUPPORTED

namespace pw::test {

/// An opaque black-box function to prevent the optimizer from removing values.
/// See: https://youtu.be/nXaxk27zwlk?t=2441
template <typename T>
inline void DoNotOptimize(const T& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

}  // namespace pw::test
