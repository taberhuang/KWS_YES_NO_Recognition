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

#include "pw_bytes/suffix.h"

#include "pw_compilation_testing/negative_compilation.h"
#include "pw_polyfill/standard.h"
#include "pw_unit_test/framework.h"

namespace {

using ::pw::operator""_b;

TEST(Suffix, ReturnsByte) {
  std::byte x = 5_b;
  EXPECT_EQ(x, std::byte(5));
}

#if PW_CXX_STANDARD_IS_SUPPORTED(20)
#if PW_NC_TEST(Suffix_ErrorsAtCompileTimeOnTooLargeOfValueInCpp20AndAbove)
PW_NC_EXPECT("ByteLiteralIsTooLarge");
TEST(Suffix, ErrorsAtCompileTimeOnTooLargeOfValueInCpp20AndAbove) {
  [[maybe_unused]] std::byte x = 256_b;
}
#endif  // PW_NC_TEST
#endif  // PW_CXX_STANDARD_IS_SUPPORTED(20)

}  // namespace
