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

#include "pw_bluetooth_sapphire/internal/host/common/random.h"

#include <pw_assert/check.h>

namespace bt {
namespace {

pw::random::RandomGenerator* g_random_generator = nullptr;

}

pw::random::RandomGenerator* random_generator() { return g_random_generator; }

void set_random_generator(pw::random::RandomGenerator* generator) {
  PW_CHECK(!generator || !g_random_generator);
  g_random_generator = generator;
}

}  // namespace bt
