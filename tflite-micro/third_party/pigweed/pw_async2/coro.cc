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

#define PW_LOG_MODULE_NAME "PW_ASYNC"

#include "pw_async2/coro.h"

#include "pw_log/log.h"

namespace pw::async2::internal {

void LogCoroAllocationFailure(size_t requested_size) {
  PW_LOG_ERROR("Failed to allocate space for a coroutine of size %zu.",
               requested_size);
}

}  // namespace pw::async2::internal
