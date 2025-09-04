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

#include "pw_system/config.h"
#include "pw_thread/thread.h"

// For now, pw_system:async only supports FreeRTOS or standard library threads.
//
// This file will be rewritten once the SEED-0128 generic thread creation APIs
// are available. Details of the threads owned by pw_system should be an
// internal implementation detail. If configuration is necessary, it can be
// exposed through regular config options, rather than requiring users to
// implement functions.

#if __has_include("FreeRTOS.h")

#include "FreeRTOS.h"
#include "task.h"

namespace pw::system {

[[noreturn]] void StartScheduler() {
  vTaskStartScheduler();
  PW_UNREACHABLE;
}

}  // namespace pw::system

#else  // STL

#include <chrono>
#include <thread>

#include "pw_thread_stl/options.h"

namespace pw::system {

[[noreturn]] void StartScheduler() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::system_clock::duration::max());
  }
}

}  // namespace pw::system

#endif  // __has_include("FreeRTOS.h")
