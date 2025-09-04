# Copyright 2022 The Pigweed Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
include_guard(GLOBAL)

include($ENV{PW_ROOT}/pw_build/pigweed.cmake)

# Backend for the pw_thread module's pw::Thread::id.
pw_add_backend_variable(pw_thread.id_BACKEND)

# Backend for the pw_thread module's pw::thread::sleep_{for,until}.
pw_add_backend_variable(pw_thread.sleep_BACKEND)

# Backend for the pw_thread module's pw::Thread to create threads.
pw_add_backend_variable(pw_thread.thread_BACKEND)

# Backend for pw::Thread's generic thread creation functionality. Must
# set to "$dir_pw_thread:generic_thread_creation_unsupported" if
# unimplemented.
pw_add_backend_variable(pw_thread.thread_creation_BACKEND
  DEFAULT_BACKEND
    pw_thread.generic_thread_creation_unsupported
)

# Backend for the pw_thread module's pw::thread::thread_iteration.
pw_add_backend_variable(pw_thread.thread_iteration_BACKEND)

# Backend for the pw_thread module's pw::thread::yield.
pw_add_backend_variable(pw_thread.yield_BACKEND)

# Backend for the pw_thread module's pw::thread::test_thread_context.
pw_add_backend_variable(pw_thread.test_thread_context_BACKEND)
