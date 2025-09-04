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

unsafe extern "Rust" {
    fn pw_kernel_target_name() -> &'static str;
    fn pw_kernel_target_console_init();
    fn pw_kernel_target_main() -> !;
}

#[inline(always)]
pub fn name() -> &'static str {
    unsafe { pw_kernel_target_name() }
}

#[inline(always)]
pub fn console_init() {
    unsafe { pw_kernel_target_console_init() }
}

#[inline(always)]
pub fn main() -> ! {
    unsafe { pw_kernel_target_main() }
}
