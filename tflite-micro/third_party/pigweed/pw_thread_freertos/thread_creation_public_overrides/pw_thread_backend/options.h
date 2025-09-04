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

#include "pw_thread_freertos/context.h"
#include "pw_thread_freertos/options.h"

namespace pw::thread::backend {

using NativeOptions = ::pw::thread::freertos::Options;

constexpr NativeOptions GetNativeOptions(NativeContext& context,
                                         const ThreadAttrs& attrs) {
  NativeOptions options = NativeOptions()
                              .set_name(attrs.name())
                              .set_priority(attrs.priority().native());

#if PW_THREAD_FREERTOS_CONFIG_DYNAMIC_ALLOCATION_ENABLED
  static_cast<void>(context);  // The static context is not used.

  if (attrs.has_external_stack()) {
    // Use the stack size, but not the stack if dynamic allocation is enabled.
    options.set_stack_size(attrs.native_stack_size());
  } else {
    options.set_stack_size(freertos::BytesToWords(attrs.stack_size_bytes()));
  }
  return options;
}
#else   // Statically allocated stack
  options.set_static_context(context);

  if (attrs.has_external_stack()) {
    freertos::internal::SetStackForContext(context, attrs.native_stack());
  }
  return options;
}

template <size_t kStackSizeBytes>
constexpr NativeOptions GetNativeOptions(
    NativeContextWithStack<kStackSizeBytes>& context,
    const ThreadAttrs& attrs) {
  return GetNativeOptions(context.context(), attrs);
}
#endif  // PW_THREAD_FREERTOS_CONFIG_DYNAMIC_ALLOCATION_ENABLED

}  // namespace pw::thread::backend
