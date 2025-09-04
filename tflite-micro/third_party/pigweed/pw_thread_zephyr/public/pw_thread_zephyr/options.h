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

#include "pw_assert/assert.h"
#include "pw_thread/attrs.h"
#include "pw_thread/options.h"
#include "pw_thread/thread.h"
#include "pw_thread_zephyr/context.h"
#include "pw_thread_zephyr/priority.h"

namespace pw::thread::backend {

class NativeContext;

/// pw::thread::Options for Zephyr RTOS.
///
/// Example usage:
///
///   pw::Thread example_thread(
///     pw::thread::zephyr::Options(static_example_thread_context)
///         .set_priority(kFooPriority)
///         .set_name("example_thread"),
///     example_thread_function, example_arg);
///
/// TODO(aeremin): Add support for time slice configuration
///     (k_thread_time_slice_set when CONFIG_TIMESLICE_PER_THREAD=y).
class NativeOptions : public thread::Options {
 public:
  constexpr NativeOptions(NativeContext& context) : context_(&context) {}
  constexpr NativeOptions(const NativeOptions&) = default;
  constexpr NativeOptions(NativeOptions&&) = default;

  /// Sets the priority for the Zephyr RTOS thread.
  ///
  /// Lower priority values have a higher scheduling priority.
  ///
  /// @arg priority The desired thread priority
  /// @return Reference to this object
  constexpr NativeOptions& set_priority(int priority) {
    PW_DASSERT(priority <= kLowestPriority);
    PW_DASSERT(priority >= kHighestPriority);
    priority_ = priority;
    return *this;
  }

  /// Sets the name for the Thread.
  ///
  /// This value will be deep copied into the context and may be truncated based
  /// on PW_THREAD_ZEPHYR_CONFIG_MAX_THREAD_NAME_LEN. This may be set natively
  /// in the zephyr thread if CONFIG_THREAD_NAME is set, where it may
  /// again be truncated based on the value of CONFIG_THREAD_MAX_NAME_LEN.
  ///
  /// @arg name The name of the thread (null terminated)
  /// @return Reference to this object
  constexpr NativeOptions& set_name(const char* name) {
    name_ = name;
    return *this;
  }

  /// Sets the Zephyr RTOS native options
  ///
  /// (https://docs.zephyrproject.org/latest/kernel/services/threads/index.html#thread-options)
  ///
  /// @arg native_options Native option flags
  /// @return Reference to this object
  constexpr NativeOptions& set_native_options(uint32_t native_options) {
    native_options_ = native_options;
    return *this;
  }

  /// Sets the stack to use for the thread
  ///
  /// @arg stack The stack span to use
  /// @return Reference to this object
  constexpr NativeOptions& set_stack(span<z_thread_stack_element> stack) {
    stack_ = stack;
    return *this;
  }

  /// @return The current name of the thread
  const char* name() const { return name_; }

  /// @return The priority of the thread
  int priority() const { return priority_; }

  /// @return The native options of the thread
  uint32_t native_options() const { return native_options_; }

  /// @return The stack span to be used by the thread
  span<z_thread_stack_element> stack() const { return stack_; }

  /// Use the current configuration to create a thread.
  ///
  /// This function can only be called once
  ///
  /// @arg thread_fn The entry point to the thread
  /// @return Pointer to the internal context
  NativeContext* CreateThread(Function<void()>&& thread_fn);

 private:
  friend Thread;

  // Note that the default name may end up truncated due to
  // PW_THREAD_ZEPHYR_CONFIG_MAX_THREAD_NAME_LEN.
  static constexpr char kDefaultName[] = "pw::Thread";

  int priority_ = kDefaultPriority;
  uint32_t native_options_ = 0;
  NativeContext* context_ = nullptr;
  const char* name_ = kDefaultName;
  span<z_thread_stack_element> stack_ = span<z_thread_stack_element>();
};

/// Convert a context and attributes to NativeOptions
///
/// Note that if both the context and attributes provide a stack, the
/// attributes' stack will be used.
///
/// @arg context The context on which the thread will be created
/// @arg attributes The desired attributes of the thread
/// @return NativeOptions representing the desired thread
constexpr NativeOptions GetNativeOptions(NativeContext& context,
                                         const ThreadAttrs& attributes) {
  NativeOptions options =
      NativeOptions(context).set_priority(attributes.priority().native());

  if (attributes.has_external_stack()) {
    options.set_stack(span(attributes.native_stack_pointer(),
                           attributes.native_stack_size()));
  } else {
    options.set_stack(context.stack());
  }
  return options;
}

}  // namespace pw::thread::backend
