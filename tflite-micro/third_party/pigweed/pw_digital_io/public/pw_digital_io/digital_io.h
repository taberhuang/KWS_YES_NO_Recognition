// Copyright 2021 The Pigweed Authors
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

#include "pw_assert/check.h"
#include "pw_digital_io/internal/conversions.h"
#include "pw_function/function.h"
#include "pw_result/result.h"
#include "pw_status/status.h"
#include "pw_status/try.h"

/// GPIO library
namespace pw::digital_io {

// The logical state of a digital line.
enum class State : bool {
  kActive = true,
  kInactive = false,
};

// The triggering configuration for an interrupt handler.
enum class InterruptTrigger : int {
  // Trigger on transition from kInactive to kActive.
  kActivatingEdge,
  // Trigger on transition from kActive to kInactive.
  kDeactivatingEdge,
  // Trigger on any state transition between kActive and kInactive.
  kBothEdges,
};

// Interrupt handling function. The argument contains the latest known state of
// the line. It is backend-specific if, when, and how this state is updated.
using InterruptHandler = ::pw::Function<void(State sampled_state)>;

/// @module{pw_digital_io}

/// A digital I/O line that may support input, output, and interrupts, but makes
/// no guarantees about whether any operations are supported. You must check the
/// various `provides_*` flags before calling optional methods. Unsupported
/// methods invoke `PW_CRASH`.
///
/// All methods are potentially blocking. Unless otherwise specified, access
/// from multiple threads to a single line must be externally synchronized - for
/// example using `pw::Borrowable`. Unless otherwise specified, none of the
/// methods are safe to call from an interrupt handler. Therefore, this
/// abstraction may not be suitable for bitbanging and other low-level uses of
/// GPIO.
///
/// Note that the initial state of a line is not guaranteed to be consistent
/// with either the "enabled" or "disabled" state. Users of the API who need to
/// ensure the line is disabled (ex. output not driving the line) should call
/// `Disable()`.
///
/// This class should almost never be used in APIs directly. Instead, use one of
/// the derived classes that explicitly supports the functionality that your
/// API needs.
///
/// This class cannot be extended directly. Instead, extend one of the
/// derived classes that explicitly support the functionality that you want to
/// implement.
class DigitalIoOptional {
 public:
  virtual ~DigitalIoOptional() = default;

  /// @returns `true` if input (getting state) is supported.
  constexpr bool provides_input() const { return config_.input; }
  /// @returns `true` if output (setting state) is supported.
  constexpr bool provides_output() const { return config_.output; }
  /// @returns `true` if interrupt handlers can be registered.
  constexpr bool provides_interrupt() const { return config_.interrupt; }

  /// Gets the state of the line.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: Returns an active or inactive state.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns Other status codes as defined by the backend.
  ///
  /// @endrst
  Result<State> GetState() { return DoGetState(); }

  /// Sets the state of the line.
  ///
  /// Callers are responsible to wait for the voltage level to settle after this
  /// call returns.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The state has been set.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status SetState(State state) { return DoSetState(state); }

  /// Checks if the line is in the active state.
  ///
  /// The line is in the active state when `GetState()` returns
  /// `State::kActive`.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: ``true`` if the line is in the active state, otherwise ``false``.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Result<bool> IsStateActive() {
    PW_TRY_ASSIGN(const State state, GetState());
    return state == State::kActive;
  }

  /// Sets the line to the active state. Equivalent to
  /// `SetState(State::kActive)`.
  ///
  /// Callers are responsible to wait for the voltage level to settle after this
  /// call returns.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The state has been set.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status SetStateActive() { return SetState(State::kActive); }

  /// Sets the line to the inactive state. Equivalent to
  /// `SetState(State::kInactive)`.
  ///
  /// Callers are responsible to wait for the voltage level to settle after
  /// this call returns.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The state has been set.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status SetStateInactive() { return SetState(State::kInactive); }

  /// Sets an interrupt handler to execute when an interrupt is triggered, and
  /// configures the condition for triggering the interrupt.
  ///
  /// The handler is executed in a backend-specific context—this may be a
  /// system interrupt handler or a shared notification thread. Do not do any
  /// blocking or expensive work in the handler. The only universally safe
  /// operations are the IRQ-safe functions on `pw_sync` primitives.
  ///
  /// In particular, it is NOT safe to get the state of a `DigitalIo`
  /// line—either from this line or any other `DigitalIoOptional`
  /// instance—inside the handler.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @pre No handler is currently set.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The interrupt handler was configured.
  ///
  ///    INVALID_ARGUMENT: The handler is empty.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status SetInterruptHandler(InterruptTrigger trigger,
                             InterruptHandler&& handler) {
    if (handler == nullptr) {
      return Status::InvalidArgument();
    }
    return DoSetInterruptHandler(trigger, std::move(handler));
  }

  /// Clears the interrupt handler and disables any existing interrupts that
  /// are enabled.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The interrupt handler was cleared.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status ClearInterruptHandler() {
    PW_TRY(DisableInterruptHandler());
    return DoSetInterruptHandler(InterruptTrigger::kActivatingEdge, nullptr);
  }

  /// Enables interrupts which will trigger the interrupt handler.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @pre A handler has been set using `SetInterruptHandler()`.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The interrupt handler was configured.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status EnableInterruptHandler() { return DoEnableInterruptHandler(true); }

  /// Disables the interrupt handler. This is a no-op if interrupts are
  /// disabled.
  ///
  /// This method can be called inside the interrupt handler for this line
  /// without any external synchronization. However, the exact behavior is
  /// backend-specific. There may be queued events that will trigger the handler
  /// again after this call returns.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The interrupt handler was disabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status DisableInterruptHandler() { return DoEnableInterruptHandler(false); }

  /// Enables the line to initialize it into the default state as determined by
  /// the backend.
  ///
  /// This may enable pull-up/down resistors, drive the line high/low, etc.
  /// The line must be enabled before getting/setting the state or enabling
  /// interrupts. Callers are responsible for waiting for the voltage level to
  /// settle after this call returns.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The line is enabled and ready for use.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status Enable() { return DoEnable(true); }

  /// Disables the line to power down any pull-up/down resistors and disconnect
  /// from any voltage sources.
  ///
  /// This is usually done to save power. Interrupt handlers are automatically
  /// disabled.
  ///
  /// @warning This method is not thread-safe and cannot be used in interrupt
  /// handlers.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The line is disabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  Status Disable() {
    if (provides_interrupt()) {
      PW_TRY(DisableInterruptHandler());
    }
    return DoEnable(false);
  }

 private:
  friend class DigitalInterrupt;
  friend class DigitalIn;
  friend class DigitalInInterrupt;
  friend class DigitalOut;
  friend class DigitalOutInterrupt;
  friend class DigitalInOut;
  friend class DigitalInOutInterrupt;

  // Private constructor so that only friends can extend us.
  constexpr DigitalIoOptional(internal::Provides config) : config_(config) {}

  /// Enables the line to initialize it into the default state as determined by
  /// the backend or disables the line to power down any pull-up/down resistors
  /// and disconnect from any voltage sources.
  ///
  /// This may enable pull-up/down resistors, drive the line high/low, etc.
  /// The line must be enabled before getting/setting the state or enabling
  /// interrupts. Callers are responsible for waiting for the voltage level to
  /// settle after this call returns.
  ///
  /// Calling DoEnable(true) on an already-enabled line should be a no-op, it
  /// shouldn't reset the line back to the "default state".
  ///
  /// Calling DoEnable(false) should force the line into the disabled state,
  /// If the line was not initialized at object construction time.
  ///
  /// @pre This method cannot be used in interrupt contexts.
  /// @pre When disabling, the interrupt handler must already be disabled.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The line is enabled and ready for use.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  virtual Status DoEnable(bool enable) = 0;

  /// Gets the state of the line.
  ///
  /// @pre This method cannot be used in interrupt contexts.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: An active or inactive state.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  virtual Result<State> DoGetState() = 0;

  /// Sets the state of the line.
  ///
  /// Callers are responsible to wait for the voltage level to settle after this
  /// call returns.
  ///
  /// @pre This method cannot be used in interrupt contexts.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The state has been set.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  virtual Status DoSetState(State level) = 0;

  /// Sets or clears an interrupt handler to execute when an interrupt is
  /// triggered, and configures the condition for triggering the interrupt.
  ///
  /// The handler is executed in a backend-specific context—this may be a
  /// system interrupt handler or a shared notification thread.
  ///
  /// The implementation is expected to provide the handler the last known state
  /// of the input. The intention is to either sample the current state and
  /// provide that or if not possible provide the state which triggerred the
  /// interrupt (e.g. active for activating edge, and inactive for deactivating
  /// edge).
  ///
  /// The handler is cleared by passing an empty handler, this can be checked by
  /// comparing the handler to a nullptr. The implementation must guarantee that
  /// the handler is not currently executing and (and will never execute again)
  /// after returning from DoSetInterruptHandler(_, nullptr).
  ///
  /// @pre This method cannot be used in interrupt contexts.
  /// @pre If setting a handler, no handler is permitted to be currently set.
  /// @pre When cleaing a handler, the interrupt handler must be disabled.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The interrupt handler was configured.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  virtual Status DoSetInterruptHandler(InterruptTrigger trigger,
                                       InterruptHandler&& handler) = 0;

  /// Enables or disables interrupts which will trigger the interrupt handler.
  ///
  /// @warning This interrupt handler disabling must be both thread-safe and,
  ///          interrupt-safe, however enabling is not interrupt-safe and not
  ///          thread-safe.
  ///
  /// @pre When enabling, a handler must have been set using
  ///      `DoSetInterruptHandler()`.
  /// @pre Interrupt handler enabling cannot be used in interrupt contexts.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The interrupt handler was configured.
  ///
  ///    FAILED_PRECONDITION: The line has not been enabled.
  ///
  /// Returns other status codes as defined by the backend.
  ///
  /// @endrst
  virtual Status DoEnableInterruptHandler(bool enable) = 0;

  // The configuration of this line.
  const internal::Provides config_;
};

// A digital I/O line that supports only interrupts.
//
// The input and output methods are hidden and must not be called.
//
// Use this class in APIs when only interrupt functionality is required.
// Extend this class to implement a line that only supports interrupts.
//
class DigitalInterrupt
    : public DigitalIoOptional,
      public internal::Conversions<DigitalInterrupt, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::SetInterruptHandler;

 protected:
  constexpr DigitalInterrupt()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalInterrupt>()) {}

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;

  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;
  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

  // These overrides invoke PW_CRASH.
  Status DoSetState(State) final;
  Result<State> DoGetState() final;
};

// A digital I/O line that supports only input (getting state).
//
// The output and interrupt methods are hidden and must not be called.
//
// Use this class in APIs when only input functionality is required.
// Extend this class to implement a line that only supports getting state.
//
class DigitalIn : public DigitalIoOptional,
                  public internal::Conversions<DigitalIn, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;

 protected:
  constexpr DigitalIn()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalIn>()) {}

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;

  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::SetInterruptHandler;
  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

  // These overrides invoke PW_CRASH.
  Status DoSetState(State) final;
  Status DoSetInterruptHandler(InterruptTrigger, InterruptHandler&&) final;
  Status DoEnableInterruptHandler(bool) final;
};

// An input line that supports interrupts.
//
// The output methods are hidden and must not be called.
//
// Use in APIs when input and interrupt functionality is required.
//
// Extend this class to implement a line that supports input (getting state) and
// listening for interrupts at the same time.
//
class DigitalInInterrupt
    : public DigitalIoOptional,
      public internal::Conversions<DigitalInInterrupt, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;
  using DigitalIoOptional::SetInterruptHandler;

 protected:
  constexpr DigitalInInterrupt()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalInInterrupt>()) {}

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;

  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

  // These overrides invoke PW_CRASH.
  Status DoSetState(State) final;
};

// A digital I/O line that supports only output (setting state).
//
// Input and interrupt functions are hidden and must not be called.
//
// Use in APIs when only output functionality is required.
// Extend this class to implement a line that supports output only.
//
class DigitalOut : public DigitalIoOptional,
                   public internal::Conversions<DigitalOut, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

 protected:
  constexpr DigitalOut()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalOut>()) {}

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;

  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;
  using DigitalIoOptional::SetInterruptHandler;

  // These overrides invoke PW_CRASH.
  Result<State> DoGetState() final;
  Status DoSetInterruptHandler(InterruptTrigger, InterruptHandler&&) final;
  Status DoEnableInterruptHandler(bool) final;
};

// A digital I/O line that supports output and interrupts.
//
// Input methods are hidden and must not be called.
//
// Use in APIs when output and interrupt functionality is required. For
// example, to represent a two-way signalling line.
//
// Extend this class to implement a line that supports both output and
// listening for interrupts at the same time.
//
class DigitalOutInterrupt
    : public DigitalIoOptional,
      public internal::Conversions<DigitalOutInterrupt, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::SetInterruptHandler;
  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

 protected:
  constexpr DigitalOutInterrupt()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalOutInterrupt>()) {}

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;

  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;

  // These overrides invoke PW_CRASH.
  Result<State> DoGetState() final;
};

// A digital I/O line that supports both input and output.
//
// Use in APIs when both input and output functionality is required. For
// example, to represent a line which is shared by multiple controllers.
//
// Extend this class to implement a line that supports both input and output at
// the same time.
//
class DigitalInOut
    : public DigitalIoOptional,
      public internal::Conversions<DigitalInOut, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;
  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

 protected:
  constexpr DigitalInOut()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalInOut>()) {}

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;

  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::SetInterruptHandler;

  // These overrides invoke PW_CRASH.
  Status DoSetInterruptHandler(InterruptTrigger, InterruptHandler&&) final;
  Status DoEnableInterruptHandler(bool) final;
};

// A line that supports input, output, and interrupts.
//
// Use in APIs when input, output, and interrupts are required. For example to
// represent a two-way shared line with state transition notifications.
//
// Extend this class to implement a line that supports all the functionality at
// the same time.
//
class DigitalInOutInterrupt
    : public DigitalIoOptional,
      public internal::Conversions<DigitalInOutInterrupt, DigitalIoOptional> {
 public:
  // Available functionality
  using DigitalIoOptional::ClearInterruptHandler;
  using DigitalIoOptional::DisableInterruptHandler;
  using DigitalIoOptional::EnableInterruptHandler;
  using DigitalIoOptional::GetState;
  using DigitalIoOptional::IsStateActive;
  using DigitalIoOptional::SetInterruptHandler;
  using DigitalIoOptional::SetState;
  using DigitalIoOptional::SetStateActive;
  using DigitalIoOptional::SetStateInactive;

 protected:
  constexpr DigitalInOutInterrupt()
      : DigitalIoOptional(internal::AlwaysProvidedBy<DigitalInOutInterrupt>()) {
  }

 private:
  // Unavailable functionality
  using DigitalIoOptional::provides_input;
  using DigitalIoOptional::provides_interrupt;
  using DigitalIoOptional::provides_output;
};

}  // namespace pw::digital_io
