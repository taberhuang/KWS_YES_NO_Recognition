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

#include <optional>

#include "pw_async2/internal/poll_internal.h"
#include "pw_polyfill/language_feature_macros.h"
#include "pw_string/to_string.h"

namespace pw {

template <typename>
class Result;

namespace async2 {

/// @module{pw_async2}

/// A type whose value indicates that an operation was able to complete (or
/// was ready to produce an output).
///
/// This type is used as the contentless "value" type for ``Poll``
/// types that do not return a value.
struct ReadyType {};

/// A type whose value indicates an operation was not yet able to complete.
///
/// This is analogous to ``std::nullopt_t``, but for ``Poll``.
struct PW_NODISCARD_STR(
    "`Poll`-returning functions may or may not have completed. Their "
    "return value should be examined.") PendingType {};

/// A value that may or may not be ready yet.
///
/// ``Poll<T>`` most commonly appears as the return type of an function
/// that checks the current status of an asynchronous operation. If
/// the operation has completed, it returns with ``Ready(value)``. Otherwise,
/// it returns ``Pending`` to indicate that the operations has not yet
/// completed, and the caller should try again in the future.
///
/// ``Poll<T>`` itself is "plain old data" and does not change on its own.
/// To check the current status of an operation, the caller must invoke
/// the ``Poll<T>`` returning function again and examine the newly returned
/// ``Poll<T>``.
template <typename T = ReadyType>
class PW_NODISCARD_STR(
    "`Poll`-returning functions may or may not have completed. Their "
    "return value should be examined.") Poll {
 public:
  using value_type = T;

  /// Basic constructors.
  Poll() = delete;
  constexpr Poll(const Poll&) = default;
  constexpr Poll& operator=(const Poll&) = default;
  constexpr Poll(Poll&&) = default;
  constexpr Poll& operator=(Poll&&) = default;

  /// Constructs a new ``Poll<T>`` from a ``Poll<U>`` where ``T`` is
  /// constructible from ``U``.
  ///
  /// To avoid ambiguity, this constructor is disabled if ``T`` is also
  /// constructible from ``Poll<U>``.
  ///
  /// This constructor is explicit if and only if the corresponding construction
  /// of ``T`` from ``U`` is explicit.
  template <
      typename U,
      internal_poll::EnableIfImplicitlyConvertible<value_type, const U&> = 0>
  constexpr Poll(const Poll<U>& other) : value_(other.value_) {}
  template <
      typename U,
      internal_poll::EnableIfExplicitlyConvertible<value_type, const U&> = 0>
  explicit constexpr Poll(const Poll<U>& other) : value_(other.value_) {}

  template <typename U,
            internal_poll::EnableIfImplicitlyConvertible<value_type, U&&> = 0>
  constexpr Poll(Poll<U>&& other)  // NOLINT
      : value_(std::move(other.value_)) {}
  template <typename U,
            internal_poll::EnableIfExplicitlyConvertible<value_type, U&&> = 0>
  explicit constexpr Poll(Poll<U>&& other) : value_(std::move(other.value_)) {}

  // Constructs the inner value `T` in-place using the provided args, using the
  // `T(U)` (direct-initialization) constructor. This constructor is only valid
  // if `T` can be constructed from a `U`. Can accept move or copy constructors.
  //
  // This constructor is explicit if `U` is not convertible to `T`. To avoid
  // ambiguity, this constructor is disabled if `U` is a `Poll<J>`, where
  // `J` is convertible to `T`.
  template <typename U = value_type,
            internal_poll::EnableIfImplicitlyInitializable<value_type, U> = 0>
  constexpr Poll(U&& u)  // NOLINT
      : Poll(std::in_place, std::forward<U>(u)) {}

  template <typename U = value_type,
            internal_poll::EnableIfExplicitlyInitializable<value_type, U> = 0>
  explicit constexpr Poll(U&& u)  // NOLINT
      : Poll(std::in_place, std::forward<U>(u)) {}

  // In-place construction of ``Ready`` variant.
  template <typename... Args>
  constexpr Poll(std::in_place_t, Args&&... args)
      : value_(std::in_place, std::move(args)...) {}

  // Convert from `T`
  constexpr Poll(value_type&& value) : value_(std::move(value)) {}
  constexpr Poll& operator=(value_type&& value) {
    value_ = std::optional<value_type>(std::move(value));
    return *this;
  }

  // Convert from `Pending`
  constexpr Poll(PendingType) noexcept : value_() {}
  constexpr Poll& operator=(PendingType) noexcept {
    value_ = std::nullopt;
    return *this;
  }

  /// Returns whether or not this value is ``Ready``.
  constexpr bool IsReady() const noexcept { return value_.has_value(); }

  /// Returns whether or not this value is ``Pending``.
  constexpr bool IsPending() const noexcept { return !value_.has_value(); }

  /// Returns a ``Poll<>`` without the inner value whose readiness matches that
  /// of ``this``.
  constexpr Poll<> Readiness() const noexcept {
    if (IsReady()) {
      return ReadyType();
    } else {
      return PendingType();
    }
  }

  /// Returns the inner value.
  ///
  /// This must only be called if ``IsReady()`` returned ``true``.
  constexpr value_type& value() & noexcept { return *value_; }
  constexpr const value_type& value() const& noexcept { return *value_; }
  constexpr value_type&& value() && noexcept { return std::move(*value_); }
  constexpr const value_type&& value() const&& noexcept {
    return std::move(*value_);
  }

  /// Accesses the inner value.
  ///
  /// This must only be called if ``IsReady()`` returned ``true``.
  constexpr const value_type* operator->() const noexcept { return &*value_; }
  constexpr value_type* operator->() noexcept { return &*value_; }

  /// Returns the inner value.
  ///
  /// This must only be called if ``IsReady()`` returned ``true``.
  constexpr const value_type& operator*() const& noexcept { return *value_; }
  constexpr value_type& operator*() & noexcept { return *value_; }
  constexpr const value_type&& operator*() const&& noexcept {
    return std::move(*value_);
  }
  constexpr value_type&& operator*() && noexcept { return std::move(*value_); }

  /// Ignores the ``Poll`` value.
  ///
  /// This method does nothing except prevent ``no_discard`` or
  /// unused variable warnings from firing.
  constexpr void IgnorePoll() const {}

 private:
  template <typename U>
  friend class Poll;
  std::optional<value_type> value_;
};

// Deduction guide to allow ``Poll(v)`` rather than ``Poll<T>(v)``.
template <typename T>
Poll(T value) -> Poll<T>;

/// Convenience alias for `pw::async2::Poll<pw::Result<T>>`.
template <typename T>
using PollResult = Poll<Result<T>>;

/// Convenience alias for Poll<std::optional>.
template <typename T>
using PollOptional = Poll<std::optional<T>>;

/// Returns whether two instances of ``Poll<T>`` are equal.
///
/// Note that this comparison operator will return ``true`` if both
/// values are currently ``Pending``, even if the eventual results
/// of each operation might differ.
template <typename T>
constexpr bool operator==(const Poll<T>& lhs, const Poll<T>& rhs) {
  if (lhs.IsReady() && rhs.IsReady()) {
    return *lhs == *rhs;
  }
  return lhs.IsReady() == rhs.IsReady();
}

/// Returns whether two instances of ``Poll<T>`` are unequal.
///
/// Note that this comparison operator will return ``false`` if both
/// values are currently ``Pending``, even if the eventual results
/// of each operation might differ.
template <typename T>
constexpr bool operator!=(const Poll<T>& lhs, const Poll<T>& rhs) {
  return !(lhs == rhs);
}

/// Returns whether ``lhs`` is pending.
template <typename T>
constexpr bool operator==(const Poll<T>& lhs, PendingType) {
  return lhs.IsPending();
}

/// Returns whether ``lhs`` is not pending.
template <typename T>
constexpr bool operator!=(const Poll<T>& lhs, PendingType) {
  return !lhs.IsPending();
}

/// Returns whether ``rhs`` is pending.
template <typename T>
constexpr bool operator==(PendingType, const Poll<T>& rhs) {
  return rhs.IsPending();
}

/// Returns whether ``rhs`` is not pending.
template <typename T>
constexpr bool operator!=(PendingType, const Poll<T>& rhs) {
  return !rhs.IsPending();
}

// ``ReadyType`` is the value type for `Poll<T>` and has no value,
// so it should always compare equal.
constexpr bool operator==(ReadyType, ReadyType) { return true; }
constexpr bool operator!=(ReadyType, ReadyType) { return false; }

// The ``Pending`` case holds no value, so is always equal.
constexpr bool operator==(PendingType, PendingType) { return true; }
constexpr bool operator!=(PendingType, PendingType) { return false; }

/// Returns a value indicating completion.
inline constexpr Poll<> Ready() { return Poll(ReadyType{}); }

/// Returns a value indicating completion with some result
/// (constructed in-place).
template <typename T, typename... Args>
constexpr Poll<T> Ready(std::in_place_t, Args&&... args) {
  return Poll<T>(std::in_place, std::forward<Args>(args)...);
}

/// Returns a value indicating completion with some result.
template <typename T>
constexpr Poll<std::remove_reference_t<T>> Ready(T&& value) {
  return Poll<std::remove_reference_t<T>>(std::forward<T>(value));
}

/// Returns a value indicating that an operation was not yet able to complete.
inline constexpr PendingType Pending() { return PendingType(); }

template <typename T>
struct UnwrapPoll {
  using Type = T;
};

template <typename T>
struct UnwrapPoll<Poll<T>> {
  using Type = T;
};

}  // namespace async2

// --- ToString implementations for ``Poll`` types ---

template <>
inline StatusWithSize ToString(const async2::ReadyType&, span<char> buffer) {
  return ToString("Ready", buffer);
}

template <>
inline StatusWithSize ToString(const async2::PendingType&, span<char> buffer) {
  return ToString("Pending", buffer);
}

// Implement ``ToString`` for ``Poll<T>``.
template <typename T>
inline StatusWithSize ToString(const async2::Poll<T>& poll, span<char> buffer) {
  if (poll.IsReady()) {
    StatusWithSize s;
    s.UpdateAndAdd(ToString("Ready(", buffer));
    s.UpdateAndAdd(ToString(*poll, buffer.subspan(s.size())));
    s.UpdateAndAdd(ToString(")", buffer.subspan(s.size())));
    s.ZeroIfNotOk();
    return s;
  }
  return ToString(async2::PendingType{}, buffer);
}

template <>
inline StatusWithSize ToString(const async2::Poll<>& poll, span<char> buffer) {
  if (poll.IsReady()) {
    return ToString(async2::ReadyType{}, buffer);
  }
  return ToString(async2::PendingType{}, buffer);
}

}  // namespace pw
