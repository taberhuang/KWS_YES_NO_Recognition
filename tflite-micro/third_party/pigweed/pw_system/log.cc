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

#include "pw_system/log.h"

#include <array>
#include <cstddef>

#include "pw_log_rpc/rpc_log_drain.h"
#include "pw_log_rpc/rpc_log_drain_map.h"
#include "pw_multisink/multisink.h"
#include "pw_sync/lock_annotations.h"
#include "pw_sync/mutex.h"
#include "pw_system/config.h"

namespace pw::system {
namespace {

using log_rpc::RpcLogDrain;

// Storage container for MultiSink used for deferred logging.
std::array<std::byte, PW_SYSTEM_LOG_BUFFER_SIZE> log_buffer;

// To save RAM, share the mutex and buffer between drains, since drains are
// flushed sequentially.
sync::Mutex drains_mutex;

// Buffer to decode and remove entries from log buffer, to send to a drain.
//
// TODO(amontanez): pw_log_rpc should provide a helper for this since there's
// proto encoding overhead unaccounted for here.
static_assert(rpc::MaxSafePayloadSize() >= PW_SYSTEM_MAX_LOG_ENTRY_SIZE);
std::array<std::byte, PW_SYSTEM_MAX_LOG_ENTRY_SIZE> log_decode_buffer
    PW_GUARDED_BY(drains_mutex);
#if PW_SYSTEM_EXTRA_LOGGING_CHANNEL_ID != PW_SYSTEM_LOGGING_CHANNEL_ID
std::array<std::byte, PW_SYSTEM_MAX_LOG_ENTRY_SIZE> log_decode_buffer_extra
    PW_GUARDED_BY(drains_mutex);
#endif

#if PW_SYSTEM_EXTRA_LOGGING_CHANNEL_ID != PW_SYSTEM_LOGGING_CHANNEL_ID
constexpr size_t drain_count = 2;
#else
constexpr size_t drain_count = 1;
#endif
std::array<RpcLogDrain, drain_count> drains{{
    RpcLogDrain(kLoggingRpcChannelId,
                log_decode_buffer,
                drains_mutex,
                RpcLogDrain::LogDrainErrorHandling::kIgnoreWriterErrors),
#if PW_SYSTEM_EXTRA_LOGGING_CHANNEL_ID != PW_SYSTEM_LOGGING_CHANNEL_ID
    RpcLogDrain(kExtraLoggingRpcChannelId,
                log_decode_buffer_extra,
                drains_mutex,
                RpcLogDrain::LogDrainErrorHandling::kIgnoreWriterErrors,
                nullptr,
                PW_SYSTEM_EXTRA_LOGGING_MAX_BUNDLE_PER_TRICKLE),
#endif
}};

log_rpc::RpcLogDrainMap drain_map(drains);

constexpr size_t kMaxPackedLogMessagesSize = rpc::MaxSafePayloadSize();
std::array<std::byte, kMaxPackedLogMessagesSize> log_packing_buffer;

}  // namespace

// Deferred log buffer, for storing log entries while logging_thread_ streams
// them independently.
multisink::MultiSink& GetMultiSink() {
  static multisink::MultiSink multisink(log_buffer);
  return multisink;
}

log_rpc::RpcLogDrainThread& GetLogThread() {
  static log_rpc::RpcLogDrainThread logging_thread(
      GetMultiSink(), drain_map, log_packing_buffer);
  return logging_thread;
}

log_rpc::LogService& GetLogService() {
  static log_rpc::LogService log_service(drain_map);
  return log_service;
}

}  // namespace pw::system
