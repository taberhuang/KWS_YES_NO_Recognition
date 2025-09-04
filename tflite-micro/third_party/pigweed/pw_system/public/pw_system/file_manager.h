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

#include <string_view>

#include "pw_persistent_ram/flat_file_system_entry.h"
#include "pw_system/config.h"
#include "pw_system/transfer_handlers.h"

namespace pw::system {

class FileManager {
 public:
  // Each transfer handler ID corresponds 1:1 with a transfer handler and
  // filesystem element pair.  The ID must be unique and increment from 0 to
  // ensure no gaps in the FileManager handler & filesystem arrays.
  // NOTE: the enumerators should never have values defined, to ensure they
  // increment from zero and kNumFileSystemEntries is correct
  enum TransferHandlerId : uint32_t {
#if PW_SYSTEM_ENABLE_CRASH_HANDLER
    kCrashSnapshotTransferHandlerId,
#endif  // PW_SYSTEM_ENABLE_CRASH_HANDLER
#if PW_SYSTEM_ENABLE_TRACE_SERVICE
    kTraceTransferHandlerId,
#endif  // PW_SYSTEM_ENABLE_TRACE_SERVICE
    kNumFileSystemEntries
  };

#if PW_SYSTEM_ENABLE_CRASH_HANDLER
  static constexpr std::string_view kCrashSnapshotFilename{
      "/snapshots/crash_0.snapshot"};
#endif  // PW_SYSTEM_ENABLE_CRASH_HANDLER
#if PW_SYSTEM_ENABLE_TRACE_SERVICE
  static constexpr std::string_view kTraceFilename{"/trace/0.bin"};
#endif  // PW_SYSTEM_ENABLE_TRACE_SERVICE

  FileManager();

  std::array<transfer::Handler*, kNumFileSystemEntries>& GetTransferHandlers() {
    return transfer_handlers_;
  }
  std::array<file::FlatFileSystemService::Entry*, kNumFileSystemEntries>&
  GetFileSystemEntries() {
    return file_system_entries_;
  }

 private:
#if PW_SYSTEM_ENABLE_CRASH_HANDLER
  CrashSnapshotBufferTransfer crash_snapshot_handler_;
  persistent_ram::FlatFileSystemPersistentBufferEntry<
      PW_SYSTEM_CRASH_SNAPSHOT_MEMORY_SIZE_BYTES>
      crash_snapshot_filesystem_entry_;
#endif  // PW_SYSTEM_ENABLE_CRASH_HANDLER

#if PW_SYSTEM_ENABLE_TRACE_SERVICE
  TraceBufferTransfer trace_data_handler_;
  persistent_ram::FlatFileSystemPersistentBufferEntry<
      PW_TRACE_BUFFER_SIZE_BYTES>
      trace_data_filesystem_entry_;
#endif  // PW_SYSTEM_ENABLE_TRACE_SERVICE

  std::array<transfer::Handler*, kNumFileSystemEntries> transfer_handlers_;
  std::array<file::FlatFileSystemService::Entry*, kNumFileSystemEntries>
      file_system_entries_;
};

FileManager& GetFileManager();

}  // namespace pw::system
