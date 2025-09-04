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

#include "pw_system/device_handler.h"

namespace pw::system::device_handler {

void RebootSystem() {}

void CapturePlatformMetadata(
    snapshot::pwpb::Metadata::StreamEncoder& /*metadata_encoder*/) {}

Status CaptureCpuState(
    const pw_cpu_exception_State& /*cpu_state*/,
    snapshot::pwpb::Snapshot::StreamEncoder& /*snapshot_encoder*/) {
  return OkStatus();
}

// Captures the main system thread as part of a snapshot
Status CaptureMainStackThread(
    const pw_cpu_exception_State& /*cpu_state*/,
    thread::proto::pwpb::SnapshotThreadInfo::StreamEncoder& /*encoder*/) {
  return OkStatus();
}

Status CaptureThreads(
    uint32_t /*running_thread_stack_pointer*/,
    thread::proto::pwpb::SnapshotThreadInfo::StreamEncoder& /*encoder*/) {
  return OkStatus();
}

}  // namespace pw::system::device_handler
