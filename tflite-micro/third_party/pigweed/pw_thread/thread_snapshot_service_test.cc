// Copyright 2022 The Pigweed Authors
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

#include "pw_thread/thread_snapshot_service.h"

#include "pw_protobuf/decoder.h"
#include "pw_rpc/raw/server_reader_writer.h"
#include "pw_span/span.h"
#include "pw_thread/thread_info.h"
#include "pw_thread/thread_iteration.h"
#include "pw_thread_private/thread_snapshot_service.h"
#include "pw_thread_protos/thread.pwpb.h"
#include "pw_thread_protos/thread_snapshot_service.pwpb.h"
#include "pw_unit_test/framework.h"

namespace pw::thread::proto {
namespace {

// Iterates through each proto encoded thread in the buffer.
bool EncodedThreadExists(ConstByteSpan serialized_thread_buffer,
                         ConstByteSpan thread_name) {
  protobuf::Decoder decoder(serialized_thread_buffer);
  while (decoder.Next().ok()) {
    switch (decoder.FieldNumber()) {
      case static_cast<uint32_t>(
          proto::pwpb::SnapshotThreadInfo::Fields::kThreads): {
        ConstByteSpan thread_buffer;
        EXPECT_EQ(OkStatus(), decoder.ReadBytes(&thread_buffer));
        ConstByteSpan encoded_name;
        EXPECT_EQ(OkStatus(), DecodeThreadName(thread_buffer, encoded_name));
        if (encoded_name.size() == thread_name.size()) {
          if (std::equal(thread_name.begin(),
                         thread_name.end(),
                         encoded_name.begin())) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

ThreadInfo CreateThreadInfoObject(std::optional<ConstByteSpan> name,
                                  std::optional<uintptr_t> low_addr,
                                  std::optional<uintptr_t> high_addr,
                                  std::optional<uintptr_t> peak_addr) {
  ThreadInfo thread_info;

  if (name.has_value()) {
    thread_info.set_thread_name(name.value());
  }
  if (low_addr.has_value()) {
    thread_info.set_stack_low_addr(low_addr.value());
  }
  if (high_addr.has_value()) {
    thread_info.set_stack_high_addr(high_addr.value());
  }
  if (peak_addr.has_value()) {
    thread_info.set_stack_peak_addr(peak_addr.value());
  }

  return thread_info;
}

// Test creates a custom thread info object and proto encodes. Checks that the
// custom object is encoded properly.
TEST(ThreadSnapshotService, DecodeSingleThreadInfoObject) {
  std::array<std::byte, RequiredServiceBufferSizeWithoutVariableFields(1)>
      encode_buffer;

  proto::pwpb::SnapshotThreadInfo::MemoryEncoder encoder(encode_buffer);

  ThreadInfo thread_info = CreateThreadInfoObject(
      as_bytes(span("MyThread")), /* thread name */
      static_cast<uintptr_t>(12345678u) /* stack low address */,
      static_cast<uintptr_t>(0u) /* stack high address */,
      static_cast<uintptr_t>(987654321u) /* stack peak address */);

  EXPECT_EQ(OkStatus(), ProtoEncodeThreadInfo(encoder, thread_info));

  ConstByteSpan response_span(encoder);
  EXPECT_TRUE(
      EncodedThreadExists(response_span, thread_info.thread_name().value()));
}

TEST(ThreadSnapshotService, DecodeMultipleThreadInfoObjects) {
  std::array<std::byte, RequiredServiceBufferSizeWithoutVariableFields(3)>
      encode_buffer;

  proto::pwpb::SnapshotThreadInfo::MemoryEncoder encoder(encode_buffer);

  ThreadInfo thread_info_1 =
      CreateThreadInfoObject(as_bytes(span("MyThread1")),
                             static_cast<uintptr_t>(123u),
                             static_cast<uintptr_t>(1023u),
                             static_cast<uintptr_t>(321u));

  ThreadInfo thread_info_2 =
      CreateThreadInfoObject(as_bytes(span("MyThread2")),
                             static_cast<uintptr_t>(1000u),
                             static_cast<uintptr_t>(999999u),
                             static_cast<uintptr_t>(0u));

  ThreadInfo thread_info_3 =
      CreateThreadInfoObject(as_bytes(span("MyThread3")),
                             static_cast<uintptr_t>(123u),
                             static_cast<uintptr_t>(1023u),
                             static_cast<uintptr_t>(321u));

  // Encode out of order.
  EXPECT_EQ(OkStatus(), ProtoEncodeThreadInfo(encoder, thread_info_3));
  EXPECT_EQ(OkStatus(), ProtoEncodeThreadInfo(encoder, thread_info_1));
  EXPECT_EQ(OkStatus(), ProtoEncodeThreadInfo(encoder, thread_info_2));

  ConstByteSpan response_span(encoder);
  EXPECT_TRUE(
      EncodedThreadExists(response_span, thread_info_1.thread_name().value()));
  EXPECT_TRUE(
      EncodedThreadExists(response_span, thread_info_2.thread_name().value()));
  EXPECT_TRUE(
      EncodedThreadExists(response_span, thread_info_3.thread_name().value()));
}

TEST(ThreadSnapshotService, DefaultBufferSize) {
  static std::array<std::byte, RequiredServiceBufferSizeWithoutVariableFields()>
      encode_buffer;

  proto::pwpb::SnapshotThreadInfo::MemoryEncoder encoder(encode_buffer);

  std::optional<uintptr_t> example_addr = std::numeric_limits<uintptr_t>::max();

  ThreadInfo thread_info = CreateThreadInfoObject(
      as_bytes(span("MyThread")), example_addr, example_addr, example_addr);

  for (int i = 0; i < PW_THREAD_MAXIMUM_THREADS; i++) {
    EXPECT_EQ(OkStatus(), ProtoEncodeThreadInfo(encoder, thread_info));
  }

  ConstByteSpan response_span(encoder);
  EXPECT_TRUE(
      EncodedThreadExists(response_span, thread_info.thread_name().value()));
}

TEST(ThreadSnapshotService, FailedPrecondition) {
  static std::array<std::byte,
                    RequiredServiceBufferSizeWithoutVariableFields(1)>
      encode_buffer;

  proto::pwpb::SnapshotThreadInfo::MemoryEncoder encoder(encode_buffer);

  ThreadInfo thread_info_no_name =
      CreateThreadInfoObject(std::nullopt,
                             static_cast<uintptr_t>(1111111111u),
                             static_cast<uintptr_t>(2222222222u),
                             static_cast<uintptr_t>(3333333333u));
  Status status = ProtoEncodeThreadInfo(encoder, thread_info_no_name);
  EXPECT_EQ(status, Status::FailedPrecondition());
  // Expected log: "Thread missing information needed by service."
  ErrorLog(status);

  // Same error log as above.
  ThreadInfo thread_info_no_high_addr =
      CreateThreadInfoObject(as_bytes(span("MyThread")),
                             static_cast<uintptr_t>(1111111111u),
                             std::nullopt,
                             static_cast<uintptr_t>(3333333333u));
  EXPECT_EQ(ProtoEncodeThreadInfo(encoder, thread_info_no_high_addr),
            Status::FailedPrecondition());
}

TEST(ThreadSnapshotService, Unimplemented) {
  static std::array<std::byte,
                    RequiredServiceBufferSizeWithoutVariableFields(1)>
      encode_buffer;

  proto::pwpb::SnapshotThreadInfo::MemoryEncoder encoder(encode_buffer);

  ThreadInfo thread_info_no_peak_addr =
      CreateThreadInfoObject(as_bytes(span("MyThread")),
                             static_cast<uintptr_t>(0u),
                             static_cast<uintptr_t>(0u),
                             std::nullopt);

  Status status = ProtoEncodeThreadInfo(encoder, thread_info_no_peak_addr);
  EXPECT_EQ(status, Status::Unimplemented());
  // Expected log: "Peak stack usage reporting not supported by your current OS
  // or configuration."
  ErrorLog(status);
}

}  // namespace
}  // namespace pw::thread::proto
