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

#include "pw_allocator/bump_allocator.h"
#include "pw_multibuf/allocator.h"
#include "pw_multibuf/multibuf_v1.h"
#include "pw_multibuf/simple_allocator.h"
#include "pw_multibuf/size_report/transfer.h"

namespace pw::multibuf::size_report {

class FrameHandlerV1 : public virtual size_report::FrameHandler<MultiBuf> {
 protected:
  FrameHandlerV1(MultiBufAllocator& mb_allocator)
      : mb_allocator_(mb_allocator) {}

 private:
  MultiBuf DoAllocateFrame() override {
    auto frame =
        mb_allocator_.AllocateContiguous(examples::kMaxDemoLinkFrameLength);
    PW_ASSERT(frame.has_value());
    return std::move(*frame);
  }

  void DoTruncate(MultiBuf& mb, size_t length) override { mb.Truncate(length); }

  void DoNarrow(MultiBuf& mb, size_t offset, size_t length) override {
    mb.DiscardPrefix(offset);
    if (length != dynamic_extent) {
      mb.Truncate(length);
    }
  }

  void DoWiden(MultiBuf& mb, size_t prefix_len, size_t suffix_len) override {
    if (prefix_len != 0) {
      PW_ASSERT(mb.ClaimPrefix(prefix_len));
    }
    if (suffix_len != 0) {
      PW_ASSERT(mb.ClaimSuffix(suffix_len));
    }
  }

  void DoPushBack(MultiBuf& mb, MultiBuf&& chunk) override {
    mb.PushSuffix(std::move(chunk));
  }

  MultiBuf::const_iterator GetBegin(const MultiBuf& mb) const override {
    return mb.begin();
  }

  MultiBuf::const_iterator GetEnd(const MultiBuf& mb) const override {
    return mb.end();
  }

  MultiBufAllocator& mb_allocator_;
};

class SenderV1 : public FrameHandlerV1, public size_report::Sender<MultiBuf> {
 public:
  SenderV1(MultiBufAllocator& mb_allocator, InlineAsyncQueue<MultiBuf>& queue)
      : FrameHandlerV1(mb_allocator), size_report::Sender<MultiBuf>(queue) {}
};

class ReceiverV1 : public FrameHandlerV1,
                   public size_report::Receiver<MultiBuf> {
 public:
  ReceiverV1(MultiBufAllocator& mb_allocator, InlineAsyncQueue<MultiBuf>& queue)
      : FrameHandlerV1(mb_allocator), size_report::Receiver<MultiBuf>(queue) {}
};

constexpr size_t kMultiBufRegionSize = 8192;
constexpr size_t kMetadataRegionSize = 1024;
std::array<std::byte, kMultiBufRegionSize> multibuf_region;
std::array<std::byte, kMetadataRegionSize> metadata_region;

void TransferMessage() {
  allocator::BumpAllocator metadata_allocator(metadata_region);
  SimpleAllocator mb_allocator(multibuf_region, metadata_allocator);
  InlineAsyncQueue<MultiBuf, 3> queue;
  SenderV1 sender(mb_allocator, queue);
  ReceiverV1 receiver(mb_allocator, queue);
  size_report::TransferMessage(sender, receiver);
}

}  // namespace pw::multibuf::size_report

int main() {
  pw::multibuf::size_report::TransferMessage();
  return 0;
}
