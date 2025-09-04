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

#include "pw_allocator/allocator.h"

#include <algorithm>
#include <cstring>

namespace pw {

using ::pw::allocator::Layout;

void* Allocator::DoReallocate(void* ptr, Layout new_layout) {
  if (Resize(ptr, new_layout.size())) {
    return ptr;
  }
  Result<Layout> old_layout = GetUsableLayout(ptr);
  if (!old_layout.ok()) {
    return nullptr;
  }
  void* new_ptr = Allocate(new_layout);
  if (new_ptr == nullptr) {
    return nullptr;
  }
  if (ptr != nullptr) {
    std::memcpy(new_ptr, ptr, std::min(new_layout.size(), old_layout->size()));
    Deallocate(ptr);
  }
  return new_ptr;
}

void* Allocator::DoReallocate(void* ptr, Layout old_layout, size_t new_size) {
  return Reallocate(ptr, Layout(new_size, old_layout.alignment()));
}

}  // namespace pw
