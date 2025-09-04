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

#include "pw_containers/size_report/inline_queue.h"

#include "pw_bloat/bloat_this_binary.h"

namespace pw::containers::size_report {

int Measure() {
  volatile uint32_t mask = bloat::kDefaultMask;
  auto& items1 = GetItems<V1>();
  int rc = MeasureInlineQueue<V1>(items1.begin(), items1.end(), mask);

#ifdef PW_CONTAINERS_SIZE_REPORT_ALTERNATE_VALUE
  auto& items2 = GetItems<V2>();
  rc += MeasureInlineQueue<V2>(items2.begin(), items2.end(), mask);
#endif

  return rc;
}

}  // namespace pw::containers::size_report

int main() { return ::pw::containers::size_report::Measure(); }
