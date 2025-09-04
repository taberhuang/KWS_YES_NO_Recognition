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

#include "pw_containers/size_report/intrusive_map.h"

#include "pw_bloat/bloat_this_binary.h"

namespace pw::containers::size_report {

int Measure() {
  volatile uint32_t mask = bloat::kDefaultMask;
  auto& pairs1 = GetPairs<MapPair<K1, V1>>();
  int rc = MeasureIntrusiveMap<K1, MapPair<K1, V1>>(
      pairs1.begin(), pairs1.end(), mask);

#ifdef PW_CONTAINERS_SIZE_REPORT_ALTERNATE_VALUE
  auto& pairs2 = GetPairs<MapPair<K1, V2>>();
  rc += MeasureIntrusiveMap<K1, MapPair<K1, V2>>(
      pairs2.begin(), pairs2.end(), mask);
#endif

#ifdef PW_CONTAINERS_SIZE_REPORT_ALTERNATE_KEY_AND_VALUE
  auto& pairs3 = GetPairs<MapPair<K2, V2>>();
  rc += MeasureIntrusiveMap<K2, MapPair<K2, V2>>(
      pairs3.begin(), pairs3.end(), mask);
#endif

  return rc;
}

}  // namespace pw::containers::size_report

int main() { return ::pw::containers::size_report::Measure(); }
