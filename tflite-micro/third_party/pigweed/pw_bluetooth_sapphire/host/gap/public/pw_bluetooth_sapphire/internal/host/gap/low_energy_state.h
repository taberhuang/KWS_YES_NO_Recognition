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
#include <cstdint>

#include "pw_bluetooth_sapphire/internal/host/hci-spec/constants.h"
#include "pw_bluetooth_sapphire/internal/host/transport/data_buffer_info.h"

namespace bt::gap {

// Stores Bluetooth Low Energy settings and state information.
class LowEnergyState final {
 public:
  // Returns true if |feature_bit| is set as supported in the local LE features
  // list.
  inline bool IsFeatureSupported(
      hci_spec::LESupportedFeature feature_bit) const {
    return supported_features_ & static_cast<uint64_t>(feature_bit);
  }

  uint64_t supported_features() const { return supported_features_; }

  // Returns the LE ACL data buffer capacity.
  const hci::DataBufferInfo& acl_data_buffer_info() const {
    return acl_data_buffer_info_;
  }

  // Returns the ISO data buffer capacity.
  const hci::DataBufferInfo& iso_data_buffer_info() const {
    return iso_data_buffer_info_;
  }

  uint16_t max_advertising_data_length() const {
    return max_advertising_data_length_;
  }

  void set_supported_features(uint64_t supported_features) {
    supported_features_ = supported_features;
  }

  bool IsConnectedIsochronousStreamSupported() const {
    return IsFeatureSupported(hci_spec::LESupportedFeature::
                                  kConnectedIsochronousStreamPeripheral) ||
           IsFeatureSupported(hci_spec::LESupportedFeature::
                                  kConnectedIsochronousStreamCentral);
  }

 private:
  friend class Adapter;
  friend class AdapterImpl;

  // Storage capacity information about the controller's internal ACL data
  // buffers.
  hci::DataBufferInfo acl_data_buffer_info_;

  // Storage capacity information about the controller's internal ISO data
  // buffers.
  hci::DataBufferInfo iso_data_buffer_info_;

  // Local supported LE Features reported by the controller.
  uint64_t supported_features_ = 0;

  // Local supported LE states reported by the controller.
  uint64_t supported_states_ = 0;

  // Maximum length of data supported by the Controller for use as advertisement
  // data or scan response data in an advertising event or as periodic
  // advertisement data
  uint16_t max_advertising_data_length_ = 0;
};

}  // namespace bt::gap
