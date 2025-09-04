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

#pragma once

#include "pw_bluetooth_sapphire/internal/host/common/bidirectional_multimap.h"
#include "pw_bluetooth_sapphire/internal/host/hci/discovery_filter.h"
#include "pw_bluetooth_sapphire/internal/host/hci/sequential_command_runner.h"
#include "pw_bluetooth_sapphire/internal/host/transport/transport.h"

namespace bt::hci {

class AdvertisingPacketFilter {
 public:
  // Some Controllers support offloaded packet filtering (e.g. in Android vendor
  // extensions). PacketFilterConfig allows us to switch on whether this feature
  // is enabled for the current Controller or not.
  class Config {
   public:
    Config(bool offloading_enabled, uint8_t max_filters)
        : offloading_enabled_(offloading_enabled), max_filters_(max_filters) {}

    bool offloading_enabled() const { return offloading_enabled_; }
    uint8_t max_filters() const { return max_filters_; }

   private:
    bool offloading_enabled_ = false;
    uint8_t max_filters_ = 0;
  };

  using ScanId = uint16_t;
  using FilterIndex = uint8_t;

  AdvertisingPacketFilter(const Config& config, Transport::WeakPtr hci);

  void SetPacketFilters(ScanId scan_id,
                        const std::vector<DiscoveryFilter>& filters);

  void UnsetPacketFilters(ScanId scan_id) {
    UnsetPacketFiltersInternal(scan_id, true);
  }

  std::unordered_set<ScanId> Matches(const AdvertisingData::ParseResult& ad,
                                     bool connectable,
                                     int8_t rssi) const;

  bool Matches(ScanId scan_id,
               const AdvertisingData::ParseResult& ad,
               bool connectable,
               int8_t rssi) const;

  bool IsOffloadedFilteringEnabled() const {
    return offloaded_filtering_enabled_;
  }

  const std::unordered_set<ScanId>& scan_ids() const { return scan_ids_; }

  // Returns the last filter index that was used. This method is primarily used
  // for testing.
  FilterIndex last_filter_index() const { return last_filter_index_; }

 private:
  enum class OffloadedFilterType {
    kServiceUUID,
    kServiceDataUUID,
    kSolicitationUUID,
    kLocalName,
    kManufacturerCode,
  };

  void UnsetPacketFiltersInternal(ScanId scan_id, bool run_commands);

  // Generate the next valid, available, and within range FilterIndex.
  // This function may fail if there are already the maximum offloadable filters
  // (Config::max_filters()) offloaded to the Controller.
  std::optional<FilterIndex> NextFilterIndex();

  // Determine whether the Controller has memory available to offload all of the
  // currently configured filters. If Controller offloading isn't enabled, this
  // method returns false.
  bool MemoryAvailable() const;

  // Determine whether the Controller has memory available to offload the given
  // filter. The |new_slots| map tracks the amount of individual filters that
  // will be taken up prior to this filter being offloaded. If Controller
  // offloading isn't enabled, this method returns false.
  bool MemoryAvailableForFilter(
      const DiscoveryFilter& filter,
      std::unordered_map<OffloadedFilterType, uint8_t>& new_slots) const;

  // Determine whether the Controller has memory available to offload all of the
  // given set of filters. If Controller offloading isn't enabled, this method
  // returns false.
  bool MemoryAvailableForFilters(
      const std::vector<DiscoveryFilter>& filters) const;

  // Determine whether the Controller has memory available to add |slots| more
  // offloaded filters of the type |filter_type|. If Controller offloading isn't
  // enabled, this method returns false.
  bool MemoryAvailableForSlots(OffloadedFilterType filter_type,
                               uint8_t slots) const;

  // Enable Controller based filtering. After the commands this method issues
  // are run, all advertising packet filtering will occur on the Controller
  // before we perform any secondary filtering on the Host.
  void EnableOffloadedFiltering();

  // Disable all Controller based filtering and fall back to Host level
  // filtering.
  void DisableOffloadedFiltering();

  // Determine if the this filter contains any predicates that can be offloaded
  // to the Controller.
  bool IsOffloadable(const DiscoveryFilter& filter);

  // Queue HCI commands necessary to offload the given filter to the Controller.
  bool Offload(ScanId scan_id, const DiscoveryFilter& filter);

  // Reset the number of open slots of Controller memory per filter type we
  // track to the default value.
  void ResetOpenSlots();

  CommandPacket BuildEnableCommand(bool enabled) const;

  CommandPacket BuildSetParametersCommand(FilterIndex filter_index,
                                          const DiscoveryFilter& filter);

  CommandPacket BuildClearParametersCommand() const;

  CommandPacket BuildUnsetParametersCommand(FilterIndex filter_index) const;

  std::optional<CommandPacket> BuildSetServiceUUID16Command(
      FilterIndex filter_index, const UUID& uuid) const;

  std::optional<CommandPacket> BuildSetServiceUUID32Command(
      FilterIndex filter_index, const UUID& uuid) const;

  CommandPacket BuildSetServiceUUID128Command(FilterIndex filter_index,
                                              const UUID& uuid) const;

  std::vector<CommandPacket> BuildSetServiceUUIDCommands(
      FilterIndex filter_index, const std::vector<UUID>& uuids) const;

  std::optional<CommandPacket> BuildSetSolicitationUUID16Command(
      FilterIndex filter_index, const UUID& uuid) const;

  std::optional<CommandPacket> BuildSetSolicitationUUID32Command(
      FilterIndex filter_index, const UUID& uuid) const;

  CommandPacket BuildSetSolicitationUUID128Command(FilterIndex filter_index,
                                                   const UUID& uuid) const;

  std::vector<CommandPacket> BuildSetSolicitationUUIDCommands(
      FilterIndex filter_index, const std::vector<UUID>& uuids) const;

  std::optional<CommandPacket> BuildSetServiceDataUUID16Command(
      FilterIndex filter_index, const UUID& uuid) const;

  std::optional<CommandPacket> BuildSetServiceDataUUID32Command(
      FilterIndex filter_index, const UUID& uuid) const;

  CommandPacket BuildSetServiceDataUUID128Command(FilterIndex filter_index,
                                                  const UUID& uuid) const;

  std::vector<CommandPacket> BuildSetServiceDataUUIDCommands(
      FilterIndex filter_index, const std::vector<UUID>& uuids) const;

  CommandPacket BuildSetLocalNameCommand(FilterIndex filter_index,
                                         const std::string& local_name) const;

  CommandPacket BuildSetManufacturerCodeCommand(
      FilterIndex filter_index, uint16_t manufacturer_code) const;

  // kStartFilterIndex is chosen to be 0xFF because adding one to it will
  // overflow to 0, the first valid filter index.
  constexpr static FilterIndex kStartFilterIndex = 0xFF;

  // The last generated filter index used as a hint to generate the next
  // available filter index.
  FilterIndex last_filter_index_ = kStartFilterIndex;

  // Tracks whether we are currently using Controller offloaded filtering.
  bool offloaded_filtering_enabled_ = false;

  // Packet filter configuration that controls how this class behaves
  // (e.g. Controller offloading enabled, etc).
  Config config_;

  // Controller memory supports up to a certain amount of filters to be
  // offloaded per type of filter. Track how many slots are available to use
  // within the Controller.
  std::unordered_map<OffloadedFilterType, uint8_t> open_slots_;

  // The total set of scan ids being tracked.
  std::unordered_set<ScanId> scan_ids_;

  // Filters associated with a particular upper layer scan session with a given
  // scan id. These filters may be offloaded to the Controller if Controller
  // offloading is supported. If not, or if the Controller memory is full, they
  // are used to perform Host level packet filtering.
  std::unordered_map<ScanId, std::vector<DiscoveryFilter>> scan_id_to_filters_;

  // Map between a scan id and the indexes of its filters offloaded to the
  // Controller.
  BidirectionalMultimap<ScanId, FilterIndex> scan_id_to_index_;

  // The HCI transport.
  Transport::WeakPtr hci_;

  // Command runner for all HCI commands sent out by implementations.
  std::unique_ptr<SequentialCommandRunner> hci_cmd_runner_;

  BT_DISALLOW_COPY_AND_ASSIGN_ALLOW_MOVE(AdvertisingPacketFilter);
};

}  // namespace bt::hci
