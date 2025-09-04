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
#include "pw_bluetooth_sapphire/internal/host/hci/advertising_handle_map.h"
#include "pw_bluetooth_sapphire/internal/host/hci/low_energy_advertiser.h"

namespace bt::hci {

class Transport;
class SequentialCommandRunner;

class LegacyLowEnergyAdvertiser final : public LowEnergyAdvertiser {
 public:
  explicit LegacyLowEnergyAdvertiser(hci::Transport::WeakPtr hci)
      : LowEnergyAdvertiser(std::move(hci),
                            hci_spec::kMaxLEAdvertisingDataLength) {}
  ~LegacyLowEnergyAdvertiser() override;

  // LowEnergyAdvertiser overrides:
  size_t MaxAdvertisements() const override { return 1; }
  bool AllowsRandomAddressChange() const override {
    return !starting_ && !IsAdvertising();
  }

  // LegacyLowEnergyAdvertiser supports only a single advertising instance,
  // hence it can report additional errors in the following conditions:
  // 1. If called while a start request is pending, reports kRepeatedAttempts.
  // 2. If called while a stop request is pending, then cancels the stop request
  //    and proceeds with start.
  void StartAdvertising(
      const DeviceAddress& address,
      const AdvertisingData& data,
      const AdvertisingData& scan_rsp,
      const AdvertisingOptions& options,
      ConnectionCallback connect_callback,
      ResultFunction<AdvertisementId> result_callback) override;

  void StopAdvertising(
      fit::function<void(Result<>)> result_cb = nullptr) override;

  void StopAdvertising(
      AdvertisementId advertisement_id,
      fit::function<void(Result<>)> result_cb = nullptr) override;

  void OnIncomingConnection(
      hci_spec::ConnectionHandle handle,
      pw::bluetooth::emboss::ConnectionRole role,
      const DeviceAddress& peer_address,
      const hci_spec::LEConnectionParameters& conn_params) override;

 private:
  CommandPacket BuildEnablePacket(
      AdvertisementId advertisement_id,
      pw::bluetooth::emboss::GenericEnableParam enable) const override;

  std::optional<SetAdvertisingParams> BuildSetAdvertisingParams(
      const DeviceAddress& address,
      const AdvertisingEventProperties& properties,
      pw::bluetooth::emboss::LEOwnAddressType own_address_type,
      const AdvertisingIntervalRange& interval) override;

  std::optional<CommandPacket> BuildSetAdvertisingRandomAddr(
      AdvertisementId advertisement_id) const override;

  std::vector<CommandPacket> BuildSetAdvertisingData(
      AdvertisementId advertisement_id,
      const AdvertisingData& data,
      AdvFlags flags) const override;

  CommandPacket BuildUnsetAdvertisingData(
      AdvertisementId advertisement_id) const override;

  std::vector<CommandPacket> BuildSetScanResponse(
      AdvertisementId advertisement_id,
      const AdvertisingData& scan_rsp) const override;

  CommandPacket BuildUnsetScanResponse(
      AdvertisementId advertisement_id) const override;

  std::optional<CommandPacket> BuildRemoveAdvertisingSet(
      AdvertisementId) const override {
    return std::nullopt;
  }

  void ResetAdvertisingState();

  void OnCurrentOperationComplete() override;

  // |starting_| is set to true if a start is pending.
  // |staged_params_| are the parameters that will be advertised.
  struct StagedParams {
    DeviceAddress address;
    AdvertisingData data;
    AdvertisingData scan_rsp;
    AdvertisingOptions options;
    ConnectionCallback connect_callback;
    StartAdvertisingInternalCallback result_callback;
  };
  std::optional<StagedParams> staged_params_;
  bool starting_ = false;
  DeviceAddress local_address_;
  std::optional<AdvertisementId> active_advertisement_id_;
  AdvertisementId::value_t next_advertisement_id_ = 1;
  std::queue<fit::closure> op_queue_;

  BT_DISALLOW_COPY_AND_ASSIGN_ALLOW_MOVE(LegacyLowEnergyAdvertiser);
};

}  // namespace bt::hci
