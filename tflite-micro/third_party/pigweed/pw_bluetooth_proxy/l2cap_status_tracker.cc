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

#include "pw_bluetooth_proxy/internal/l2cap_status_tracker.h"

#include <mutex>

#include "pw_containers/algorithm.h"
#include "pw_log/log.h"

namespace pw::bluetooth::proxy {

void L2capStatusTracker::RegisterDelegate(L2capStatusDelegate& delegate) {
  std::lock_guard lock(mutex_);
  delegates_.push_front(delegate);
}

void L2capStatusTracker::UnregisterDelegate(L2capStatusDelegate& delegate) {
  std::lock_guard lock(mutex_);
  delegates_.remove(delegate);
}

void L2capStatusTracker::HandleConnectionComplete(
    const L2capChannelConnectionInfo& info) {
  std::lock_guard lock(mutex_);
  if (pending_connection_complete_.has_value()) {
    PW_LOG_ERROR("Connection complete already pending");
    return;
  }
  pending_connection_complete_ = info;
}

void L2capStatusTracker::HandleAclDisconnectionComplete(
    uint16_t connection_handle) {
  std::lock_guard lock(mutex_);
  if (pending_acl_disconnection_complete_.has_value()) {
    PW_LOG_ERROR("ACL disconnection complete already pending");
    return;
  }
  pending_acl_disconnection_complete_ = connection_handle;
}

void L2capStatusTracker::HandleDisconnectionComplete(
    const DisconnectParams& params) {
  std::lock_guard lock(mutex_);
  if (pending_disconnection_complete_.has_value()) {
    PW_LOG_ERROR("Disconnection complete already pending");
    return;
  }
  pending_disconnection_complete_ = params;
}

void L2capStatusTracker::HandleConfigurationChanged(
    const L2capChannelConfigurationInfo& info) {
  std::lock_guard lock(mutex_);
  if (pending_configuration_complete_.has_value()) {
    PW_LOG_ERROR("Configuration already pending");
    return;
  }
  pending_configuration_complete_ = info;
}

void L2capStatusTracker::DeliverPendingConnectionComplete(
    const L2capChannelConnectionInfo& info) {
  bool track = false;
  for (L2capStatusDelegate& delegate : delegates_) {
    if (!delegate.ShouldTrackPsm(info.psm)) {
      continue;
    }

    track = true;
    delegate.HandleConnectionComplete(info);
  }

  if (track) {
    if (connected_channel_infos_.full()) {
      // TODO: https://pwbug.dev/379558046 - Let client know we won't be able to
      // notify on disconnect.
      PW_LOG_ERROR(
          "Couldn't track l2cap channel connection as requested, so will not "
          "be able to send disconnect event to client.");
      return;
    }
    connected_channel_infos_.push_back(info);
  }
}

void L2capStatusTracker::DeliverPendingAclDisconnectionComplete(
    uint16_t connection_handle) {
  for (size_t i = 0; i < connected_channel_infos_.size();) {
    L2capChannelConnectionInfo& info = connected_channel_infos_[i];

    if (info.connection_handle == connection_handle) {
      containers::ForEach(delegates_, [info](L2capStatusDelegate& delegate) {
        if (delegate.ShouldTrackPsm(info.psm)) {
          delegate.HandleDisconnectionComplete(info);
        }
      });
      // Deleting this entry in Vector, so do not increment index.
      connected_channel_infos_.erase(&info);
    } else {
      // Not deleting this entry in Vector, so increment index.
      ++i;
    }
  }
}

void L2capStatusTracker::DeliverPendingDisconnectionComplete(
    const DisconnectParams& params) {
  for (L2capStatusDelegate& delegate : delegates_) {
    auto match = [&params](const L2capChannelConnectionInfo& i) {
      return params.connection_handle == i.connection_handle &&
             params.remote_cid == i.remote_cid &&
             params.local_cid == i.local_cid;
    };
    auto connection_it = std::find_if(connected_channel_infos_.begin(),
                                      connected_channel_infos_.end(),
                                      match);
    if (connection_it == connected_channel_infos_.end()) {
      return;
    }

    delegate.HandleDisconnectionComplete(*connection_it);
    connected_channel_infos_.erase(connection_it);
  }
}

void L2capStatusTracker::DeliverPendingConfigurationComplete(
    const L2capChannelConfigurationInfo& config_info) {
  for (L2capStatusDelegate& delegate : delegates_) {
    auto match =
        [&config_info](const L2capChannelConnectionInfo& connection_info) {
          return config_info.connection_handle ==
                     connection_info.connection_handle &&
                 config_info.local_cid == connection_info.local_cid &&
                 config_info.remote_cid == connection_info.remote_cid;
        };
    auto connection_it = std::find_if(connected_channel_infos_.begin(),
                                      connected_channel_infos_.end(),
                                      match);
    if (connection_it == connected_channel_infos_.end()) {
      return;
    }
    delegate.HandleConfigurationChanged(config_info);
  }
}

void L2capStatusTracker::DeliverPendingEvents() {
  std::lock_guard lock(mutex_);
  if (pending_connection_complete_.has_value()) {
    DeliverPendingConnectionComplete(*pending_connection_complete_);
    pending_connection_complete_.reset();
  }

  if (pending_acl_disconnection_complete_.has_value()) {
    DeliverPendingAclDisconnectionComplete(
        *pending_acl_disconnection_complete_);
    pending_acl_disconnection_complete_.reset();
  }

  if (pending_disconnection_complete_.has_value()) {
    DeliverPendingDisconnectionComplete(*pending_disconnection_complete_);
    pending_disconnection_complete_.reset();
  }

  if (pending_configuration_complete_.has_value()) {
    DeliverPendingConfigurationComplete(*pending_configuration_complete_);
    pending_configuration_complete_.reset();
  }
}

}  // namespace pw::bluetooth::proxy
