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
#include <atomic>
#include <memory>
#include <thread>

#include "pw_bluetooth/controller.h"
#include "pw_bluetooth_sapphire/internal/host/common/inspect.h"
#include "pw_bluetooth_sapphire/internal/host/common/macros.h"
#include "pw_bluetooth_sapphire/internal/host/common/weak_self.h"
#include "pw_bluetooth_sapphire/internal/host/transport/acl_data_channel.h"
#include "pw_bluetooth_sapphire/internal/host/transport/command_channel.h"
#include "pw_bluetooth_sapphire/internal/host/transport/iso_data_channel.h"
#include "pw_bluetooth_sapphire/internal/host/transport/sco_data_channel.h"
#include "pw_bluetooth_sapphire/lease.h"

namespace bt::hci {

// Represents the HCI transport layer. This object owns the HCI command, ACL,
// SCO, and ISO data channels and provides the necessary control-flow mechanisms
// to send and receive HCI packets from the underlying Bluetooth controller.
class Transport final : public WeakSelf<Transport> {
 public:
  explicit Transport(
      std::unique_ptr<pw::bluetooth::Controller> controller,
      pw::async::Dispatcher& dispatcher,
      pw::bluetooth_sapphire::LeaseProvider& wake_lease_provider);

  // Initializes the command channel and features. The result will be reported
  // via |complete_callback|.
  //
  // NOTE: AclDataChannel and ScoDataChannel will be left uninitialized. They
  // must be initialized after available data buffer information has been
  // obtained from the controller (via HCI_Read_Buffer_Size and
  // HCI_LE_Read_Buffer_Size).
  void Initialize(fit::callback<void(bool /*success*/)> complete_callback);

  // TODO(armansito): hci::Transport::~Transport() should send a shutdown
  // message to the bt-hci device, which would be responsible for sending
  // HCI_Reset upon exit.
  ~Transport();

  // Initializes the ACL data channel with the given parameters. Returns false
  // if an error occurs during initialization. Initialize() must have been
  // called successfully prior to calling this method.
  bool InitializeACLDataChannel(const DataBufferInfo& bredr_buffer_info,
                                const DataBufferInfo& le_buffer_info);

  // Initializes the SCO data channel with the given parameters. Returns false
  // if an error occurs during initialization.
  bool InitializeScoDataChannel(const DataBufferInfo& buffer_info);

  // Initializes the ISO data channel with the given parameters. Returns false
  // if an error occurs during initialization.
  bool InitializeIsoDataChannel(const DataBufferInfo& buffer_info);

  pw::bluetooth::Controller::FeaturesBits GetFeatures();

  // Returns a pointer to the HCI command and event flow control handler.
  // CommandChannel is guaranteed to live as long as Transport, but may stop
  // processing packets after the Transport error callback has been called.
  CommandChannel* command_channel() const { return command_channel_.get(); }

  // Returns a pointer to the HCI ACL data flow control handler. Nullptr until
  // InitializeACLDataChannel() has succeeded.
  // AclDataChannel is guaranteed to live as long as Transport.
  AclDataChannel* acl_data_channel() const { return acl_data_channel_.get(); }

  // Returns a pointer to the HCI SCO data flow control handler. Nullptr until
  // InitializeScoDataChannel succeeds.
  // ScoDataChannel is guaranteed to live as long as Transport.
  ScoDataChannel* sco_data_channel() const { return sco_data_channel_.get(); }

  // Returns a pointer to the HCI ISO data flow control handler. Nullptr until
  // InitializeIsoDataChannel succeeds. IsoDataChannel is guaranteed to live as
  // long as Transport.
  IsoDataChannel* iso_data_channel() const { return iso_data_channel_.get(); }

  // Set a callback that should be invoked when any one of the underlying
  // channels experiences a fatal error (e.g. the HCI device has disappeared).
  //
  // When this callback is called the channels will be in an invalid state and
  // packet processing is no longer guaranteed to work. However, the channel
  // pointers are guaranteed to still be valid. It is the responsibility of the
  // callback implementation to clean up this Transport instance.
  void SetTransportErrorCallback(fit::closure callback);

  // Attach hci transport inspect node as a child node of |parent|.
  static constexpr const char* kInspectNodeName = "hci";
  void AttachInspect(inspect::Node& parent,
                     const std::string& name = kInspectNodeName);

 private:
  // Callback called by CommandChannel or ACLDataChannel on errors.
  void OnChannelError();

  pw::async::Dispatcher& dispatcher_;

  // HCI inspect node.
  inspect::Node hci_node_;

  // Callback invoked when the transport is closed (due to a channel error).
  fit::closure error_cb_;

  std::unique_ptr<pw::bluetooth::Controller> controller_;

  std::optional<pw::bluetooth::Controller::FeaturesBits> features_;

  pw::bluetooth_sapphire::LeaseProvider& wake_lease_provider_;

  // The HCI command and event flow control handler.
  // CommandChannel must be constructed first & shut down last because
  // AclDataChannel and ScoDataChannel depend on it. CommandChannel must live as
  // long as Transport to meet the expectations of upper layers, which may try
  // to send commands on destruction.
  std::unique_ptr<CommandChannel> command_channel_;

  // The ACL data flow control handler.
  std::unique_ptr<AclDataChannel> acl_data_channel_;

  // The SCO data flow control handler.
  std::unique_ptr<ScoDataChannel> sco_data_channel_;

  // The ISO data flow control handler.
  std::unique_ptr<IsoDataChannel> iso_data_channel_;

  BT_DISALLOW_COPY_AND_ASSIGN_ALLOW_MOVE(Transport);
};

}  // namespace bt::hci
