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

#include "pw_bluetooth_sapphire/internal/host/testing/fake_gatt_server.h"

#include <lib/fit/function.h>
#include <pw_assert/check.h>
#include <pw_bytes/endian.h>

#include "pw_bluetooth_sapphire/internal/host/att/packet.h"
#include "pw_bluetooth_sapphire/internal/host/common/log.h"
#include "pw_bluetooth_sapphire/internal/host/gap/gap.h"
#include "pw_bluetooth_sapphire/internal/host/gatt/gatt_defs.h"
#include "pw_bluetooth_sapphire/internal/host/testing/fake_controller.h"
#include "pw_bluetooth_sapphire/internal/host/testing/fake_peer.h"

namespace bt::testing {

FakeGattServer::FakeGattServer(FakePeer* dev) : dev_(dev) {
  PW_CHECK(dev_);

  // Initialize services
  services_.insert({/*start_handle=*/1,
                    Service{.start_handle = 1,
                            .end_handle = 1,
                            .type = gap::kGenericAccessService}});
  services_.insert({/*start_handle=*/2,
                    Service{.start_handle = 2,
                            .end_handle = 2,
                            .type = gatt::types::kGenericAttributeService}});
}

void FakeGattServer::HandlePdu(hci_spec::ConnectionHandle conn,
                               const ByteBuffer& pdu) {
  if (pdu.size() < sizeof(att::OpCode)) {
    bt_log(WARN, "fake-hci", "malformed ATT packet!");
    return;
  }

  att::OpCode opcode =
      pw::bytes::ConvertOrderFrom(cpp20::endian::little, pdu.To<att::OpCode>());
  switch (opcode) {
    case att::kExchangeMTURequest:
      // Always reply back with the default ATT_MTU.
      Send(conn,
           StaticByteBuffer(att::kExchangeMTUResponse, att::kLEMinMTU, 0x00));
      break;
    case att::kReadByGroupTypeRequest:
      HandleReadByGrpType(conn, pdu.view(sizeof(att::OpCode)));
      break;
    case att::kFindByTypeValueRequest:
      HandleFindByTypeValue(conn, pdu.view(sizeof(att::OpCode)));
      break;
    default:
      SendErrorRsp(conn, opcode, 0, att::ErrorCode::kRequestNotSupported);
      break;
  }
}

void FakeGattServer::RegisterWithL2cap(FakeL2cap* l2cap_) {
  auto cb = fit::bind_member<&FakeGattServer::HandlePdu>(this);
  l2cap_->RegisterHandler(l2cap::kATTChannelId, cb);
}

void FakeGattServer::HandleReadByGrpType(hci_spec::ConnectionHandle conn,
                                         const ByteBuffer& bytes) {
  // Don't support 128-bit group types.
  if (bytes.size() != sizeof(att::ReadByGroupTypeRequestParams16)) {
    SendErrorRsp(
        conn, att::kReadByGroupTypeRequest, 0, att::ErrorCode::kInvalidPDU);
    return;
  }

  const auto& params = bytes.To<att::ReadByGroupTypeRequestParams16>();
  att::Handle start =
      pw::bytes::ConvertOrderFrom(cpp20::endian::little, params.start_handle);
  att::Handle end =
      pw::bytes::ConvertOrderFrom(cpp20::endian::little, params.end_handle);
  if (!start || end < start) {
    SendErrorRsp(conn,
                 att::kReadByGroupTypeRequest,
                 start,
                 att::ErrorCode::kInvalidHandle);
    return;
  }

  // Only support primary service discovery.
  uint16_t grp_type =
      pw::bytes::ConvertOrderFrom(cpp20::endian::little, params.type);
  if (grp_type != gatt::types::kPrimaryService16 || start > att::kHandleMin) {
    SendErrorRsp(conn,
                 att::kReadByGroupTypeRequest,
                 start,
                 att::ErrorCode::kAttributeNotFound);
    return;
  }

  // We report back the standard services.
  // TODO(armansito): Support standard characteristics and more services.
  constexpr uint8_t entry_size =
      (sizeof(att::AttributeGroupDataEntry) + sizeof(att::AttributeType16));
  DynamicByteBuffer rsp(sizeof(att::OpCode) +
                        sizeof(att::ReadByGroupTypeResponseParams) +
                        services_.size() * entry_size);
  att::PacketWriter rsp_writer(att::kReadByGroupTypeResponse, &rsp);
  auto rsp_params =
      rsp_writer.mutable_payload<att::ReadByGroupTypeResponseParams>();
  rsp_params->length = entry_size;
  MutableBufferView next_entry = rsp_writer.mutable_payload_data().mutable_view(
      sizeof(att::ReadByGroupTypeResponseParams));

  for (auto& [_, service] : services_) {
    // FakeGattServer only supports 16bit UUIDs currently.
    PW_CHECK(service.type.CompactSize(/*allow_32bit=*/false) ==
             UUIDElemSize::k16Bit);
    att::AttributeGroupDataEntry* entry =
        next_entry.AsMutable<att::AttributeGroupDataEntry>();
    entry->start_handle =
        pw::bytes::ConvertOrderTo(cpp20::endian::little, service.start_handle);
    entry->group_end_handle =
        pw::bytes::ConvertOrderTo(cpp20::endian::little, service.end_handle);
    next_entry.Write(service.type.CompactView(/*allow_32bit=*/false),
                     sizeof(att::AttributeGroupDataEntry));
    next_entry = next_entry.mutable_view(entry_size);
  }

  Send(conn, rsp);
}

void FakeGattServer::HandleFindByTypeValue(hci_spec::ConnectionHandle conn,
                                           const ByteBuffer& bytes) {
  if (bytes.size() < sizeof(att::FindByTypeValueRequestParams)) {
    bt_log(WARN, "fake-gatt", "find by type value request buffer too small");
    SendErrorRsp(
        conn, att::kFindByTypeValueRequest, 0, att::ErrorCode::kInvalidPDU);
    return;
  }

  // It is safe to read members because bytes is at least the size of the
  // params, and the params struct is packed.
  att::Handle start = pw::bytes::ConvertOrderFrom(
      cpp20::endian::little,
      bytes.ReadMember<&att::FindByTypeValueRequestParams::start_handle>());
  att::Handle end = pw::bytes::ConvertOrderFrom(
      cpp20::endian::little,
      bytes.ReadMember<&att::FindByTypeValueRequestParams::end_handle>());
  att::AttributeType16 service_kind = pw::bytes::ConvertOrderFrom(
      cpp20::endian::little,
      bytes.ReadMember<&att::FindByTypeValueRequestParams::type>());
  BufferView service_uuid_bytes =
      bytes.view(sizeof(att::FindByTypeValueRequestParams));
  UUID service_uuid;
  if (!UUID::FromBytes(service_uuid_bytes, &service_uuid)) {
    bt_log(WARN,
           "fake-gatt",
           "find by type value request has invalid service UUID");
    SendErrorRsp(
        conn, att::kFindByTypeValueRequest, 0, att::ErrorCode::kInvalidPDU);
    return;
  }

  // Support only primary service discovery by service UUID. Support only a
  // single request/response per UUID (additional requests return
  // kAttributeNotFound, ending the procedure).
  if (service_kind != gatt::types::kPrimaryService16 ||
      start > att::kHandleMin || end < att::kHandleMax) {
    SendErrorRsp(conn,
                 att::kFindByTypeValueRequest,
                 start,
                 att::ErrorCode::kAttributeNotFound);
    return;
  }

  // Send a response with the first service with a matching UUID.
  auto entry =
      std::find_if(services_.begin(), services_.end(), [&](auto entry) {
        return entry.second.type == service_uuid;
      });
  if (entry == services_.end()) {
    bt_log(WARN,
           "fake-gatt",
           "attempt to discover unsupported service UUID (uuid: %s)",
           bt_str(service_uuid));
    SendErrorRsp(conn,
                 att::kFindByTypeValueRequest,
                 start,
                 att::ErrorCode::kAttributeNotFound);
    return;
  }

  Service service = entry->second;
  StaticByteBuffer<sizeof(att::OpCode) +
                   sizeof(att::FindByTypeValueResponseParams)>
      rsp;
  att::PacketWriter writer(att::kFindByTypeValueResponse, &rsp);
  att::FindByTypeValueResponseParams* rsp_params =
      writer.mutable_payload<att::FindByTypeValueResponseParams>();
  rsp_params->handles_information_list[0].handle =
      pw::bytes::ConvertOrderTo(cpp20::endian::little, service.start_handle);
  rsp_params->handles_information_list[0].group_end_handle =
      pw::bytes::ConvertOrderTo(cpp20::endian::little, service.end_handle);
  Send(conn, rsp);
}

void FakeGattServer::Send(hci_spec::ConnectionHandle conn,
                          const ByteBuffer& pdu) {
  if (dev_->controller()) {
    dev_->controller()->SendL2CAPBFrame(conn, l2cap::kATTChannelId, pdu);
  } else {
    bt_log(WARN, "fake-hci", "no assigned FakeController!");
  }
}

void FakeGattServer::SendErrorRsp(hci_spec::ConnectionHandle conn,
                                  att::OpCode opcode,
                                  att::Handle handle,
                                  att::ErrorCode ecode) {
  StaticByteBuffer<sizeof(att::ErrorResponseParams) + sizeof(att::OpCode)>
      buffer;
  att::PacketWriter writer(att::kErrorResponse, &buffer);
  auto* params = writer.mutable_payload<att::ErrorResponseParams>();
  params->request_opcode =
      pw::bytes::ConvertOrderTo(cpp20::endian::little, opcode);
  params->attribute_handle =
      pw::bytes::ConvertOrderTo(cpp20::endian::little, handle);
  params->error_code = ecode;

  Send(conn, buffer);
}

}  // namespace bt::testing
