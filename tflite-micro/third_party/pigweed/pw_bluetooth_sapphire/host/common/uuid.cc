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

#include "pw_bluetooth_sapphire/internal/host/common/uuid.h"

#include <pw_assert/check.h>
#include <pw_bytes/endian.h>

#include "pw_bluetooth_sapphire/internal/host/common/byte_buffer.h"
#include "pw_bluetooth_sapphire/internal/host/common/random.h"
#include "pw_string/format.h"

namespace bt {
bool UUID::FromBytes(const ByteBuffer& bytes, UUID* out_uuid) {
  switch (bytes.size()) {
    case UUIDElemSize::k16Bit: {
      uint16_t dst;
      memcpy(&dst, bytes.data(), sizeof(dst));
      *out_uuid = UUID(pw::bytes::ConvertOrderFrom(cpp20::endian::little, dst));
      return true;
    }
    case UUIDElemSize::k32Bit: {
      uint32_t dst;
      memcpy(&dst, bytes.data(), sizeof(dst));
      *out_uuid = UUID(pw::bytes::ConvertOrderFrom(cpp20::endian::little, dst));
      return true;
    }
    case UUIDElemSize::k128Bit: {
      UInt128 dst;
      memcpy(dst.data(), bytes.data(), sizeof(dst));
      *out_uuid = UUID(dst);
      return true;
    }
  }

  return false;
}

UUID UUID::Generate() {
  // We generate a 128-bit random UUID in the form of version 4 as described in
  // ITU-T Rec. X.667(10/2012) Sec 15.1. This is the same as RFC 4122.
  UInt128 uuid = Random<UInt128>();
  //  Set the four most significant bits (bits 15 through 12) of the
  //  "VersionAndTimeHigh" field to 4.
  constexpr uint8_t version_number = 0b0100'0000;
  uuid[6] = (uuid[6] & 0b0000'1111) | version_number;
  // Set the two most significant bits (bits 7 and 6) of the
  // "VariantAndClockSeqHigh" field to 1 and 0, respectively.
  uuid[8] = (uuid[8] & 0b0011'1111) | 0b1000'0000;
  return UUID(uuid);
}

UUID::UUID(const ByteBuffer& bytes) {
  bool result = FromBytes(bytes, this);
  PW_CHECK(result, "|bytes| must contain a 16, 32, or 128-bit UUID");
}

bool UUID::operator==(const UUID& uuid) const { return value_ == uuid.value_; }

bool UUID::operator==(uint16_t uuid16) const {
  if (type_ == Type::k16Bit)
    return uuid16 == ValueAs16Bit();

  // Quick conversion is not possible; compare as two 128-bit UUIDs.
  return *this == UUID(uuid16);
}

bool UUID::operator==(uint32_t uuid32) const {
  if (type_ != Type::k128Bit)
    return uuid32 == ValueAs32Bit();

  // Quick conversion is not possible; compare as two 128-bit UUIDs.
  return *this == UUID(uuid32);
}

bool UUID::operator==(const UInt128& uuid128) const {
  return value_ == uuid128;
}

bool UUID::CompareBytes(const ByteBuffer& bytes) const {
  UUID other;
  if (!FromBytes(bytes, &other)) {
    return false;
  }
  return *this == other;
}

std::string UUID::ToString() const {
  char out[sizeof("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")];
  pw::StatusWithSize result = pw::string::Format(
      {out, sizeof(out)},
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      value_[15],
      value_[14],
      value_[13],
      value_[12],
      value_[11],
      value_[10],
      value_[9],
      value_[8],
      value_[7],
      value_[6],
      value_[5],
      value_[4],
      value_[3],
      value_[2],
      value_[1],
      value_[0]);
  PW_DCHECK(result.ok());
  return out;
}

UUIDElemSize UUID::CompactSize(bool allow_32bit) const {
  switch (type_) {
    case Type::k16Bit:
      return UUIDElemSize::k16Bit;
    case Type::k32Bit:
      if (allow_32bit)
        return UUIDElemSize::k32Bit;

      // Fall through if 32-bit UUIDs are not allowed.
      [[fallthrough]];
    case Type::k128Bit:
      return UUIDElemSize::k128Bit;
  };
  PW_CRASH("uuid type of %du is invalid", static_cast<uint8_t>(type_));
}

size_t UUID::ToBytes(MutableByteBuffer* bytes, bool allow_32bit) const {
  size_t size = CompactSize(allow_32bit);
  size_t offset = (size == UUIDElemSize::k128Bit) ? 0u : kBaseOffset;
  bytes->Write(value_.data() + offset, size);
  return size;
}

BufferView UUID::CompactView(bool allow_32bit) const {
  size_t size = CompactSize(allow_32bit);
  size_t offset = (size == UUIDElemSize::k128Bit) ? 0u : kBaseOffset;
  return BufferView(value_.data() + offset, size);
}

std::size_t UUID::Hash() const {
  static_assert(sizeof(value_) % sizeof(size_t) == 0);
  // Morally we'd like to assert this, but:
  //
  // 'alignof' applied to an expression is a GNU extension.
  //
  // static_assert(alignof(value_) % alignof(size_t) == 0);
  size_t hash = 0;
  for (size_t i = 0; i < (sizeof(value_) / sizeof(size_t)); i++) {
    hash ^=
        *reinterpret_cast<const size_t*>(value_.data() + (i * sizeof(size_t)));
  }
  return hash;
}

std::optional<uint16_t> UUID::As16Bit() const {
  std::optional<uint16_t> ret;
  if (type_ == Type::k16Bit) {
    ret = ValueAs16Bit();
  }
  return ret;
}

std::optional<uint32_t> UUID::As32Bit() const {
  std::optional<uint32_t> ret;
  if (type_ != Type::k128Bit) {
    ret = ValueAs32Bit();
  }
  return ret;
}

uint16_t UUID::ValueAs16Bit() const {
  PW_DCHECK(type_ == Type::k16Bit);

  return pw::bytes::ConvertOrderFrom(
      cpp20::endian::little,
      *reinterpret_cast<const uint16_t*>(value_.data() + kBaseOffset));
}

uint32_t UUID::ValueAs32Bit() const {
  PW_DCHECK(type_ != Type::k128Bit);

  return pw::bytes::ConvertOrderFrom(
      cpp20::endian::little,
      *reinterpret_cast<const uint32_t*>(value_.data() + kBaseOffset));
}

}  // namespace bt
