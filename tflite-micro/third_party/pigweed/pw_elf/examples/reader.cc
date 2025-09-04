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

#include "pw_elf/reader.h"

#include <vector>

#include "pw_status/try.h"
#include "pw_stream/std_file_stream.h"

pw::Status ReaderExample() {
  // Open a file stream for the ELF image.
  pw::stream::StdFileReader stream("/tmp/example.elf");

  // Create an ElfReader from the file stream.
  PW_TRY_ASSIGN(auto reader, pw::elf::ElfReader::FromStream(stream));

  // Read the .example section into a vector.
  PW_TRY_ASSIGN(std::vector<std::byte> section_data,
                reader.ReadSection(".example"));

  return pw::OkStatus();
}
