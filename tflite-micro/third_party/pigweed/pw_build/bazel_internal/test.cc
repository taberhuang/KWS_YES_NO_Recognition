// Copyright 2020 The Pigweed Authors
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

#include <array>

#include "pw_preprocessor/compiler.h"

// This symbol is defined by the linker script. If the linker script is not
// correctly used, this will be missing.
extern int _linker_defined_symbol;

// This symbol is used to create a section with a known size that we can verify
// in the linker script.
PW_PLACE_IN_SECTION(".test_section")
std::array<std::byte, 128> test_symbol = {};

// This file is intentionally very simple and is used only to test that the
// linker script generator works as expected.
int main() {
  volatile int linker_defined_pointer = _linker_defined_symbol;
  return 0;
}

// Stub to silence linker warning about this symbol missing.
extern "C" void pw_boot_Entry() { main(); }
