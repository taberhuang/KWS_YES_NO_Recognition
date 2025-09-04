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

use kernel::memory::{MemoryRegion, MemoryRegionType};
use kernel_config::{KernelConfig, RiscVKernelConfigInterface as _};

use crate::regs::pmp::*;

/// RISC-V memory configuration
///
/// Represents the full configuration of RISC-V memory configuration through
/// the PMP block.
#[derive(Clone)]
pub struct MemoryConfig {
    pmp_config: PmpConfig<{ KernelConfig::PMP_ENTRIES }>,
    regions: &'static [MemoryRegion],
}

impl MemoryConfig {
    /// Create a new `MemoryConfig` in a `const` context
    ///
    /// # Panics
    /// Will panic if the current target's PMP will does not have enough entries
    /// to represent the provided `regions`.
    pub const fn const_new(regions: &'static [MemoryRegion]) -> Self {
        match PmpConfig::new(regions) {
            Ok(cfg) => Self {
                pmp_config: cfg,
                regions,
            },
            Err(_) => panic!("Can't create Memory config"),
        }
    }

    /// Write this memory configuration to the PMP registers.
    pub unsafe fn write(&self) {
        unsafe { self.pmp_config.write() }
    }
}

impl kernel::memory::MemoryConfig for MemoryConfig {
    const KERNEL_THREAD_MEMORY_CONFIG: Self = Self::const_new(&[MemoryRegion::new(
        MemoryRegionType::ReadWriteExecutable,
        0x0000_0000,
        0xffff_ffff,
    )]);

    fn range_has_access(
        &self,
        access_type: MemoryRegionType,
        start_addr: usize,
        end_addr: usize,
    ) -> bool {
        let validation_region = MemoryRegion::new(access_type, start_addr, end_addr);
        MemoryRegion::regions_have_access(&self.regions, &validation_region)
    }
}
