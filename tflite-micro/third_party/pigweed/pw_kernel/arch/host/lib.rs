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

#![no_std]

use kernel::scheduler::SchedulerState;
use kernel::scheduler::thread::{Stack, ThreadState};
use kernel::sync::spinlock::SpinLockGuard;
use kernel::{Arch, Kernel, KernelState, MemoryRegionType};
use pw_log::info;
use pw_status::Result;

mod spinlock;

#[derive(Copy, Clone, Default)]
pub struct HostArch;

kernel::impl_thread_arg_for_default_zst!(HostArch);

pub struct ArchThreadState;

impl Arch for HostArch {
    type ThreadState = ArchThreadState;
    type BareSpinLock = spinlock::BareSpinLock;
    type Clock = Clock;
    type AtomicUsize = core::sync::atomic::AtomicUsize;

    unsafe fn context_switch(
        self,
        _sched_state: SpinLockGuard<'_, Self, SchedulerState<Self>>,
        _old_thread_state: *mut ArchThreadState,
        _new_thread_state: *mut ArchThreadState,
    ) -> SpinLockGuard<'_, Self, SchedulerState<Self>> {
        pw_assert::panic!("unimplemented");
    }

    fn thread_local_state(self) -> &'static kernel::scheduler::ThreadLocalState<Self> {
        todo!("unimplemented");
    }

    fn now(self) -> time::Instant<Clock> {
        use time::Clock as _;
        Clock::now()
    }

    fn enable_interrupts(self) {
        todo!("unimplemented");
    }
    fn disable_interrupts(self) {
        todo!("");
    }
    fn interrupts_enabled(self) -> bool {
        todo!("");
    }

    fn early_init(self) {
        info!("HOST arch early init");
    }
    fn init(self) {
        info!("HOST arch init");
    }
}

impl Kernel for HostArch {
    fn get_state(self) -> &'static KernelState<HostArch> {
        static STATE: KernelState<HostArch> = KernelState::new();
        &STATE
    }
}

impl ThreadState for ArchThreadState {
    const NEW: Self = Self;
    type MemoryConfig = MemoryConfig;

    fn initialize_kernel_frame(
        &mut self,
        _kernel_stack: Stack,
        _memory_config: *const MemoryConfig,
        _initial_function: extern "C" fn(usize, usize, usize),
        _args: (usize, usize, usize),
    ) {
        pw_assert::panic!("unimplemented");
    }

    #[cfg(feature = "user_space")]
    fn initialize_user_frame(
        &mut self,
        _kernel_stack: Stack,
        _memory_config: *const MemoryConfig,
        _initial_sp: usize,
        _initial_pc: usize,
        _args: (usize, usize, usize),
    ) -> Result<()> {
        pw_assert::panic!("unimplemented");
    }
}

pub struct Clock;

impl time::Clock for Clock {
    const TICKS_PER_SEC: u64 = 1_000_000;

    fn now() -> time::Instant<Self> {
        time::Instant::from_ticks(0)
    }
}

pub struct MemoryConfig;

impl kernel::memory::MemoryConfig for MemoryConfig {
    const KERNEL_THREAD_MEMORY_CONFIG: Self = Self;

    fn range_has_access(
        &self,
        _access_type: MemoryRegionType,
        _start_addr: usize,
        _end_addr: usize,
    ) -> bool {
        false
    }
}
