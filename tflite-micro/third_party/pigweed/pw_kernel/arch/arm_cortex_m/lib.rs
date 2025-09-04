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

use core::arch::asm;

use kernel::{Kernel, KernelState};
use pw_log::info;

mod exceptions;
mod protection;
mod regs;
mod spinlock;
mod syscall;
mod threads;
mod timer;

// Re-exports to conform to simplify public API.
pub use protection::MemoryConfig;
pub use spinlock::BareSpinLock;
pub use threads::ArchThreadState;

#[derive(Copy, Clone, Default)]
pub struct Arch;

kernel::impl_thread_arg_for_default_zst!(Arch);

impl Kernel for Arch {
    fn get_state(self) -> &'static KernelState<Arch> {
        static STATE: KernelState<Arch> = KernelState::new();
        &STATE
    }
}

fn ipsr_register_read() -> u32 {
    let ipsr: u32;
    // Note: cortex-m crate does not implement this register for some reason, so
    // read and mask manually
    unsafe {
        asm!("mrs {ipsr}, ipsr", ipsr = out(reg) ipsr);
    }
    ipsr
}

// Utility function to read whether or not the cpu considers itself in a handler
fn in_interrupt_handler() -> bool {
    // IPSR[8:0] is the current exception handler (or 0 if in thread mode)
    // TODO: konkers - Create register wrapper for IPSR.
    let current_exception = ipsr_register_read() & 0x1ff;

    // Treat SVCall (0xb) as in thread mode as we drop the SVCall pending bit
    // during system call execution to allow it to block and be preempted.
    // The code that manages this is in
    // [`crate::syscall::handle_syscall()`]
    current_exception != 0 && current_exception != 0xb
}

#[allow(dead_code)]
fn dump_int_pri() {
    info!(
        "basepri {} primask {}",
        cortex_m::register::basepri::read() as u8,
        cortex_m::register::primask::read().is_active() as u8
    );
}
