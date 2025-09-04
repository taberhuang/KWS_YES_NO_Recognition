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

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use foreign_box::{ForeignBox, ForeignRc};
use list::*;
use pw_log::info;
use pw_status::Result;

use crate::Kernel;
use crate::object::{KernelObject, ObjectTable};

/// The memory backing a thread's stack before it has been started.
///
/// After a thread has been started, ownership of its stack's memory is (from
/// the Rust Abstract Machine (AM) perspective) relinquished, and so the type we
/// use to represent that memory is irrelevant.
///
/// However, while we are initializing a thread in preparation for starting it,
/// we are operating on that memory as a normal Rust variable, and so its type
/// is important.
///
/// Using `MaybeUninit<u8>` instead of `u8` is important for two reasons:
/// - It ensures that it is sound to write values which are not entirely
///   initialized (e.g., which contain padding bytes).
/// - It ensures that pointers written to the stack [retain
///   provenance][provenance].
///
/// [provenance]: https://github.com/rust-lang/unsafe-code-guidelines/issues/286#issuecomment-2837585644
pub type StackStorage<const N: usize> = [MaybeUninit<u8>; N];

pub trait StackStorageExt {
    const ZEROED: Self;
}

impl<const N: usize> StackStorageExt for StackStorage<N> {
    const ZEROED: StackStorage<N> = [MaybeUninit::new(0); N];
}

#[derive(Clone, Copy)]
pub struct Stack {
    // Starting (lowest) address of the stack.  Inclusive.
    start: *const MaybeUninit<u8>,

    // Ending (highest) address of the stack.  Exclusive.
    end: *const MaybeUninit<u8>,
}

#[allow(dead_code)]
impl Stack {
    #[must_use]
    pub const fn from_slice(slice: &[MaybeUninit<u8>]) -> Self {
        let start: *const MaybeUninit<u8> = slice.as_ptr();
        // Safety: offset based on known size of slice.
        let end = unsafe { start.add(slice.len()) };
        Self { start, end }
    }

    #[must_use]
    const fn new() -> Self {
        Self {
            start: core::ptr::null(),
            end: core::ptr::null(),
        }
    }

    #[must_use]
    pub fn start(&self) -> *const MaybeUninit<u8> {
        self.start
    }

    #[must_use]
    pub fn end(&self) -> *const MaybeUninit<u8> {
        self.end
    }

    /// # Safety
    /// Caller must ensure exclusive mutable access to underlying data
    #[must_use]
    pub unsafe fn end_mut(&self) -> *mut MaybeUninit<u8> {
        self.end as *mut MaybeUninit<u8>
    }

    #[must_use]
    pub fn contains(&self, ptr: *const MaybeUninit<u8>) -> bool {
        ptr >= self.start && ptr < self.end
    }

    #[must_use]
    pub fn aligned_stack_allocation_mut<T: Sized>(
        sp: *mut MaybeUninit<u8>,
        alignment: usize,
    ) -> *mut T {
        let sp = sp.wrapping_byte_sub(size_of::<T>());
        let offset = sp.align_offset(alignment);
        if offset > 0 {
            sp.wrapping_byte_sub(alignment - offset).cast()
        } else {
            sp.cast()
        }
    }

    pub fn aligned_stack_allocation<T: Sized>(
        sp: *mut MaybeUninit<u8>,
        alignment: usize,
    ) -> *const T {
        Self::aligned_stack_allocation_mut::<*mut T>(sp, alignment).cast()
    }
}

// TODO: want to name this ThreadState, but collides with ArchThreadstate
#[derive(Copy, Clone, PartialEq)]
pub(super) enum State {
    New,
    Initial,
    Ready,
    Running,
    Stopped,
    Waiting,
}

// TODO: use From or Into trait (unclear how to do it with 'static str)
pub(super) fn to_string(s: State) -> &'static str {
    match s {
        State::New => "New",
        State::Initial => "Initial",
        State::Ready => "Ready",
        State::Running => "Running",
        State::Stopped => "Stopped",
        State::Waiting => "Waiting",
    }
}

pub trait ThreadState: 'static + Sized {
    const NEW: Self;

    // TODO: Maybe have a `MemoryConfigContext` super-trait of `ThreadState`?
    type MemoryConfig: crate::memory::MemoryConfig;

    /// Initialize the default frame of a kernel thread
    ///
    /// Arranges for the thread to start at `initial_function` with arguments
    /// passed in the first two argument slots.  The stack pointer of the thread
    /// is set to the top of the kernel stack.
    fn initialize_kernel_frame(
        &mut self,
        kernel_stack: Stack,
        memory_config: *const Self::MemoryConfig,
        initial_function: extern "C" fn(usize, usize, usize),
        args: (usize, usize, usize),
    );

    /// Initialize the default frame of a user thread
    ///
    /// Arranges for the thread to start at `initial_function` with arguments
    /// passed in the first two argument slots
    #[cfg(feature = "user_space")]
    fn initialize_user_frame(
        &mut self,
        kernel_stack: Stack,
        memory_config: *const Self::MemoryConfig,
        initial_sp: usize,
        initial_pc: usize,
        args: (usize, usize, usize),
    ) -> Result<()>;
}

pub struct Process<K: Kernel> {
    // List of the processes in the system
    pub link: Link,

    // TODO - konkers: allow this to be tokenized.
    pub name: &'static str,

    memory_config: <K::ThreadState as ThreadState>::MemoryConfig,

    object_table: ForeignBox<dyn ObjectTable<K>>,

    thread_list: UnsafeList<Thread<K>, ProcessThreadListAdapter<K>>,
}

list::define_adapter!(pub ProcessListAdapter<K: Kernel> => Process<K>::link);

impl<K: Kernel> Process<K> {
    /// Creates a new, empty, unregistered process.
    #[must_use]
    pub const fn new(
        name: &'static str,
        memory_config: <K::ThreadState as ThreadState>::MemoryConfig,
        object_table: ForeignBox<dyn ObjectTable<K>>,
    ) -> Self {
        Self {
            link: Link::new(),
            name,
            memory_config,
            object_table,
            thread_list: UnsafeList::new(),
        }
    }

    pub fn get_object(
        &self,
        kernel: K,
        handle: u32,
    ) -> Option<ForeignRc<K::AtomicUsize, dyn KernelObject<K>>> {
        self.object_table.get_object(kernel, handle)
    }

    /// Registers process with scheduler.
    pub fn register(&mut self, kernel: K) {
        unsafe {
            kernel
                .get_scheduler()
                .lock(kernel)
                .add_process_to_list(self);
        }
    }

    pub fn add_to_thread_list(&mut self, thread: &mut Thread<K>) {
        unsafe {
            self.thread_list.push_front_unchecked(thread);
        }
    }

    /// A simple ID for debugging purposes, currently the pointer to the thread
    /// structure itself.
    ///
    /// # Safety
    ///
    /// The returned value should not be relied upon as being a valid pointer.
    /// Even in the current implementation, `id` does not expose the pointer's
    /// provenance.
    #[must_use]
    pub fn id(&self) -> usize {
        core::ptr::from_ref(self).addr()
    }

    pub fn dump(&self) {
        info!("process {} ({:#x})", self.name as &str, self.id() as usize);
        unsafe {
            let _ = self
                .thread_list
                .for_each(|thread| -> core::result::Result<(), ()> {
                    thread.dump();
                    Ok(())
                });
        }
    }
}

pub struct Thread<K: Kernel> {
    // List of threads in a given process.
    pub process_link: Link,

    // Active state link (run queue, wait queue, etc)
    pub active_link: Link,

    // Safety: All accesses to the parent process must be done with the
    // scheduler lock held.
    process: *mut Process<K>,

    pub(super) state: State,
    stack: Stack,

    // Architecturally specific thread state, saved on context switch
    pub arch_thread_state: UnsafeCell<K::ThreadState>,

    // TODO - konkers: allow this to be tokenized.
    pub name: &'static str,
}

list::define_adapter!(pub ThreadListAdapter<K: Kernel> => Thread<K>::active_link);
list::define_adapter!(pub ProcessThreadListAdapter<K: Kernel> => Thread<K>::process_link);

impl<K: Kernel> Thread<K> {
    // Create an empty, uninitialzed thread
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Thread {
            process_link: Link::new(),
            active_link: Link::new(),
            process: core::ptr::null_mut(),
            state: State::New,
            arch_thread_state: UnsafeCell::new(K::ThreadState::NEW),
            stack: Stack::new(),
            name,
        }
    }

    pub fn get_object(
        &self,
        kernel: K,
        handle: u32,
    ) -> Option<ForeignRc<K::AtomicUsize, dyn KernelObject<K>>> {
        // SAFETY: `self.process` will always outlive `self`.
        unsafe { self.process.as_ref()? }.get_object(kernel, handle)
    }

    extern "C" fn trampoline<A1: ThreadArg>(entry_point: usize, arg0: usize, arg1: usize) {
        let entry_point = core::ptr::with_exposed_provenance::<()>(entry_point);
        // SAFETY: This function is only ever passed to the
        // architecture-specific call to `initialize_frame` below. It is
        // never called directly. In `initialize_frame`, the first argument
        // is `entry_point as usize`. `entry_point` is a `fn(usize)`. Thus,
        // this transmute preserves validity, and the preceding
        // `with_exposed_provenance` ensures that the resulting `fn(usize)`
        // has valid provenance for its referent.
        let entry_point: fn(K, A1) = unsafe { core::mem::transmute(entry_point) };
        let kernel = unsafe { K::from_usize(arg0) };
        let arg1 = unsafe { A1::from_usize(arg1) };
        entry_point(kernel, arg1);
    }

    pub fn initialize_kernel_thread<A: ThreadArg>(
        &mut self,
        kernel: K,
        kernel_stack: Stack,
        entry_point: fn(K, A),
        arg: A,
    ) -> &mut Thread<K> {
        pw_assert::assert!(self.state == State::New);
        let process = kernel.get_scheduler().lock(kernel).kernel_process.get();
        let args = (entry_point as usize, kernel.into_usize(), arg.into_usize());
        unsafe {
            (*self.arch_thread_state.get()).initialize_kernel_frame(
                kernel_stack,
                &raw const (*process).memory_config,
                Self::trampoline::<A>,
                args,
            );
        }

        unsafe { self.initialize(kernel, process, kernel_stack) }
    }

    #[cfg(feature = "user_space")]
    /// # Safety
    /// It is up to the caller to ensure that *process is valid.
    /// Initialize the mutable parts of the non privileged thread, must be
    /// called once per thread prior to starting it
    pub unsafe fn initialize_non_priv_thread(
        &mut self,
        kernel: K,
        kernel_stack: Stack,
        initial_sp: usize,
        process: *mut Process<K>,
        initial_pc: usize,
        args: (usize, usize, usize),
    ) -> Result<&mut Thread<K>> {
        pw_assert::assert!(self.state == State::New);

        unsafe {
            (*self.arch_thread_state.get()).initialize_user_frame(
                kernel_stack,
                &raw const (*process).memory_config,
                // be passed in from user space.
                initial_sp,
                initial_pc,
                args,
            )?;
        }
        unsafe { Ok(self.initialize(kernel, process, kernel_stack)) }
    }

    /// # Preconditions
    /// `self.arch_thread_state` is initialized before calling this function.
    ///
    /// # Safety
    /// It is up to the caller to ensure that *process is valid.
    /// Initialize the mutable parts of the thread, must be called once per
    /// thread prior to starting it
    unsafe fn initialize(
        &mut self,
        kernel: K,
        process: *mut Process<K>,
        kernel_stack: Stack,
    ) -> &mut Thread<K> {
        self.stack = kernel_stack;
        self.process = process;

        self.state = State::Initial;

        let _sched_state = kernel.get_scheduler().lock(kernel);
        unsafe {
            // Safety: *process is only accessed with the scheduler lock held.

            // Assert that the parent process is added to the scheduler.
            pw_assert::assert!(
                (*process).link.is_linked(),
                "Tried to add a Thread to an unregistered Process"
            );
            // Add thread to processes thread list.
            (*process).add_to_thread_list(self);
        }

        self
    }

    // Dump to the console useful information about this thread
    #[allow(dead_code)]
    pub fn dump(&self) {
        info!(
            "- thread {} ({:#x}) state {}",
            self.name as &str,
            self.id() as usize,
            to_string(self.state) as &str
        );
    }

    /// A simple ID for debugging purposes, currently the pointer to the thread
    /// structure itself.
    ///
    /// # Safety
    ///
    /// The returned value should not be relied upon as being a valid pointer.
    /// Even in the current implementation, `id` does not expose the pointer's
    /// provenance.
    #[must_use]
    pub fn id(&self) -> usize {
        core::ptr::from_ref(self).addr()
    }

    // An ID that can not be assigned to any thread in the system.
    #[must_use]
    pub const fn null_id() -> usize {
        // `core::ptr::null::<Self>() as usize` can not be evaluated at const time
        // and a null pointer is defined to be at address 0 (see
        // https://doc.rust-lang.org/beta/core/ptr/fn.null.html).
        0usize
    }
}

pub use arg::ThreadArg;
mod arg {
    pub trait ThreadArg {
        fn into_usize(self) -> usize;

        /// # Safety
        ///
        /// `u` must have previously been returned by [`x.into_usize()`]. The
        /// returned `Self` is guaranteed to be equal to `x`.
        ///
        /// [`x.into_usize()`]: ThreadArg::into_usize
        unsafe fn from_usize(u: usize) -> Self;
    }

    impl ThreadArg for usize {
        fn into_usize(self) -> usize {
            self
        }

        unsafe fn from_usize(u: usize) -> Self {
            u
        }
    }

    impl<T> ThreadArg for *const T {
        fn into_usize(self) -> usize {
            self.expose_provenance()
        }

        unsafe fn from_usize(u: usize) -> Self {
            core::ptr::with_exposed_provenance(u)
        }
    }

    impl<T> ThreadArg for *mut T {
        fn into_usize(self) -> usize {
            self.expose_provenance()
        }

        unsafe fn from_usize(u: usize) -> Self {
            core::ptr::with_exposed_provenance_mut(u)
        }
    }

    #[allow(clippy::needless_lifetimes)]
    impl<'a, T> ThreadArg for &'a T {
        fn into_usize(self) -> usize {
            let s: *const T = self;
            s.into_usize()
        }

        unsafe fn from_usize(u: usize) -> Self {
            // SAFETY: The caller promises that `u` was previously returned by
            // `into_usize`, which is implemented as `<*const T as
            // ThreadArg>::into_usize`. Thus, `u` was previously returned by
            // `<*const T as ThreadArg>::into_usize`.
            let ptr = unsafe { <*const T>::from_usize(u) };
            // SAFETY: By the preceding safety comment, `ptr` is equal to `self
            // as *const T` where `self: &'a T`, including provenance.
            unsafe { &*ptr }
        }
    }

    #[allow(clippy::needless_lifetimes)]
    impl<'a, T> ThreadArg for &'a mut T {
        fn into_usize(self) -> usize {
            let s: *mut T = self;
            s.into_usize()
        }

        unsafe fn from_usize(u: usize) -> Self {
            // SAFETY: The caller promises that `u` was previously returned by
            // `into_usize`, which is implemented as `<*mut T as
            // ThreadArg>::into_usize`. Thus, `u` was previously returned by
            // `<*mut T as ThreadArg>::into_usize`.
            let ptr = unsafe { <*mut T>::from_usize(u) };
            // SAFETY: By the preceding safety comment, `ptr` is equal to `self
            // as *mut T` where `self: &'a mut T`, including provenance. Since
            // `into_usize` consumes `self` by value, no other references to the
            // same referent exist, and so mutable aliasing is satisfied.
            unsafe { &mut *ptr }
        }
    }

    #[macro_export]
    macro_rules! impl_thread_arg_for_default_zst {
        ($t:ty) => {
            const _: () = assert!(size_of::<$t>() == 0);
            impl $crate::scheduler::thread::ThreadArg for $t {
                fn into_usize(self) -> usize {
                    0
                }

                unsafe fn from_usize(_u: usize) -> Self {
                    // SAFETY: We asserted above that `size_of::<$t>() == 0`, so
                    // there is only one value of `Self`. Thus, this
                    // implementation of `from_usize` returns the same value
                    // passed to any call to `into_usize`, as there is only one
                    // possible such value.
                    <$t as Default>::default()
                }
            }
        };
    }
}

/// Constructs a new [`Thread`] in global static storage.
///
/// # Safety
///
/// Each invocation of `init_thread!` must be executed at most once at run time.
// TODO: davidroth - Add const assertions to ensure stack sizes aren't too
// small, once the sizing analysis has been done to understand what a reasonable
// minimum is.
#[macro_export]
macro_rules! init_thread {
    ($name:literal, $entry:expr, $stack_size:expr $(,)?) => {{
        use $crate::__private::foreign_box::ForeignBox;
        use $crate::scheduler::thread::{Stack, StackStorage, StackStorageExt, Thread};
        use $crate::static_mut_ref;

        /// SAFETY: This must be executed at most once at run time.
        unsafe fn __init_thread() -> ForeignBox<Thread<arch::Arch>> {
            info!("allocating thread: {}", $name as &'static str);
            // SAFETY: The caller promises that this function will be executed
            // at most once.
            let thread = unsafe { static_mut_ref!(Thread<arch::Arch> = Thread::new($name)) };
            let mut thread = ForeignBox::from(thread);

            info!("initializing thread: {}", $name as &'static str);
            thread.initialize_kernel_thread(
                $crate::arch::Arch,
                // SAFETY: The caller promises that this function will be
                // executed at most once.
                Stack::from_slice(unsafe { static_mut_ref!(StackStorage<{ $stack_size }> = StackStorageExt::ZEROED)}),
                $entry,
                0
            );

            thread
        }

        __init_thread()
    }};
}

/// Constructs a new [`Process`] in global static storage and registers it.
///
/// # Safety
///
/// Each invocation of `init_non_priv_process!` must be executed at most once at
/// run time.
#[cfg(feature = "user_space")]
#[macro_export]
macro_rules! init_non_priv_process {
    ($name:literal, $memory_config:expr, $object_table:expr $(,)?) => {{
        use $crate::scheduler::thread::Process;

        /// SAFETY: This must be executed at most once at run time.
        unsafe fn __init_non_priv_process() -> &'static mut Process<arch::Arch> {
            use pw_log::info;
            info!(
                "allocating non-privileged process: {}",
                $name as &'static str
            );

            // SAFETY: The caller promises that this function will be executed
            // at most once.
            let proc =
                unsafe {
                    $crate::static_mut_ref!(Process<arch::Arch> =
                         Process::new($name, $memory_config, $object_table))
                };            proc.register(arch::Arch);
            proc
        }

        __init_non_priv_process()
    }};
}

/// Constructs a new [`Thread`] in global static storage and registers it.
///
/// # Safety
///
/// Each invocation of `init_non_priv_thread!` must be executed at most once at
/// run time.
#[cfg(feature = "user_space")]
#[macro_export]
macro_rules! init_non_priv_thread {
    ($name:literal, $process:expr, $entry:expr, $initial_sp:expr, $kernel_stack_size:expr  $(,)?) => {{
        use $crate::static_mut_ref;
        use $crate::__private::foreign_box::ForeignBox;
        use $crate::scheduler::thread::{Process, Stack, StackStorage, StackStorageExt, Thread};

        /// SAFETY: This must be executed at most once at run time.
        unsafe fn __init_non_priv_thread(
            proc: &mut Process<arch::Arch>,
            entry: usize,
            initial_sp: usize,
        ) -> ForeignBox<Thread<arch::Arch>> {
            use pw_log::info;
            info!(
                "allocating non-privileged thread: {}, entry {:#x}",
                $name as &'static str, entry as usize
            );
            // SAFETY: The caller promises that this function will be executed
            // at most once.
            let thread = unsafe { static_mut_ref!(Thread<arch::Arch> = Thread::new($name)) };
            let mut thread = ForeignBox::from(thread);

            info!(
                "initializing non-privileged thread: {}",
                $name as &'static str
            );
            unsafe {
                if let Err(e) = thread.initialize_non_priv_thread(
                    arch::Arch,
                    // SAFETY: The caller promises that this function will be
                    // executed at most once.
                    Stack::from_slice(unsafe { static_mut_ref!(StackStorage<{ $kernel_stack_size }> = StackStorageExt::ZEROED)}),
                    initial_sp,
                    proc,
                    entry,
                    (0, 0, 0)
                ) {
                    $crate::macro_exports::pw_assert::panic!(
                        "Error initializing thread: {}: {}",
                        $name as &'static str,
                        e as u32
                    );
                }
            }

            thread
        }

        __init_non_priv_thread($process, $entry, $initial_sp)
    }};
}

/// Initializes a thread in the given storage.
pub fn init_thread_in<K: Kernel, T: ThreadArg, const STACK_SIZE: usize>(
    kernel: K,
    thread: &'static mut Thread<K>,
    stack: &'static mut StackStorage<STACK_SIZE>,
    name: &'static str,
    entry: fn(K, T),
    arg: T,
) -> ForeignBox<Thread<K>> {
    *thread = Thread::new(name);
    let mut thread = ForeignBox::from(thread);
    thread.initialize_kernel_thread(kernel, Stack::from_slice(&*stack), entry, arg);
    thread
}
