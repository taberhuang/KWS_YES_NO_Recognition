.. _module-pw_allocator-api:

=============
API reference
=============
.. pigweed-module-subpage::
   :name: pw_allocator

This module provides the following:

- Generic allocator interfaces that can be injected into routines that need
  dynamic memory. These include :ref:`module-pw_allocator-api-allocator`, as
  well as the :ref:`module-pw_allocator-api-layout` type that is passed to it
  and the managed smart pointer types, such as
  :ref:`module-pw_allocator-api-unique_ptr`, that can be returned from it.
- Concrete allocator implementations used to provide memory dynamically.
- "Forwarding" allocators, as described by
  :ref:`module-pw_allocator-design-forwarding`.
- :ref:`module-pw_allocator-api-block` mix-in types and implementations.
- Additional allocator utility classes. These are typically used by allocator
  implementers.
- Test utilities for testing allocator implementations. These are typically used
  by allocator implementers.

---------------
Core interfaces
---------------
This module defines several types as part of a generic interface for memory
users.

.. _module-pw_allocator-api-layout:

Layout
======
A request for memory includes a requested size and alignment as a ``Layout``:

.. doxygenclass:: pw::allocator::Layout
   :members:

.. _module-pw_allocator-api-allocator:

Allocator
=========
``Allocator`` is the most commonly used interface. It can be used to request
memory of different layouts:

.. doxygenclass:: pw::Allocator
   :members:

.. _module-pw_allocator-api-pool:

Pool
====
``Pool`` differs from ``Allocator`` in that it can be used to request memory of
a single, fixed layout:

.. doxygenclass:: pw::allocator::Pool
   :members:

.. _module-pw_allocator-api-deallocator:

Deallocator
===========
Both ``Allocator`` and ``Pool`` derive from and extend ``Deallocator``. This
type is intended for allocator implementers and not for module consumers.

.. doxygenclass:: pw::Deallocator
   :members:

.. _module-pw_allocator-api-capabilities:

Capabilities
============
Types deriving from ``Deallocator`` can communicate about their optional
methods and behaviors using ``Capabilities``. This type is intended for
allocator implementers and not for module consumers.

.. doxygenclass:: pw::allocator::Capabilities
   :members:

.. _module-pw_allocator-api-unique_ptr:

UniquePtr
=========
.. doxygenclass:: pw::UniquePtr
   :members:

SharedPtr
=========
.. doxygenclass:: pw::SharedPtr
   :members:

WeakPtr
=========
.. doxygenclass:: pw::WeakPtr
   :members:

--------------------
Module configuration
--------------------

.. _module-pw_allocator-config-block_poison_interval:

PW_ALLOCATOR_BLOCK_POISON_INTERVAL
==================================
.. doxygendefine:: PW_ALLOCATOR_BLOCK_POISON_INTERVAL

.. _module-pw_allocator-config-hardening:

PW_ALLOCATOR_HARDENING
======================
.. doxygendefine:: PW_ALLOCATOR_HARDENING

.. _module-pw_allocator-config-suppress_deprecated_warnings:

PW_ALLOCATOR_SUPPRESS_DEPRECATED_WARNINGS
=========================================
.. doxygendefine:: PW_ALLOCATOR_SUPPRESS_DEPRECATED_WARNINGS

-------------------------
Allocator implementations
-------------------------
This module provides several concrete allocator implementations of the
:ref:`module-pw_allocator-api-allocator` interface:

.. _module-pw_allocator-api-block_allocator:

BlockAllocator
==============
Several allocators use :ref:`module-pw_allocator-api-block` types to manage
memory, and derive from this abstract base type.

.. doxygenclass:: pw::allocator::BlockAllocator
   :members:

.. _module-pw_allocator-api-best_fit_allocator:

BestFitAllocator
----------------
.. doxygenclass:: pw::allocator::BestFitAllocator
   :members:

.. _module-pw_allocator-api-bucket_block_allocator:

BucketAllocator
---------------
.. doxygenclass:: pw::allocator::BucketAllocator
   :members:

.. _module-pw_allocator-api-dl_allocator:

DlAllocator
-------------
.. doxygenclass:: pw::allocator::DlAllocator
   :members:

.. _module-pw_allocator-api-first_fit_allocator:

FirstFitAllocator
-----------------
.. doxygenclass:: pw::allocator::FirstFitAllocator
   :members:

.. _module-pw_allocator-api-tlsf_allocator:

TlsfAllocator
-------------
.. doxygenclass:: pw::allocator::TlsfAllocator
   :members:

.. _module-pw_allocator-api-worst_fit_allocator:

WorstFitAllocator
-----------------
.. doxygenclass:: pw::allocator::WorstFitAllocator
   :members:

.. _module-pw_allocator-api-buddy_allocator:

BuddyAllocator
==============
.. doxygenclass:: pw::allocator::BuddyAllocator
   :members:

.. _module-pw_allocator-api-bump_allocator:

BumpAllocator
=============
.. doxygenclass:: pw::allocator::BumpAllocator
   :members:

AsyncPool
=========
.. doxygenclass:: pw::allocator::AsyncPool
   :members:

.. _module-pw_allocator-api-chunk_pool:

ChunkPool
=========
.. doxygenclass:: pw::allocator::ChunkPool
   :members:

.. _module-pw_allocator-api-libc_allocator:

LibCAllocator
=============
.. doxygenclass:: pw::allocator::LibCAllocator
   :members:

.. _module-pw_allocator-api-null_allocator:

NullAllocator
=============
.. doxygenclass:: pw::allocator::NullAllocator
   :members:

.. _module-pw_allocator-api-typed_pool:

TypedPool
=========
.. doxygenclass:: pw::allocator::TypedPool
   :members:

---------------------
Forwarding Allocators
---------------------
This module provides several "forwarding" allocators, as described in
:ref:`module-pw_allocator-design-forwarding`.

.. _module-pw_allocator-api-allocator_as_pool:

AllocatorAsPool
===============
.. doxygenclass:: pw::allocator::AllocatorAsPool
   :members:

.. _module-pw_allocator-api-fallback_allocator:

FallbackAllocator
=================
.. doxygenclass:: pw::allocator::FallbackAllocator
   :members:

.. _module-pw_allocator-api-guarded_allocator:

GuardedAllocator
================
.. doxygenclass:: pw::allocator::GuardedAllocator
   :members:

.. _module-pw_allocator-api-pmr_allocator:

PmrAllocator
============
.. doxygenclass:: pw::allocator::PmrAllocator
   :members:

.. _module-pw_allocator-api-synchronized_allocator:

SynchronizedAllocator
=====================
.. doxygenclass:: pw::allocator::SynchronizedAllocator
   :members:

.. _module-pw_allocator-api-tracking_allocator:

TrackingAllocator
=================
.. doxygenclass:: pw::allocator::TrackingAllocator
   :members:

.. _module-pw_allocator-api-block:

--------------------------------
Testing and Debugging Allocators
--------------------------------
This module provides a allocators for use in testing and debugging.

FaultInjectingAllocator
=======================
.. doxygenclass:: pw::allocator::test::FaultInjectingAllocator
   :members:

-----
Block
-----
A block is an allocatable region of memory, and is the fundamental type managed
by several of the block allocator implementations.

.. tip::
   Avoid converting pointers to allocations into block instances, even if
   you know your memory is coming from a ``BlockAllocator``. Breaking the
   abstraction in this manner will limit your flexibility to change to a
   different allocator in the future.

Block mix-ins
=============
Blocks are defined using several stateless "mix-in" interface types. These
provide specific functionality, while deferring the detailed representation of a
block to a derived type.

.. TODO(b/378549332): Add a diagram of mix-in relationships.

.. _module-pw_allocator-api-basic_block:

BasicBlock
----------
.. doxygenclass:: pw::allocator::BasicBlock
   :members:

.. _module-pw_allocator-api-contiguous_block:

ContiguousBlock
---------------
.. doxygenclass:: pw::allocator::ContiguousBlock
   :members:

.. _module-pw_allocator-api-allocatable_block:

AllocatableBlock
----------------
.. doxygenclass:: pw::allocator::AllocatableBlock
   :members:

.. _module-pw_allocator-api-alignable_block:

AlignableBlock
--------------
.. doxygenclass:: pw::allocator::AlignableBlock
   :members:

.. _module-pw_allocator-api-block_with_layout:

BlockWithLayout
---------------
.. doxygenclass:: pw::allocator::BlockWithLayout
   :members:

.. _module-pw_allocator-api-iterable_block:

IterableBlock
--------------------
.. doxygenclass:: pw::allocator::IterableBlock
   :members:

.. _module-pw_allocator-api-poisonable_block:

PoisonableBlock
---------------
.. doxygenclass:: pw::allocator::PoisonableBlock
   :members:

BlockResult
-----------
This type is not a block mix-in. It is used to communicate whether a method
succeeded, what block was produced or modified, and what side-effects the call
produced.

.. doxygenclass:: pw::allocator::BlockResult
   :members:

Block implementations
=====================
The following combine block mix-ins and provide both the methods they require as
well as a concrete representation of the data those methods need.

.. _module-pw_allocator-api-small_block:

SmallBlock
----------
This implementation includes just enough mix-ins for fixed-alignment
allocations.

.. doxygenclass:: pw::allocator::SmallBlock
   :members:

.. _module-pw_allocator-api-small_alignable_block:

SmallAlignableBlock
-------------------
This implementation includes just enough mix-ins for variable-alignment
allocations.

.. doxygenclass:: pw::allocator::SmallAlignableBlock
   :members:

.. _module-pw_allocator-api-tiny_block:

TinyBlock
---------
This implementation is similar to :ref:`module-pw_allocator-api-small_block`,
but packs its information into just 4 bytes of overhead per allocation. This
constrains both its miniumum and maximum allocatable sizes, and incurs small
code size and performance costs for packing and unpacking header information.

.. doxygenclass:: pw::allocator::TinyBlock
   :members:

.. _module-pw_allocator-api-detailed_block:

DetailedBlock
-------------
This implementation includes all block mix-ins. This makes it very flexible at
the cost of additional code size.

.. doxygenstruct:: pw::allocator::DetailedBlockParameters
   :members:

.. doxygenclass:: pw::allocator::DetailedBlockImpl
   :members:

.. _module-pw_allocator-api-bucket:

-------
Buckets
-------
Several block allocator implementations improve performance by managing buckets,
which are data structures that track free blocks. Several bucket implementations
are provided that trade off between performance and per-block space needed when
free.

.. _module-pw_allocator-api-bucket_base:

BucketBase
==========
This type is not a standalone bucket, but a CRTP-style base class that provides
the common interface for other blocks.

.. doxygenclass:: pw::allocator::internal::BucketBase
   :members:

FastSortedBucket
================
.. doxygenclass:: pw::allocator::FastSortedBucket
   :members:

ForwardSortedBucket
===================
.. doxygenclass:: pw::allocator::ForwardSortedBucket
   :members:

ReverseFastSortedBucket
=======================
.. doxygenclass:: pw::allocator::ReverseFastSortedBucket
   :members:

ReverseSortedBucket
===================
.. doxygenclass:: pw::allocator::ReverseSortedBucket
   :members:

SequencedBucket
===============
.. doxygenclass:: pw::allocator::SequencedBucket
   :members:

UnorderedBucket
===============
.. doxygenclass:: pw::allocator::UnorderedBucket
   :members:

---------------
Utility Classes
---------------
In addition to providing allocator implementations themselves, this module
includes some utility classes.

.. _module-pw_allocator-api-metrics_adapter:

Metrics
=======
.. doxygenclass:: pw::allocator::internal::Metrics
   :members:

This class is templated on a ``MetricsType`` struct. See
:ref:`module-pw_allocator-design-metrics` for additional details on how the
struct, this class, and :ref:`module-pw_allocator-api-tracking_allocator`
interact.

Module consumers can define their own metrics structs using the
following macros:

.. doxygendefine:: PW_ALLOCATOR_METRICS_DECLARE
.. doxygendefine:: PW_ALLOCATOR_METRICS_ENABLE

.. _module-pw_allocator-api-fragmentation:

Fragmentation
=============
.. doxygenstruct:: pw::allocator::Fragmentation
   :members:


Buffer management
=================
.. doxygenclass:: pw::allocator::WithBuffer
   :members:

.. _module-pw_allocator-api-size_reports:

------------
Size reports
------------
This module includes utilities to help generate code size reports for allocator
implementations. These are used to generate the code size reports for the
allocators provided by this module, and can also be used to evaluate your own
custom allocator implementations.

.. doxygenfunction:: pw::allocator::size_report::GetBuffer
.. doxygenfunction:: pw::allocator::size_report::MeasureAllocator

------------
Test support
------------
This module includes test utilities for allocator implementers. These
facilitate writing unit tests and fuzz tests for both concrete and forwarding
allocator implementations. They are not intended to be used by module consumers.

.. _module-pw_allocator-api-allocator_for_test:

AllocatorForTest
================
.. doxygenclass:: pw::allocator::test::AllocatorForTest
   :members:

.. _module-pw_allocator-api-test_harness:

TestHarness
===========
.. doxygenclass:: pw::allocator::test::TestHarness
   :members:

.. _module-pw_allocator-api-fuzzing_support:

FuzzTest support
================
.. doxygenfunction:: pw::allocator::test::ArbitraryRequest
.. doxygenfunction:: pw::allocator::test::ArbitraryRequests
.. doxygenfunction:: pw::allocator::test::MakeRequest
