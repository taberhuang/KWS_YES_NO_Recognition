.. _module-pw_multibuf:

===========
pw_multibuf
===========
.. pigweed-module::
   :name: pw_multibuf

Sending or receiving messages via RPC, transfer, or sockets often requires a
series of intermediate buffers, each requiring their own copy of the data.
``pw_multibuf`` allows data to be written *once*, eliminating the memory, CPU
and latency overhead of copying.

-----------------
How does it work?
-----------------
``pw_multibuf`` uses several techniques to minimize copying of data:

- **Header and Footer Reservation**: Lower-level components can reserve space
  within a buffer for headers and/or footers. This allows headers and footers
  to be added to user-provided data without moving users' data.
- **Native Scatter/Gather and Fragmentation Support**: Buffers can refer to
  multiple separate chunks of memory. Messages can be built up from
  discontiguous allocations, and users' data can be fragmented across multiple
  packets.
- **Divisible Memory Regions**: Incoming buffers can be divided without a copy,
  allowing incoming data to be freely demultiplexed.

-------------------------------
What kinds of data is this for?
-------------------------------
``pw_multibuf`` is best used in code that wants to read, write, or pass along
data which are one of the following:

- **Large**: ``pw_multibuf`` is designed to allow breaking up data into
  multiple chunks. It also supports asynchronous allocation for when there may
  not be sufficient space for incoming data.
- **Communications-Oriented**: Data which is being received or sent across
  sockets, various packets, or shared-memory protocols can benefit from the
  fragmentation, multiplexing, and header/footer-reservation properties of
  ``pw_multibuf``.
- **Copy-Averse**: ``pw_multibuf`` is structured to allow users to pass around
  and mutate buffers without copying or moving data in-memory. This can be
  especially useful when working in systems that are latency-sensitive,
  need to pass large amounts of data, or when memory usage is constrained.

---------------------------
Why are there two versions?
---------------------------
We are currently investigating an alternate approach that keeps many of the
aspects described above, while separating out the concern of memory allocation
and instead using ``MultiBuf`` to present different logical, span-like views of
sequences of buffers. This version is currently experimental, but we welcome
feedback on the direction it is taking!

-----------
MultiBuf v1
-----------

API Reference
=============
Most users of ``pw_multibuf`` will start by allocating a ``MultiBuf`` using
a ``MultiBufAllocator`` class, such as the ``SimpleAllocator``.

``MultiBuf`` s consist of a number of ``Chunk`` s of contiguous memory regions.
``Chunk`` s can be grown or shrunk which allows ``MultiBuf`` s to be grown or
shrunk. This allows, for example, lower layers to reserve part of a
``MultiBuf`` for a header or footer (see ``Chunk`` for more details).

``MultiBuf`` exposes an ``std::byte`` iterator interface as well as a ``Chunk``
iterator available through the ``Chunks()`` method. It allows extracting a
``Chunk`` as an RAII-style ``OwnedChunk`` which manages its own lifetime.

.. doxygenclass:: pw::multibuf::Chunk
   :members:

.. doxygenclass:: pw::multibuf::OwnedChunk
   :members:

.. doxygenclass:: pw::multibuf::MultiBuf
   :members:

.. doxygenfunction:: pw::multibuf::FromSpan

.. doxygenclass:: pw::multibuf::MultiBufChunks
   :members:

.. doxygenclass:: pw::multibuf::MultiBufAllocator
   :members:

.. doxygenclass:: pw::multibuf::MultiBufAllocatorAsync
   :members:

.. doxygenclass:: pw::multibuf::MultiBufAllocationFuture
   :members:

.. doxygenclass:: pw::multibuf::SimpleAllocator
   :members:

.. doxygenclass:: pw::multibuf::Stream
   :members:

Test-only features
------------------
.. doxygenclass:: pw::multibuf::test::SimpleAllocatorForTest
   :members:

Allocator Implementors' API
===========================
Some users will need to directly implement the ``MultiBufAllocator`` interface
in order to provide allocation out of a particular region, provide particular
allocation policy, fix Chunks to some size (such as MTU size - header for
socket implementations), or specify other custom behavior.

These users will also need to understand and implement the following APIs:

.. doxygenclass:: pw::multibuf::ChunkRegionTracker
   :members:

A simple implementation of a ``ChunkRegionTracker`` is provided, called
``HeaderChunkRegionTracker``. It stores its ``Chunk`` and region metadata in a
``Allocator`` allocation alongside the data. The allocation process is
synchronous, making this class suitable for testing. The allocated region or
``Chunk`` must not outlive the provided allocator.

.. doxygenclass:: pw::multibuf::HeaderChunkRegionTracker
   :members:

Another ``ChunkRegionTracker`` specialization is the lightweight
``SingleChunkRegionTracker``, which does not rely on ``Allocator`` and uses the
provided memory view to create a single chunk. This is useful when a single
``Chunk`` is sufficient at no extra overhead. However, the user needs to own
the provided memory and know when a new ``Chunk`` can be requested.

.. doxygenclass:: pw::multibuf::SingleChunkRegionTracker
   :members:

-----------
MultiBuf v2
-----------
Everything in this section is experimental and subject to change without
warning!

API Reference
=============
.. doxygengroup:: pw_multibuf
   :content-only:
   :members:

.. doxygenenum:: pw::MultiBufProperty

.. doxygenclass:: pw::MultiBufObserver
   :members:

.. toctree::
   :hidden:

   code_size
