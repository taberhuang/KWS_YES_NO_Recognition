.. _module-pw_containers-vectors:

=======
Vectors
=======
.. pigweed-module-subpage::
   :name: pw_containers

A vector is a one-dimensional array with a variable length.

----------
pw::Vector
----------
:doxylink:`Vector` class is similar to ``std::vector``, except it is backed by a
fixed-size buffer.

Vectors must be declared with an explicit maximum size
(e.g. ``Vector<int, 10>``) but vectors can be used and referred to without the
max size template parameter (e.g. ``Vector<int>``).

To allow referring to a ``pw::Vector`` without an explicit maximum size, all
Vector classes inherit from the generic ``Vector<T>``, which stores the maximum
size in a variable. This allows Vectors to be used without having to know
their maximum size at compile time. It also keeps code size small since
function implementations are shared for all maximum sizes.

Example
=======
.. literalinclude:: examples/vector.cc
   :language: cpp
   :linenos:
   :start-after: [pw_containers-vector]
   :end-before: [pw_containers-vector]

Size report
===========
The tables below illustrate the following scenarios:

* The memory and code size cost incurred by a adding a single ``Vector``.
* The memory and code size cost incurred by adding another ``Vector`` with the
  same type as the first scenario, but with a different size. As ``Vector``
  is templated on both type and size, a different size results in additional
  code being generated.
* The memory and code size cost incurred by adding another ``Vector`` with the
  same size as the first scenario, but with a different type. As ``Vector``
  is templated on both type and size, a different size results in additional
  code being generated.

.. include:: vectors_size_report

-----------------
pw::DynamicVector
-----------------
:doxylink:`DynamicVector` is similar to :doxylink:`Vector`, except that
it uses :doxylink:`Allocator` for memory operations.

-------------
API reference
-------------
Moved: :doxylink:`pw_containers_vectors`
