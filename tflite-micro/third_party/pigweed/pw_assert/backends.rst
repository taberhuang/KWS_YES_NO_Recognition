.. _module-pw_assert-backends:

========
Backends
========
.. pigweed-module-subpage::
   :name: pw_assert

.. TODO: b/323607687 - Add backend guidance here

The following backends are already implemented and available for use in your
projects:

- ``pw_assert:print_and_abort_backend`` - **Stable** - Uses the ``printf`` and
  ``abort`` standard library functions to implement the assert facade. Prints
  the assert expression, evaluated arguments if any, file/line, function name,
  and user message, then aborts. Only suitable for targets that support these
  standard library functions.
- :ref:`module-pw_assert_basic` - **Stable** - The assert basic module is a
  simple assert handler that displays the failed assert line and the values of
  captured arguments. Output is directed to ``pw_sys_io``. This module is a
  great ready-to-roll module when bringing up a system, but is likely not the
  best choice for production.
- :ref:`module-pw_assert_log` - **Stable** - This assert backend redirects to
  logging, but with a logging flag set that indicates an assert failure. This
  is our advised approach to get **tokenized asserts**--by using tokenized
  logging, then using the ``pw_assert_log`` backend.

.. toctree::
   :maxdepth: 1

   Basic <../pw_assert_basic/docs>
   Pigweed logging <../pw_assert_log/docs>
   Fuchsia <../pw_assert_fuchsia/docs>
   Tokenized <../pw_assert_tokenized/docs>
   Trap <../pw_assert_trap/docs>
   Zephyr <../pw_assert_zephyr/docs>
