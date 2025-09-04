.. _module-pw_third_party_freertos:

========
FreeRTOS
========
The ``//third_party/freertos`` directory in Pigweed contains build system
integration helpers for FreeRTOS.

-------------
Build Support
-------------
This module provides support to compile FreeRTOS with GN, CMake, and Bazel.
This is required when compiling backends modules for FreeRTOS.

GN
==
In order to use this you are expected to configure the following variables from
``$pw_external_freertos:freertos.gni``:

#. Set the GN ``dir_pw_third_party_freertos`` to the path of the FreeRTOS
   installation.
#. Set ``pw_third_party_freertos_CONFIG`` to a ``pw_source_set`` which provides
   the FreeRTOS config header.
#. Set ``pw_third_party_freertos_PORT`` to a ``pw_source_set`` which provides
   the FreeRTOS port specific includes and sources.

After this is done a ``pw_source_set`` for the FreeRTOS library is created at
``$pw_external_freertos``.

CMake
=====
In order to use this you are expected to set the following variables from
``third_party/freertos/CMakeLists.txt``:

#. Set ``dir_pw_third_party_freertos`` to the path of the FreeRTOS installation.
#. Set ``pw_third_party_freertos_CONFIG`` to a library target which provides
   the FreeRTOS config header.
#. Set ``pw_third_party_freertos_PORT`` to a library target which provides
   the FreeRTOS port specific includes and sources.

Bazel
=====
The FreeRTOS build is configured through `constraint_settings
<https://bazel.build/reference/be/platforms-and-toolchains#constraint_setting>`_.
The `platform <https://bazel.build/extending/platforms>`_ you are building for
must specify values for the following settings:

*   ``@freertos//:port``, to set which FreeRTOS port to use. You can
    select a value from those defined in
    ``third_party/freertos/freertos.BUILD.bazel`` (for example,
    ``@freertos//:port_ARM_CM4F``).
*   ``@freertos//:malloc``, to set which FreeRTOS malloc implementation to use.
    You can select a value from those defined in
    ``third_party/freertos/BUILD.bazel`` (for example,
    ``@freertos//:malloc_heap_1``).
*   ``@freertos//:disable_task_statics_setting``, to determine whether statics
    should be disabled during compilation of the tasks.c source file (see next
    section). This setting has only two possible values, also defined in
    ``third_party/freertos/BUILD.bazel``: ``@freertos//:disable_task_statics``
    and ``@freertos//:no_disable_task_statics``.

In addition, you need to set the ``@freertos//:freertos_config`` label flag to
point to the library target providing the FreeRTOS config header. See
:ref:`docs-build_system-bazel_configuration` for a discussion of how to work
with our label flags.


.. _third_party-freertos_disable_task_statics:

Linking against FreeRTOS kernel's static internals
==================================================
In order to link against internal kernel data structures through the use of
extern "C", statics can be optionally disabled for the tasks.c source file
to enable use of things like pw_thread_freertos/util.h's ``ForEachThread``.

To facilitate this, Pigweed offers an opt-in option which can be enabled,

*  in GN through ``pw_third_party_freertos_DISABLE_TASKS_STATICS = true``,
*  in CMake through ``set(pw_third_party_freertos_DISABLE_TASKS_STATICS ON
   CACHE BOOL "" FORCE)``,
*  in Bazel through ``@freertos//:disable_task_statics``.

This redefines ``static`` to nothing for the ``Source/tasks.c`` FreeRTOS source
file when building through ``$pw_external_freertos`` in GN and through
``pw_third_party.freertos`` in CMake.

.. attention:: If you use this, make sure that your FreeRTOSConfig.h and port
  does not rely on any statics inside of tasks.c. For example, you cannot use
  ``PW_CHECK`` for ``configASSERT`` when this is enabled.

As a helper ``PW_THIRD_PARTY_FREERTOS_NO_STATICS=1`` is defined when statics are
disabled to help manage conditional configuration.

We highly recommend :ref:`our configASSERT wrapper
<third_party-freertos_config_assert>` when  using this configuration, which
correctly sets ``configASSERT`` to use ``PW_CHECK`` and ``PW_ASSERT`` for you.

-----------------------------
OS Abstraction Layers Support
-----------------------------
Support for Pigweed's :ref:`docs-os` are provided for FreeRTOS via the following
modules:

* :ref:`module-pw_chrono_freertos`
* :ref:`module-pw_sync_freertos`
* :ref:`module-pw_thread_freertos`

Backend group
=============
In GN, import ``pw_targets_FREERTOS_BACKEND_GROUP`` to set backends for
:ref:`module-pw_chrono`, :ref:`module-pw_sync`, and :ref:`module-pw_thread` for
FreeRTOS. The backends can be overridden individually if needed.

.. code-block:: none

   # Toolchain configuration
   import("$dir_pigweed/targets/common/freertos.gni")

   _backend_setting_example = {
     # Since this target is using FreeRTOS, adopt FreeRTOS backends by default.
     forward_variables_from(pw_targets_FREERTOS_BACKEND_GROUP, "*")

     # Set other backends or override the default FreeRTOS selections if needed.
     ...
   }

.. _third_party-freertos_config_assert:

--------------------------
configASSERT and pw_assert
--------------------------
To make it easier to use :ref:`module-pw_assert` with FreeRTOS a helper header
is provided under ``pw_third_party/freertos/config_assert.h`` which defines
``configASSERT`` for you using Pigweed's assert system for your
``FreeRTOSConfig.h`` if you chose to use it.

.. code-block:: cpp

   // Instead of defining configASSERT, simply include this header in its place.
   #include "pw_third_party/freertos/config_assert.h"

---------------------------------------------
FreeRTOS application function implementations
---------------------------------------------
.. doxygengroup:: FreeRTOS_application_functions
