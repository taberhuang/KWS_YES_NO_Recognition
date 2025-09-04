.. _docs-bazel-integration-modules:

===================================
Use Pigweed modules in your project
===================================
If you're integrating Pigweed into an existing project using Bazel, you can get
started on this as soon as you've completed
:ref:`docs-bazel-integration-add-pigweed-as-a-dependency`.

Let's say you want to use ``pw::Vector`` from :ref:`module-pw_containers`, our
embedded-friendly replacement for ``std::vector``.

#. Include the header you want in your code:

   .. code-block:: cpp

      #include "pw_containers/vector.h"

#. Look at the module's `build file
   <https://cs.opensource.google/pigweed/pigweed/+/main:pw_containers/BUILD.bazel>`__
   to figure out which build target you need to provide the header and
   implementation. For ``pw_containers/vector.h``, it's
   ``//pw_containers:vector``.

#. Add this target to the ``deps`` of your
   `cc_library <https://bazel.build/reference/be/c-cpp#cc_library>`__ or
   `cc_binary <https://bazel.build/reference/be/c-cpp#cc_binary>`__:

   .. code-block:: python

      cc_library(
          name = "my_library",
          srcs  = ["my_library.cc"],
          hdrs = ["my_library.h"],
          deps = [
              "@pigweed//pw_containers:vector",  # <-- The new dependency
          ],
      )

#. Add a dependency on ``@pigweed//pw_build:default_link_extra_lib`` to your
   final *binary* target. See :ref:`docs-build_system-bazel_link-extra-lib`
   for a discussion of why this is necessary, and what the alternatives are.

   .. code-block:: python

      cc_binary(
          name = "my_binary",
          srcs  = ["my_binary.cc"],
          deps = [
              ":my_library",
              "@pigweed//pw_build:default_link_extra_lib",  # <-- The new dependency
          ],
      )

--------------------------------------------
Configure backends for facades you depend on
--------------------------------------------
Pigweed makes extensive use of :ref:`docs-facades`, and any module you choose
to use will likely have a transitive dependency on some facade (typically
:ref:`module-pw_assert` or :ref:`module-pw_log`). Continuing with our example,
``pw::Vector`` depends on :ref:`module-pw_assert`.

In Bazel, facades already have a default backend (implementation) that works
for host builds (builds targeting your local development machine). But to build
a binary for your embedded target, you'll need to select a suitable backend
yourself.

Fortunately, the default backend for :ref:`module-pw_assert` is
:ref:`module-pw_assert_basic`, which is a suitable place to start for most
embedded targets, too. But it depends on :ref:`module-pw_sys_io`, another
facade for which you *will* have to choose a backend yourself.

The simplest way to do so is to set the corresponding `label flag
<https://bazel.build/extending/config#label-typed-build-settings>`__ when
invoking Bazel. For example, to use the
:ref:`module-pw_sys_io_baremetal_stm32f429` backend for :ref:`module-pw_sys_io`
provided in upstream Pigweed:

.. code-block:: console

   $ bazel build \
       --@pigweed//targets/pw_sys_io_backend=@pigweed//pw_sys_io_baremetal_stm32f429 \
       //path/to/your:target

You can also define backends within your own project. (If Pigweed doesn't
include a :ref:`module-pw_sys_io` backend suitable for your embedded platform,
that's what you should do now.) See
:ref:`docs-build_system-bazel_configuration` for a tutorial that dives deeper
into facade configuration with Bazel.

----------
Next steps
----------
To ensure the correctness of your project's code, set up
:ref:`docs-automated-analysis`.
