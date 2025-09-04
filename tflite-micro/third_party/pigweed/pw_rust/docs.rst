.. _module-pw_rust:

=======
pw_rust
=======
.. pigweed-module::
   :name: pw_rust

Rust support in pigweed is **highly** experimental.  Currently functionality
is split between Bazel and GN support.

-----
Bazel
-----
Bazel support is based on `rules_rust <https://github.com/bazelbuild/rules_rust>`_
and supports a rich set of targets for both host and target builds.

Building and Running the Embedded Examples
==========================================
The examples can be built for both the ``lm3s6965evb`` and ``microbit``
QEMU machines.  The examples can be built and run using the following commands
where ``PLATFORM`` is one of ``lm3s6965evb`` or ``microbit``.

embedded_hello
--------------
.. code-block:: bash

   $ bazel build //pw_rust/examples/embedded_hello:hello \
     --platforms //pw_build/platforms:${PLATFORM}

   $ qemu-system-arm \
     -machine ${PLATFORM} \
     -nographic \
     -semihosting-config enable=on,target=native \
     -kernel ./bazel-bin/pw_rust/examples/embedded_hello/hello
   Hello, Pigweed!


tokenized_logging
-----------------
.. code-block:: bash

   $ bazel build //pw_rust/examples/tokenized_logging:tokenized_logging \
     --//pw_log/rust:pw_log_backend=//pw_rust/examples/tokenized_logging:pw_log_backend\
     --platforms //pw_build/platforms:${PLATFORM}

   $ qemu-system-arm \
     -machine ${PLATFORM} \
     -nographic \
     -semihosting-config enable=on,target=native \
     -kernel ./bazel-bin/pw_rust/examples/tokenized_logging/tokenized_logging \
     | python -m pw_tokenizer.detokenize \
     base64 \
     ./bazel-bin/pw_rust/examples/tokenized_logging/tokenized_logging

--
GN
--
In GN, currently only building a single host binary using the standard
libraries is supported.  Windows builds are currently unsupported.

Building
========
To build the sample rust targets, you need to enable
``pw_rust_ENABLE_EXPERIMENTAL_BUILD``:

.. code-block:: bash

   $ gn gen out --args="pw_rust_ENABLE_EXPERIMENTAL_BUILD=true"

Once that is set, you can build and run the ``hello`` example:

.. code-block:: bash

   $ ninja -C out host_clang_debug/obj/pw_rust/examples/basic_executable/bin/hello
   $ ./out/host_clang_debug/obj/pw_rust/examples/basic_executable/bin/hello
   Hello, Pigweed!

no_std
------
Set ``pw_rust_USE_STD = false`` in the toolchain configuration, if the target
toolchain does not support ``std``.

``no_std`` toolchain builds target
``//pw_rust/examples/basic_executable/bin/hello_pw_log``. It also prints
"Hello, Pigweed!", but links and uses ``pw_log`` C++ backend.

------------------
Third Party Crates
------------------
Thrid party crates are vendored in the
`third_party/rust_crates <https://pigweed.googlesource.com/third_party/rust_crates>`_
respository.  Currently referencing these is only supported through the bazel
build.
