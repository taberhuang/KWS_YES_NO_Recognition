.. _module-pw_unit_test:

============
pw_unit_test
============
.. pigweed-module::
   :name: pw_unit_test

.. tab-set::

   .. tab-item:: mylib_test.cpp

      .. code-block:: c++

         #include "mylib.h"

         #include "pw_unit_test/framework.h"

         namespace {

         TEST(MyTestSuite, MyTestCase) {
           pw::InlineString<32> expected = "(╯°□°)╯︵ ┻━┻";
           pw::InlineString<32> actual = mylib::flip_table();
           EXPECT_STREQ(expected.c_str(), actual.c_str());
         }

         }

   .. tab-item:: BUILD.bazel

      .. code-block:: python

         load("@pigweed//pw_unit_test:pw_cc_test.bzl", "pw_cc_test")

         cc_library(
             name = "mylib",
             srcs = ["mylib.cc"],
             hdrs = ["mylib.h"],
             includes = ["."],
             deps = ["@pigweed//pw_string"],
         )

         pw_cc_test(
             name = "mylib_test",
             srcs = ["mylib_test.cc"],
             deps = [
                 "@pigweed//pw_unit_test",
                 ":mylib",
             ],
         )

   .. tab-item:: mylib.cc

      .. code-block:: c++

         #include "mylib.h"

         #include "pw_string/string.h"

         namespace mylib {

         pw::InlineString<32> flip_table() {
           pw::InlineString<32> textmoji = "(╯°□°)╯︵ ┻━┻";
           return textmoji;
         }

         }

   .. tab-item:: mylib.h

      .. code-block:: c++

         #include "pw_string/string.h"

         namespace mylib {

         pw::InlineString<32> flip_table();

         }

.. _GoogleTest: https://google.github.io/googletest/

``pw_unit_test`` provides a `GoogleTest`_-compatible unit testing framework for
Pigweed-based projects. The default backend is a lightweight subset of
GoogleTest that uses embedded-friendly primitives.

.. grid:: 1

   .. grid-item-card:: :octicon:`rocket` Quickstart
      :link: module-pw_unit_test-quickstart
      :link-type: ref
      :class-item: sales-pitch-cta-primary

      Set up your project for testing and learn testing basics.

.. grid:: 2

   .. grid-item-card:: :octicon:`list-unordered` Guides
      :link: module-pw_unit_test-guides
      :link-type: ref
      :class-item: sales-pitch-cta-secondary

      Learn how to do common tasks.

   .. grid-item-card:: :octicon:`code-square` C++ API reference
      :link: module-pw_unit_test-cpp
      :link-type: ref
      :class-item: sales-pitch-cta-secondary

      Get detailed C++ API reference information.

.. grid:: 2

   .. grid-item-card:: :octicon:`code-square` Bazel API reference
      :link: module-pw_unit_test-bazel
      :link-type: ref
      :class-item: sales-pitch-cta-secondary

      Get detailed Bazel API reference information.

   .. grid-item-card:: :octicon:`code-square` GN API reference
      :link: module-pw_unit_test-gn
      :link-type: ref
      :class-item: sales-pitch-cta-secondary

      Get detailed GN API reference information.

.. grid:: 2

   .. grid-item-card:: :octicon:`code-square` CMake API reference
      :link: module-pw_unit_test-cmake
      :link-type: ref
      :class-item: sales-pitch-cta-secondary

      Get detailed CMake API reference information.

   .. grid-item-card:: :octicon:`code-square` Python API reference
      :link: module-pw_unit_test-py
      :link-type: ref
      :class-item: sales-pitch-cta-secondary

      Get detailed Python API reference information.

.. _module-pw_unit_test-quickstart:

----------
Quickstart
----------

Set up your build system
========================
.. tab-set::

   .. tab-item:: Bazel

      Load the :ref:`module-pw_unit_test-pw_cc_test` rule and create a target
      that depends on ``@pigweed//pw_unit_test`` as well as the code you want
      to test:

      .. code-block:: python

         load("@pigweed//pw_unit_test:pw_cc_test.bzl", "pw_cc_test")

         cc_library(
             name = "mylib",
             srcs = ["mylib.cc"],
             hdrs = ["mylib.h"],
             includes = ["."],
             deps = ["..."],
         )

         pw_cc_test(
             name = "mylib_test",
             srcs = ["mylib_test.cc"],
             deps = [
                 "@pigweed//pw_unit_test",
                 ":mylib",
             ],
         )

      See also :ref:`module-pw_unit_test-bazel`.

   .. tab-item:: GN

      Import ``$dir_pw_unit_test/test.gni`` and create a ``pw_test`` rule that
      depends on the code you want to test:

      .. code-block:: python

         import("$dir_pw_unit_test/test.gni")

         pw_source_set("mylib") {
           sources = [ "mylib.cc" ]
         }

         pw_test("mylib_test") {
           sources = [ "mylib_test.cc" ]
           deps = [ ":mylib" ]
         }

      See :ref:`module-pw_unit_test-gn` for more information.

``pw_unit_test`` generates a simple ``main`` function for running tests on
:ref:`target-host`. See :ref:`module-pw_unit_test-main` to learn how to
create your own ``main`` function for running on-device tests.

Write tests
===========
Create test suites and test cases:

.. code-block:: c++

   #include "mylib.h"

   #include "pw_unit_test/framework.h"

   namespace {

   TEST(MyTestSuite, MyTestCase) {
     pw::InlineString<32> expected = "(╯°□°)╯︵ ┻━┻";
     pw::InlineString<32> actual = app::flip_table();
     EXPECT_STREQ(expected.c_str(), actual.c_str());
   }

   }  // namespace

``pw_unit_test`` provides a standard set of ``TEST``, ``EXPECT``, ``ASSERT``
and ``FAIL`` macros. The default backend, ``pw_unit_test:light``, offers an
embedded-friendly implementation which prioritizes small code size.

Alternativley, users can opt into a larger set of assertion macros, matchers,
and more detailed error messages by using the ``pw_unit_test:googletest``
backend. See :ref:`module-pw_unit_test-backends`.

See `GoogleTest Primer <https://google.github.io/googletest/primer.html>`_ for
the basics of using GoogleTest.

Run tests
=========
.. tab-set::

   .. tab-item:: Bazel

      .. code-block:: console

         $ bazel test //src:mylib_test

   .. tab-item:: GN

      Run the generated test binary:

      .. code-block:: console

         $ ./out/.../mylib_test

.. _module-pw_unit_test-guides:

------
Guides
------

.. _module-pw_unit_test-backends:

Choose a backend
================
The default backend, ``pw_unit_test:light``, is a lightweight subset of
GoogleTest that uses embedded-friendly primitives. It's also highly portable
because it offloads the responsibility of test reporting and output to the
underlying system, communicating its results through a common interface. This
lets you write unit tests once and run them under many different environments.

If the :ref:`subset <module-pw_unit_test-compatibility>` of GoogleTest that
``pw_unit_test:light`` supports doesn't meet your needs, you can access the
full upstream GoogleTest API through ``pw_unit_test:googletest``. See
:ref:`module-pw_unit_test-upstream`.

.. _module-pw_unit_test-main:

Create a custom ``main`` function
=================================
For simple unit tests that run on :ref:`target-host` the workflow outlined in
:ref:`module-pw_unit_test-quickstart` is all you need. Pigweed's build templates
generate a simple ``main`` function to run the tests with.

To do more complex testing, such as on-device testing:

1. Create your own ``main`` function:

   .. code-block:: c++

      #include "pw_unit_test/framework.h"
      // pw_unit_test:light requires an event handler to be configured.
      #include "pw_unit_test/simple_printing_event_handler.h"

      void WriteString(std::string_view string, bool newline) {
        printf("%s", string.data());
        if (newline) {
          printf("\n");
        }
      }

      int main() {
        // Make the binary compatible with pw_unit_test:googletest. Has no effect
        // when using pw_unit_test:light.
        testing::InitGoogleTest();
        // Set up the event handler for pw_unit_test:light.
        pw::unit_test::SimplePrintingEventHandler handler(WriteString);
        pw::unit_test::RegisterEventHandler(&handler);
        return RUN_ALL_TESTS();
      }

   See :ref:`module-pw_unit_test-event-handlers` for more information about
   handling events.

2. Set the build argument that instructs your build system to use your custom
   ``main`` function:

   * Bazel: :option:`@pigweed//pw_unit_test:main`
   * GN: :option:`pw_unit_test_MAIN`

.. _module-pw_unit_test-event-handlers:

Create event handlers
=====================
.. _//pw_unit_test/public/pw_unit_test/event_handler.h: https://cs.opensource.google/pigweed/pigweed/+/main:pw_unit_test/public/pw_unit_test/event_handler.h

The ``pw::unit_test::EventHandler`` class defines the interface through which
``pw_unit_test:light`` communicates the results of its test runs. If you're
using a :ref:`custom main function <module-pw_unit_test-main>` you need to
register an event handler to receive test output from the framework.

.. _module-pw_unit_test-predefined-event-handlers:

Predefined event handlers
-------------------------
Pigweed provides some standard event handlers to simplify the process of
getting started with ``pw_unit_test:light``. All event handlers provide for
GoogleTest-style output using the shared
:cpp:class:`pw::unit_test::GoogleTestStyleEventHandler` base. Example
output:

.. code-block::

   [==========] Running all tests.
   [ RUN      ] Status.Default
   [       OK ] Status.Default
   [ RUN      ] Status.ConstructWithStatusCode
   [       OK ] Status.ConstructWithStatusCode
   [ RUN      ] Status.AssignFromStatusCode
   [       OK ] Status.AssignFromStatusCode
   [ RUN      ] Status.CompareToStatusCode
   [       OK ] Status.CompareToStatusCode
   [ RUN      ] Status.Ok_OkIsTrue
   [       OK ] Status.Ok_OkIsTrue
   [ RUN      ] Status.NotOk_OkIsFalse
   [       OK ] Status.NotOk_OkIsFalse
   [ RUN      ] Status.KnownString
   [       OK ] Status.KnownString
   [ RUN      ] Status.UnknownString
   [       OK ] Status.UnknownString
   [==========] Done running all tests.
   [  PASSED  ] 8 test(s).

.. _module-pw_unit_test-subset:

Run a subset of test suites
===========================
.. _//pw_unit_test/light_public_overrides/pw_unit_test/framework_backend.h: https://cs.opensource.google/pigweed/pigweed/+/main:pw_unit_test/light_public_overrides/pw_unit_test/framework_backend.h

To run only a subset of registered test suites, use the
``pw::unit_test::SetTestSuitesToRun`` function. See
`//pw_unit_test/light_public_overrides/pw_unit_test/framework_backend.h`_.

This is useful when you've got a lot of test suites bundled up into a
:ref:`single test binary <module-pw_unit_test-main>` and you only need
to run some of them.

.. _module-pw_unit_test-skip:

Skip tests in Bazel
===================
Use ``target_compatible_with`` in Bazel to skip tests. The following test is
skipped when :ref:`using upstream GoogleTest <module-pw_unit_test-upstream>`:

.. code-block:: python

   load("//pw_unit_test:pw_cc_test.bzl", "pw_cc_test")

   pw_cc_test(
       name = "no_upstream_test",
       srcs = ["no_upstream_test.cc"],
       target_compatible_with = select({
           "//pw_unit_test:backend_is_googletest": ["@platforms//:incompatible"],
           "//conditions:default": [],
       }),
   )

.. _module-pw_unit_test-static:

Run tests in static libraries
=============================
To run tests in a static library, use the
:c:macro:`PW_UNIT_TEST_LINK_FILE_CONTAINING_TEST` macro.

Linkers usually ignore tests through static libraries (i.e. ``.a`` files)
because test registration relies on the test instance's static constructor
adding itself to a global list of tests. When linking against a static library,
static constructors in an object file are ignored unless at least one entity
in that object file is linked.

.. _module-pw_unit_test-upstream:

Use upstream GoogleTest
=======================
To use the upstream GoogleTest backend (``pw_unit_test:googletest``) instead
of the default backend:

.. _GoogleTestHandlerAdapter: https://cs.opensource.google/pigweed/pigweed/+/main:pw_unit_test/public/pw_unit_test/googletest_handler_adapter.h

1. Clone the GoogleTest repository into your project. See
   :ref:`module-pw_third_party_googletest`.

2. :ref:`Create a custom main function <module-pw_unit_test-main>`.

3. Combine `GoogleTestHandlerAdapter`_ with a :ref:`predefined event
   handler <module-pw_unit_test-predefined-event-handlers>` to enable your
   ``main`` function to work with upstream GoogleTest without modification.

   .. code-block:: c++

      #include "pw_unit_test/framework.h"
      #include "pw_unit_test/logging_event_handler.h"

      int main() {
        testing::InitGoogleTest();
        pw::unit_test::LoggingEventHandler logger;
        pw::unit_test::RegisterEventHandler(&logger);
        return RUN_ALL_TESTS();
      }

4. If your tests needs GoogleTest functionality not included in the default
   ``pw_unit_test:light`` backend, include the upstream GoogleTest headers
   (e.g. ``gtest/gtest.h``) directly and guard your target definition to avoid
   compiling with ``pw_unit_test:light`` (the default backend).

.. _module-pw_unit_test-serial-runner:

Run tests over serial
=====================
To accelerate automated unit test bringup for devices with plain-text logging,
``pw_unit_test`` provides a serial-based test runner in Python that triggers a
device flash and evaluates whether the test passed or failed based on the
produced output.

To set up a serial test runner in Python:

.. _//pw_unit_test/py/pw_unit_test/serial_test_runner.py: https://cs.opensource.google/pigweed/pigweed/+/main:pw_unit_test/py/pw_unit_test/serial_test_runner.py

1. Implement a ``SerialTestingDevice`` class for your device. See
   `//pw_unit_test/py/pw_unit_test/serial_test_runner.py`_.
2. Configure your device code to wait to run unit tests until
   ``DEFAULT_TEST_START_CHARACTER`` is sent over the serial connection.

.. _module-pw_unit_test-rpc:

Run tests over RPC
==================
.. _//pw_unit_test/pw_unit_test_proto/unit_test.proto: https://cs.opensource.google/pigweed/pigweed/+/main:pw_unit_test/pw_unit_test_proto/unit_test.proto

``pw_unit_test`` provides an RPC service which runs unit tests on demand and
streams the results back to the client. The service is defined in
`//pw_unit_test/pw_unit_test_proto/unit_test.proto`_.

The RPC service is primarily intended for use with the default
``pw_unit_test:light`` backend. It has some support for the upstream GoogleTest
backend (``pw_unit_test:googletest``), however some features (such as test suite
filtering) are missing.

To set up RPC-based unit tests in your application:

1. Depend on the relevant target for your build system:

   * Bazel: ``@pigweed//pw_unit_test:rpc_service``
   * GN: ``$dir_pw_unit_test:rpc_service``

2. Initialize a ``pw::unit_test::UnitTestThread`` instance and define a
   ``pw::Thread`` using it as the entry point. (See example below.)

3. Register the thread's ``Service`` with your RPC server.

   .. code-block:: c++

      #include "pw_rpc/server.h"
      #include "pw_thread/thread.h"
      #include "pw_unit_test/unit_test_service.h"

      pw::rpc::Channel channels[] = {
        pw::rpc::Channel::Create<1>(&my_output),
      };
      pw::rpc::Server server(channels);

      constexpr pw::ThreadAttrs kUnitTestThreadAttrs =
      pw::ThreadAttrs()
         .set_name("UnitTestThread")
         .set_priority(pw::ThreadPriority::Medium())
         .set_stack_size_bytes(4096);
      pw::ThreadContextFor<kUnitTestThreadAttrs> unit_test_thread_context;

      pw::unit_test::UnitTestThread unit_test_thread;

      void AppInit() {
        pw::Thread(
           pw::GetThreadOptions(unit_test_thread_context), unit_test_thread)
         .detach();
        server.RegisterService(unit_test_thread.service());
      }

   See :ref:`module-pw_rpc` for more guidance around setting up RPC.

6. Run tests that have been flashed to a device by calling
   ``pw_unit_test.rpc.run_tests()`` in Python. The argument should be an RPC
   client services object that has the unit testing RPC service enabled. By
   default, the results output via logging. The return value is a
   ``TestRecord`` dataclass instance containing the results of the test run.

   .. code-block:: python

      import serial

      from pw_hdlc import rpc
      from pw_unit_test.rpc import run_tests

      PROTO = Path(
          os.environ['PW_ROOT'],
          'pw_unit_test/pw_unit_test_proto/unit_test.proto'
      )
      serial_device = serial.Serial(device, baud)
      with rpc.SerialReader(serial_device) as reader:
          with rpc.HdlcRpcClient(
              reader, PROTO, rpc.default_channels(serial_device.write)
          ) as client:
              run_tests(client.rpcs())

.. _module-pw_unit_test-cpp:

-----------------
C++ API reference
-----------------

``pw_status`` Helpers
=====================
Both the light and GoogleTest backends of ``pw_unit_test`` expose some matchers
for dealing with Pigweed ``pw::Status`` and ``pw::Result`` values. See
:ref:`module-pw_unit_test-api-expect` and :ref:`module-pw_unit_test-api-assert`
for details.

.. _module-pw_unit_test-compatibility:

``pw_unit_test:light`` API compatibility
========================================
``pw_unit_test:light`` offers a number of primitives for test declaration,
assertion, event handlers, and configuration.

.. note::

   The ``googletest_test_matchers`` target which provides Pigweed-specific
   ``StatusIs``, ``IsOkAndHolds`` isn't part of the ``pw_unit_test:light``
   backend. These matchers are only usable when including the full upstream
   GoogleTest backend.

Missing features include:

* GoogleMock and matchers (e.g. :c:macro:`EXPECT_THAT`).
* Death tests (e.g. :c:macro:`EXPECT_DEATH`). ``EXPECT_DEATH_IF_SUPPORTED``
  does nothing but silently passes.
* Value-parameterized tests.
* Stream messages (e.g. ``EXPECT_TRUE(...) << "My message"``) will compile, but
  no message will be logged.

See :ref:`module-pw_unit_test-upstream` for guidance on using the
upstream GoogleTest backend (``pw_unit_test:googletest``) instead of
``pw_unit_test:light``.

.. _module-pw_unit_test-declare:

Test declaration
================
Note that ``TEST_F`` may allocate fixtures separately from the stack.
Large variables should be stored in test fixture fields,
rather than stack variables. This allows the test framework to statically ensure
that enough space is available to store these variables.

.. doxygendefine:: TEST
.. doxygendefine:: GTEST_TEST
.. doxygendefine:: TEST_F
.. doxygendefine:: FRIEND_TEST

.. _module-pw_unit_test-control:

Test control
============

.. doxygenfunction:: RUN_ALL_TESTS
.. doxygendefine:: FAIL
.. doxygendefine:: GTEST_FAIL
.. doxygendefine:: SUCCEED
.. doxygendefine:: GTEST_SUCCEED
.. doxygendefine:: GTEST_SKIP
.. doxygendefine:: ADD_FAILURE
.. doxygendefine:: GTEST_HAS_DEATH_TEST
.. doxygendefine:: EXPECT_DEATH_IF_SUPPORTED
.. doxygendefine:: ASSERT_DEATH_IF_SUPPORTED

.. _module-pw_unit_test-api-expect:

Expectations
============
When a test fails an expectation, the framework marks the test as a failure
and then continues executing the test. They're useful when you want to
verify multiple dimensions of the same feature and see all the errors at the
same time.

.. doxygendefine:: EXPECT_TRUE
.. doxygendefine:: EXPECT_FALSE
.. doxygendefine:: EXPECT_EQ
.. doxygendefine:: EXPECT_NE
.. doxygendefine:: EXPECT_GT
.. doxygendefine:: EXPECT_GE
.. doxygendefine:: EXPECT_LT
.. doxygendefine:: EXPECT_LE
.. doxygendefine:: EXPECT_NEAR
.. doxygendefine:: EXPECT_FLOAT_EQ
.. doxygendefine:: EXPECT_DOUBLE_EQ
.. doxygendefine:: EXPECT_STREQ
.. doxygendefine:: EXPECT_STRNE
.. doxygendefine:: PW_TEST_EXPECT_OK

.. _module-pw_unit_test-api-assert:

Assertions
==========
Assertions work the same as expectations except they stop the execution of the
test as soon as a failed condition is met.

.. doxygendefine:: ASSERT_TRUE
.. doxygendefine:: ASSERT_FALSE
.. doxygendefine:: ASSERT_EQ
.. doxygendefine:: ASSERT_NE
.. doxygendefine:: ASSERT_GT
.. doxygendefine:: ASSERT_GE
.. doxygendefine:: ASSERT_LT
.. doxygendefine:: ASSERT_LE
.. doxygendefine:: ASSERT_NEAR
.. doxygendefine:: ASSERT_FLOAT_EQ
.. doxygendefine:: ASSERT_DOUBLE_EQ
.. doxygendefine:: ASSERT_STREQ
.. doxygendefine:: ASSERT_STRNE
.. doxygendefine:: PW_TEST_ASSERT_OK
.. doxygendefine:: PW_TEST_ASSERT_OK_AND_ASSIGN

.. _module-pw_unit_test-api-event-handlers:

Event handlers
==============
.. doxygenfunction:: pw::unit_test::RegisterEventHandler(EventHandler* event_handler)
.. doxygenclass:: pw::unit_test::EventHandler
   :members:
.. doxygenclass:: pw::unit_test::GoogleTestHandlerAdapter
.. doxygenclass:: pw::unit_test::GoogleTestStyleEventHandler
.. doxygenclass:: pw::unit_test::SimplePrintingEventHandler
.. doxygenclass:: pw::unit_test::LoggingEventHandler
.. doxygenclass:: pw::unit_test::PrintfEventHandler
.. doxygenclass:: pw::unit_test::MultiEventHandler
.. doxygenclass:: pw::unit_test::TestRecordEventHandler

.. _module-pw_unit_test-cpp-config:

Configuration
=============
.. doxygenfile:: pw_unit_test/config.h
   :sections: define

.. _module-pw_unit_test-cpp-helpers:

Helpers
=======
.. doxygendefine:: PW_UNIT_TEST_LINK_FILE_CONTAINING_TEST

.. _module-pw_unit_test-py:

--------------------
Constexpr unit tests
--------------------
.. doxygenfile:: pw_unit_test/constexpr.h
   :sections: detaileddescription

API reference
=============
.. doxygendefine:: PW_CONSTEXPR_TEST

.. block-submission: disable
.. c:macro:: SKIP_CONSTEXPR_TESTS_DONT_SUBMIT

   Define the ``SKIP_CONSTEXPR_TESTS_DONT_SUBMIT`` macro to temporarily disable
   the ``constexpr`` portion of subsequent :c:macro:`PW_CONSTEXPR_TEST`\s. Use
   this to view GoogleTest output, which is usually more informative than the
   compiler's ``constexpr`` test failure output.

   Defines of this macro should never be submitted. If a test shouldn't run at
   compile time, use a plain ``TEST()``.

   .. literalinclude:: constexpr_test.cc
      :language: cpp
      :start-after: [pw_unit_test-constexpr-skip]
      :end-before: [pw_unit_test-constexpr-skip]
.. block-submission: enable

--------------------
Python API reference
--------------------
.. _module-pw_unit_test-py-serial_test_runner:

``pw_unit_test.serial_test_runner``
===================================
.. automodule:: pw_unit_test.serial_test_runner
   :members:
     DEFAULT_TEST_START_CHARACTER,
     SerialTestingDevice,
     run_device_test,

.. _module-pw_unit_test-py-rpc:

``pw_unit_test.rpc``
====================
.. automodule:: pw_unit_test.rpc
   :members: EventHandler, run_tests, TestRecord

.. _module-pw_unit_test-helpers:

----------------------
Build helper libraries
----------------------
The following helper libraries can simplify setup and are supported in all
build systems.

.. object:: simple_printing_event_handler

   When running tests, output test results as plain text over ``pw_sys_io``.

.. object:: simple_printing_main

   Implements a ``main()`` function that simply runs tests using the
   ``simple_printing_event_handler``.

.. object:: logging_event_handler

   When running tests, log test results as plain text using
   :ref:`module-pw_log`. Make sure your target has set a ``pw_log`` backend.

.. object:: logging_main

   Implements a ``main()`` function that simply runs tests using the
   ``logging_event_handler``.

.. _module-pw_unit_test-bazel:

-------------------
Bazel API reference
-------------------

See also :ref:`module-pw_unit_test-helpers`.

.. _module-pw_unit_test-pw_cc_test:

``pw_cc_test``
==============
.. _cc_test: https://bazel.build/reference/be/c-cpp#cc_test

``pw_cc_test`` is a wrapper for `cc_test`_ that provides some defaults, such as
a dependency on ``@pigweed//pw_unit_test:main``. It supports and passes through
all the arguments recognized by ``cc_test``.

.. _module-pw_unit_test-bazel-args:

Label flags
===========
.. option:: @pigweed//pw_unit_test:backend

   The GoogleTest implementation to use for Pigweed unit tests. This library
   provides ``gtest/gtest.h`` and related headers. Defaults to
   ``@pigweed//pw_unit_test:light``, which implements a subset of GoogleTest.

   Type: Bazel target label

   Usage: Typically specified as part of the platform definition, but can also
   be set manually on the command line.

.. option:: @pigweed//pw_unit_test:main

   Implementation of a main function for ``pw_cc_test`` unit test binaries. To
   use upstream GoogleTest, set this to ``@com_google_googletest//:gtest_main``.
   Note that this may not work on most embedded devices, see :bug:`310957361`.

   Type: Bazel target label

   Usage: Typically specified as part of the platform definition, but can also
   be set manually on the command line.

.. _module-pw_unit_test-gn:

------------
GN reference
------------
See also :ref:`module-pw_unit_test-helpers`.

.. _module-pw_unit_test-pw_test:

``pw_test``
===========
``pw_test`` defines a single unit test suite.

Targets
-------

.. object:: <target_name>

   The test suite within a single binary. The test code is linked against
   the target set in the build arg ``pw_unit_test_MAIN``.

.. object:: <target_name>.run

   If ``pw_unit_test_AUTOMATIC_RUNNER`` is set, this target runs the test as
   part of the build.

.. object:: <target_name>.lib

   The test sources without ``pw_unit_test_MAIN``.

Arguments
---------
All GN executable arguments are accepted and forwarded to the underlying
``pw_executable``.

.. option:: enable_if

   Boolean indicating whether the test should be built. If false, replaces the
   test with an empty target. Default true.

.. _get_target_outputs: https://gn.googlesource.com/gn/+/main/docs/reference.md#func_get_target_outputs

.. option:: source_gen_deps

   List of target labels that generate source files used by this test. The
   labels must meet the constraints of GN's `get_target_outputs`_, namely they must have been previously defined in the
   current file. This argument is required if a test uses generated source files
   and ``enable_if`` can evaluate to false.

.. option:: test_main

   Target label to add to the tests's dependencies to provide the ``main()``
   function. Defaults to ``pw_unit_test_MAIN``. Set to ``""`` if ``main()``
   is implemented in the test's ``sources``.

.. option:: test_automatic_runner_args

   Array of args to pass to :ref:`automatic test
   runner <module-pw_unit_test-serial-runner>`. Defaults to
   ``pw_unit_test_AUTOMATIC_RUNNER_ARGS``.

.. option:: envvars

   Array of ``key=value`` strings representing environment variables to set
   when invoking the automatic test runner.

Example
-------

.. code-block::

   import("$dir_pw_unit_test/test.gni")

   pw_test("large_test") {
     sources = [ "large_test.cc" ]
     enable_if = device_has_1m_flash
   }

.. _module-pw_unit_test-pw_test_group:

``pw_test_group``
=================
``pw_test_group`` defines a collection of tests or other test groups.

Targets
-------
.. object:: <target_name>

   The test group itself.

.. object:: <target_name>.run

   If ``pw_unit_test_AUTOMATIC_RUNNER`` is set, this target runs all of the
   tests in the group and all of its group dependencies individually. See
   :ref:`module-pw_unit_test-serial-runner`.

.. object:: <target_name>.lib

   The sources of all of the tests in this group and their dependencies.

.. object:: <target_name>.bundle

   All of the tests in the group and its dependencies bundled into a single binary.

.. object:: <target_name>.bundle.run

   Automatic runner for the test bundle.

Arguments
---------
.. option:: tests

   List of the ``pw_test`` targets in the group.

.. option:: group_deps

   List of other ``pw_test_group`` targets on which this one depends.

.. option:: enable_if

   Boolean indicating whether the group target should be created. If false, an
   empty GN group is created instead. Default true.

Example
-------
.. code-block::

   import("$dir_pw_unit_test/test.gni")

   pw_test_group("tests") {
     tests = [
       ":bar_test",
       ":foo_test",
     ]
   }

   pw_test("foo_test") {
     # ...
   }

   pw_test("bar_test") {
     # ...
   }

.. _module-pw_unit_test-pw_facade_test:

``pw_facade_test``
==================
Pigweed facade test templates allow individual unit tests to build under the
current device target configuration while overriding specific build arguments.
This allows these tests to replace a facade's backend for the purpose of testing
the facade layer.

Facade tests are disabled by default. To build and run facade tests, set the GN
arg :option:`pw_unit_test_FACADE_TESTS_ENABLED` to ``true``.

.. warning::
   Facade tests are costly because each facade test will trigger a re-build of
   every dependency of the test. While this sounds excessive, it's the only
   technically correct way to handle this type of test.

.. warning::
   Some facade test configurations may not be compatible with your target. Be
   careful when running a facade test on a system that heavily depends on the
   facade being tested.

.. _module-pw_unit_test-gn-args:

GN build arguments
==================
.. option:: pw_unit_test_BACKEND <source_set>

   The GoogleTest implementation to use for Pigweed unit tests. This library
   provides ``gtest/gtest.h`` and related headers. Defaults to
   ``pw_unit_test:light``, which implements a subset of GoogleTest.

   Type: string (GN path to a source set)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_MAIN <source_set>

   Implementation of a main function for ``pw_test`` unit test binaries.
   See :ref:`module-pw_unit_test-main`.

   Type: string (GN path to a source set)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_AUTOMATIC_RUNNER <executable>

   Path to a test runner to automatically run unit tests after they are built.
   See :ref:`module-pw_unit_test-serial-runner`.

   If set, a ``pw_test`` target's ``<target_name>.run`` action invokes the
   test runner specified by this argument, passing the path to the unit test to
   run. If this is unset, the ``pw_test`` target's ``<target_name>.run`` step
   will do nothing.

   Targets that don't support parallelized execution of tests (e.g. an
   on-device test runner that must flash a device and run the test in serial)
   should set ``pw_unit_test_POOL_DEPTH`` to ``1``.

   Type: string (name of an executable on ``PATH``, or a path to an executable)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_AUTOMATIC_RUNNER_ARGS <args>

   An optional list of strings to pass as args to the test runner specified by
   ``pw_unit_test_AUTOMATIC_RUNNER``.

   Type: list of strings (args to pass to ``pw_unit_test_AUTOMATIC_RUNNER``)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_AUTOMATIC_RUNNER_TIMEOUT <timeout_seconds>

   An optional timeout to apply when running the automatic runner. Timeout is
   in seconds. Defaults to empty which means no timeout.

   Type: string (number of seconds to wait before killing test runner)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_POOL_DEPTH <pool_depth>

   The maximum number of unit tests that may be run concurrently for the
   current toolchain. Setting this to 0 disables usage of a pool, allowing
   unlimited parallelization.

   Note: A single target with two toolchain configurations (e.g. ``release``
   and ``debug``) uses two separate test runner pools by default. Set
   ``pw_unit_test_POOL_TOOLCHAIN`` to the same toolchain for both targets to
   merge the pools and force serialization.

   Type: integer

   Usage: toolchain-controlled only

.. option:: pw_unit_test_POOL_TOOLCHAIN <toolchain>

   The toolchain to use when referring to the ``pw_unit_test`` runner pool.
   When this is disabled, the current toolchain is used. This means that every
   toolchain will use its own pool definition. If two toolchains should share
   the same pool, this argument should be by one of the toolchains to the GN
   path of the other toolchain.

   Type: string (GN path to a toolchain)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_EXECUTABLE_TARGET_TYPE <template name>

   The name of the GN target type used to build ``pw_unit_test`` executables.

   Type: string (name of a GN template)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_EXECUTABLE_TARGET_TYPE_FILE <gni file path>

   The path to the ``.gni`` file that defines
   ``pw_unit_test_EXECUTABLE_TARGET_TYPE``.

   If ``pw_unit_test_EXECUTABLE_TARGET_TYPE`` is not the default of
   ``pw_executable``, this ``.gni`` file is imported to provide the template
   definition.

   Type: string (path to a .gni file)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_FACADE_TESTS_ENABLED <boolean>

   Controls whether to build and run facade tests. Facade tests add considerably
   to build time, so they are disabled by default.

.. option:: pw_unit_test_TESTONLY <boolean>

   Controls the ``testonly`` variable in ``pw_test``, ``pw_test_group``, and
   miscellaneous testing targets. This is useful if your test libraries (e.g.
   GoogleTest) used by pw_unit_test have the ``testonly`` flag set. False by
   default for backwards compatibility.

.. _module-pw_unit_test-cmake:

---------------
CMake reference
---------------
See also :ref:`module-pw_unit_test-helpers`.

.. _module-pw_unit_test-pw_add_test:

``pw_add_test``
===============
``pw_add_test`` declares a single unit test suite.

.. tip::
   Upstream Pigweed tests can be disabled in downstream projects by setting
   ``pw_unit_test_ENABLE_PW_ADD_TEST`` to ``OFF`` before adding the ``pigweed``
   directory to an existing cmake build.

   .. code-block:: cmake

      set(pw_unit_test_ENABLE_PW_ADD_TEST OFF)
      add_subdirectory(path/to/pigweed pigweed)
      set(pw_unit_test_ENABLE_PW_ADD_TEST ON)

   See also: :ref:`module-pw_build-existing-cmake-project`.

Targets
-------
.. object:: {NAME}

   Depends on ``${NAME}.run`` if ``pw_unit_test_AUTOMATIC_RUNNER`` is set, else
   it depends on ``${NAME}.bin``.

.. object:: {NAME}.lib

   Contains the provided test sources as a library target, which can then be
   linked into a test executable.

.. object:: {NAME}.bin

   A standalone executable which contains only the test sources specified in the
   ``pw_unit_test`` template.

.. object:: {NAME}.run

   Runs the unit test executable after building it if
   ``pw_unit_test_AUTOMATIC_RUNNER`` is set, else it fails to build.

Required arguments
------------------
.. option:: NAME

   Name to use for the produced test targets specified above.

Optional arguments
------------------
.. option:: SOURCES

   Source files for this library.

.. option:: HEADERS

   Header files for this library.

.. option:: PRIVATE_DEPS

   Private ``pw_target_link_targets`` arguments.
.. option:: PRIVATE_INCLUDES

   Public ``target_include_directories`` argument.

.. option:: PRIVATE_DEFINES

   Private ``target_compile_definitions`` arguments.

.. option:: PRIVATE_COMPILE_OPTIONS

   Private ``target_compile_options`` arguments.

.. option:: PRIVATE_LINK_OPTIONS

   Private ``target_link_options`` arguments.

Example
-------

.. code-block::

   include($ENV{PW_ROOT}/pw_unit_test/test.cmake)

   pw_add_test(my_module.foo_test
     SOURCES
       foo_test.cc
     PRIVATE_DEPS
       my_module.foo
   )

.. _module-pw_unit_test-pw_add_test_group:

``pw_add_test_group``
=====================
``pw_add_test_group`` defines a collection of tests or other test groups.

Targets
-------
.. object:: {NAME}

   Depends on ``${NAME}.run`` if ``pw_unit_test_AUTOMATIC_RUNNER`` is set, else
   it depends on ``${NAME}.bin``.

.. object:: {NAME}.bundle

   Depends on ``${NAME}.bundle.run`` if ``pw_unit_test_AUTOMATIC_RUNNER`` is
   set, else it depends on ``${NAME}.bundle.bin``.

.. object:: {NAME}.lib

   Depends on ``${NAME}.bundle.lib``.

.. object:: {NAME}.bin

   Depends on the provided ``TESTS``'s ``<test_dep>.bin`` targets.

.. object:: {NAME}.run

   Depends on the provided ``TESTS``'s ``<test_dep>.run`` targets if
   ``pw_unit_test_AUTOMATIC_RUNNER`` is set, else it fails to build.

.. object:: {NAME}.bundle.lib

   Contains the provided tests bundled as a library target, which can then be
   linked into a test executable.

.. object:: {NAME}.bundle.bin

   Standalone executable which contains the bundled tests.

.. object:: {NAME}.bundle.run

   Runs the ``{NAME}.bundle.bin`` test bundle executable after building it if
   ``pw_unit_test_AUTOMATIC_RUNNER`` is set, else it fails to build.

Required arguments
------------------
.. option:: NAME

   The name of the executable target to be created.

.. option:: TESTS

   ``pw_add_test`` targets and ``pw_add_test_group`` bundles to be
   included in this test bundle.

Example
-------
.. code-block::

   include($ENV{PW_ROOT}/pw_unit_test/test.cmake)

   pw_add_test_group(tests
     TESTS
       bar_test
       foo_test
   )

   pw_add_test(foo_test
     # ...
   )

   pw_add_test(bar_test
     # ...
   )

.. _module-pw_unit_test-cmake-args:

CMake build arguments
=====================
.. option:: pw_unit_test_BACKEND <target>

   The GoogleTest implementation to use for Pigweed unit tests. This library
   provides ``gtest/gtest.h`` and related headers. Defaults to
   ``pw_unit_test.light``, which implements a subset of GoogleTest.

   Type: string (CMake target name)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_AUTOMATIC_RUNNER <executable>

   Path to a test runner to automatically run unit tests after they're built.

   If set, a ``pw_test`` target's ``${NAME}`` and ``${NAME}.run`` targets will
   invoke the test runner specified by this argument, passing the path to the
   unit test to run. If this is unset, the ``pw_test`` target's ``${NAME}`` will
   only build the unit test(s) and ``${NAME}.run`` will fail to build.

   Type: string (name of an executable on the PATH, or path to an executable)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_AUTOMATIC_RUNNER_ARGS <args>

   An optional list of strings to pass as args to the test runner specified
   by ``pw_unit_test_AUTOMATIC_RUNNER``.

   Type: list of strings (args to pass to pw_unit_test_AUTOMATIC_RUNNER)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_AUTOMATIC_RUNNER_TIMEOUT_SECONDS <timeout_seconds>

   An optional timeout to apply when running the automatic runner. Timeout is
   in seconds. Defaults to empty which means no timeout.

   Type: string (number of seconds to wait before killing test runner)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_ADD_EXECUTABLE_FUNCTION <function name>

   The name of the CMake function used to build ``pw_unit_test`` executables.
   The provided function must take a ``NAME`` and a ``TEST_LIB`` argument which
   are the expected name of the executable target and the target which provides
   the unit test(s).

   Type: string (name of a CMake function)

   Usage: toolchain-controlled only

.. option:: pw_unit_test_ADD_EXECUTABLE_FUNCTION_FILE <cmake file path>

   The path to the ``.cmake`` file that defines
   ``pw_unit_test_ADD_EXECUTABLE_FUNCTION``.

   Type: string (path to a ``.cmake`` file)

   Usage: toolchain-controlled only

====================
Zephyr backend tests
====================
Zephyr backend tests use Zephyr's test runner `twister`_. The test runner is
installed into the Pigweed CLI and can be run by calling ``pw twister-runner``.
The runner intercepts a few arguments in order to add features to the normal
test runner; beyond that, the runner just calls twister directly.

The runner will intercept the ``-T`` or ``--testsuite-root`` arguments and do
the following:

* For every ``testcase.yaml`` file under the testsuite root directory it appends
  ``-T path/to/testcase.yaml``.
* For every ``testtemplate.yaml`` it creates a new directory using the
  ``{name}`` field in the YAML file and the ``--build-dir`` to create a new test
  root at ``{build_dir}/{name}``. It will then take the ``testcase`` node in the
  template and create a new ``{build_dir}/{name}/testcase.yaml``. This new YAML
  file will be appended to the final twister command using ``-T
  {build_dir}/{name}``.

--------------
Test templates
--------------
Twister normally uses a combination of ``testcase.yaml`` files along
``CMakeLists.txt`` files which serve as the test's entry point. This doesn't
work when trying to write tests side by side with the source. The twister runner
introduces the concept of ``testtemplate.yaml`` which allows the test to live in
the same directory as the source but is compatible with twister. The test
template YAML file contains the following required keys:

* ``name`` - The name of the test directory to create. This must be unique per
  twister run.
* ``testcase`` - The content of this entry will be copied into a new
  ``{build_dir}/{name}/testcase.yaml`` file.

Additional fields are available in the template and are optional:

* ``file-map`` - This key/value dictionary contains a mapping of relative file
  paths (reltive to the ``testtemplate.yaml`` file) to output relative file
  paths (relative to the generated ``{build_dir}/{name)``). Each entry will
  create a symlink.
* ``use-default-cmake`` - By default ``true`` and allows the test to leverage a
  common ``CMakeLists.txt`` that should fit the majority of test cases. It sets
  up a simple Zephyr app which uses the ``simple_printing_main.cc`` to run the
  tests. This ``CMake`` file also expects that the test will be a library
  created by ``pw_add_test``. For example, if we wanted to test the
  ``pw_base64.base64_test`` unit test we would just need to pass
  ``TEST_LIB=pw_base64.base64_test`` as a ``CMake`` argument.
* ``use-default-prj-conf`` - By default ``true`` and allows the test to leverage
  a common ``prj.conf`` that should fit the majority of test cases. It sets up
  C++20 and a few other common Kconfigs. Additional configs can be added as a
  part of the ``testtemplate.yaml`` using the ``extra_configs`` key.

-----------------
Running the tests
-----------------
Automatically, tests will be picked up by ``pw presubmit --step zephyr_build``.
This presubmit test will call the ``twister-runner`` script to run all the tests
defined in either ``testcase.yaml`` or ``testtemplate.yaml``. Coverage reports
can be found later in ``twister-out/coverage/``.

.. _twister: https://docs.zephyrproject.org/latest/develop/test/twister.html
