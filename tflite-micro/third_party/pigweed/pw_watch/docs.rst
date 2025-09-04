.. _module-pw_watch:

.. rst-class:: with-subtitle

========
pw_watch
========
.. pigweed-module::
   :name: pw_watch

   * **Automatically trigger build actions when source files change**

----------
Background
----------
In the web development space, file system watchers like `nodemon
<https://www.npmjs.com/package/nodemon>`_ and `watchman
<https://facebook.github.io/watchman/>`_ are prevalent. These watchers trigger
actions when files change (such as reloading a web server), making development
much faster. In the embedded space, file system watchers are less prevalent but
no less useful!

.. _module-pw_watch-design:

------------
Our solution
------------
``pw_watch`` is similar to file system watchers found in web development
tooling but is focused around embedded development use cases. After changing
source code, ``pw_watch`` can instantly  compile, flash, and run tests.

.. figure:: https://storage.googleapis.com/pigweed-media/pw_watch_test_demo2.gif
   :width: 1420
   :alt: ``pw watch`` running in fullscreen mode and displaying errors

   ``pw watch`` running in fullscreen mode and displaying errors.

Combined with the GN-based build which expresses the full dependency tree,
only the exact tests affected by source changes are run.

The demo below shows ``pw_watch`` building for a STMicroelectronics
STM32F429I-DISC1 development board, flashing the board with the affected test,
and verifying the test runs as expected. Once this is set up, you can attach
multiple devices to run tests in a distributed manner to reduce the time it
takes to run tests.

.. figure:: https://storage.googleapis.com/pigweed-media/pw_watch_on_device_demo.gif
   :width: 800
   :alt: pw_watch running on-device tests

.. _module-pw_watch-get-started:

-----------
Get started
-----------

GN
==
.. code-block:: bash

   cd ~/pigweed
   source activate.sh
   pw watch

The simplest way to get started with ``pw_watch`` is to launch it from a shell
using the Pigweed environment as ``pw watch``. By default, ``pw_watch`` watches
for repository changes and triggers the default Ninja build target at ``//out``.
To override this behavior, provide the ``-C`` argument to ``pw watch``.

See :ref:`module-pw_watch-guide` for more examples and
:ref:`module-pw_watch-cli` for detailed CLI usage information.

Bazel
=====
The Bazel build provides a ``//:watch`` entrypoint to ``pw_watch``, which
executes commands with ``bazelisk``. Arguments are forwarded directly to
``bazel`` / ``bazelisk``, so any ``bazel`` subcommands may be used.

.. code-block:: sh

   # Runs bazelisk build //... when files change.
   bazelisk run //:watch build //...

   # Runs an executable when files change.
   bazelisk run //:watch -- run //important:script -a --value 52

   # Builds //..., then runs the //foo:bar test.
   bazelisk run //:watch build //... , test //foo:bar

.. important::

   ``bazelisk run`` interprets arguments that start with ``-``. Pass ``--``
   before any other arguments to separate arguments to Bazel from command
   arguments.

   .. code-block:: sh

      bazelisk run //:watch -- <commands to run>

``//:watch`` is an alias of ``//pw_watch/py:bazel``. External projects may run
this tool directly as ``@pigweed//:watch``, or create their own ``//:watch``
alias.

Custom commands
===============
.. automodule:: pw_watch.run

.. toctree::
   :hidden:
   :maxdepth: 1

   guide
   cli
