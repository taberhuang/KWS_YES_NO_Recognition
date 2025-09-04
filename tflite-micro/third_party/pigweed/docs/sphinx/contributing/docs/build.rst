.. _contrib-docs-build:

=========================================
Working with the pigweed.dev build system
=========================================
``pigweed.dev`` is built with Bazel. When you want to add or remove files
used by ``pigweed.dev``, you'll need to interact with this Bazel-based
documentation generation (docgen) system.

Check out :ref:`contrib-docs-build-appendix-architecture` for a top-down explanation
of the main components of the docgen system.

.. _contrib-docs-build-quickstart:

----------
Quickstart
----------
#. Build the docs:

   .. code-block:: console

      $ bazelisk build //docs

#. Locally preview the docs:

   .. code-block:: console

      $ bazelisk run //docs:serve

.. _contrib-docs-build-setup:

-----
Setup
-----
Before you can do anything with the Bazel-based docgen system, you must
complete this setup:

#. Complete :ref:`docs-first-time-setup`.

#. :ref:`docs-install-bazel`.

.. _contrib-docs-build-files:

---------------------------
Add files to the docs build
---------------------------

.. _contrib-docs-build-files-doxygen:

Add files to the C/C++ API reference auto-generation system (Doxygen)
=====================================================================
#. Package your headers into a ``filegroup``:

   .. code-block:: py

      filegroup(
          name = "doxygen",
          srcs = [
              "public/pw_string/format.h",
              "public/pw_string/string.h",
              "public/pw_string/string_builder.h",
              "public/pw_string/utf_codecs.h",
              "public/pw_string/util.h",
          ],
      )

#. Update ``srcs`` in ``//docs/doxygen/BUILD.bazel`` to take a
   dependency on your new ``filegroup``:

   .. code-block:: py

      filegroup(
          name = "srcs",
          srcs = [
              # …
              "//pw_string:doxygen",
              # …
          ]
      )

.. _Breathe directive: https://breathe.readthedocs.io/en/latest/directives.html

#. Use a `Breathe directive`_ such as ``.. doxygenclass::`` to pull the API
   reference content into a reStructuredText file.

.. _contrib-docs-build-files-autodoc:

Add files to the Python API reference auto-generation system (autodoc)
======================================================================
If you see an error like this:

.. code-block:: text

   sphinx.errors.SphinxWarning: autodoc: failed to import module 'benchmark'
   from module 'pw_rpc'; the following exception was raised:
   No module named 'pw_rpc.benchmark'

.. inclusive-language: disable
.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. inclusive-language: enable

It means that `autodoc`_ (the program we use to auto-generate Python API
references) could not find the source code for the module that it was
supposed to document. To fix this:

#. Add your Python target as a dependency to the ``sphinx_build_binary``
   rule in ``//docs/sphinx/BUILD.bazel``:

   .. code-block:: py

      sphinx_build_binary(
          name = "sphinx_build",
          target_compatible_with = incompatible_with_mcu(),
          deps = [
              # …
              "//pw_rpc/py:pw_rpc_benchmark",
              # …
          ],
      )

.. _contrib-docs-build-files-sphinx:

Add reStructuredText files to Sphinx
====================================
#. Package your inputs into a ``sphinx_docs_library``:

   .. code-block:: py

      load("@rules_python//sphinxdocs:sphinx_docs_library.bzl", "sphinx_docs_library")
      load("//pw_build:compatibility.bzl", "incompatible_with_mcu")

      sphinx_docs_library(
          name = "docs",
          srcs = [
              "docs.rst",
          ],
          prefix = "pw_elf/",
          target_compatible_with = incompatible_with_mcu(),
          visibility = ["//visibility:public"],
      )

#. Update ``docs`` in ``//docs/sphinx/BUILD.bazel`` to take a dependency on
   your new ``sphinx_docs_library``:

   .. code-block:: py

      sphinx_docs(
          name = "docs",
          # …
          deps = [
              # …
              "//pw_elf:docs",
              # …
          ]
      )

.. _toctree: https://documatt.com/restructuredtext-reference/element/toctree.html

#. Add your new reStructuredText files to an existing `toctree`_, or create a new one.

.. _contrib-docs-build-files-source:

Add source code to the docs build
=================================
Whenever possible, don't manually write code examples in your reStructuredText
(reST) docs. These code examples will bitrot over time. Instead, put your code
examples in real source code that can actually be built and tested, and then
use Sphinx's ``literalinclude`` feature to insert the code example into your
doc.

#. Put your code example into a unit test:

   .. code-block:: c++

      // examples.cc

      TEST(StringExamples, BufferExample) {
        // START: BufferExample
        // …
        // END: BufferExample
      }

#. Include the code example in your reST:

   .. code-block:: rest

      .. literalinclude:: ./examples.cc
         :language: cpp
         :dedent:
         :start-after: // START: BufferExample
         :end-before: // END: BufferExample

#. Add the source code file to the ``srcs`` list in your
   ``sphinx_docs_library`` target:

   .. code-block:: py

      sphinx_docs_library(
          name = "docs",
          srcs = [
              # …
              "examples.cc",
              # …
          ],
      )

.. _contrib-docs-build-files-images:

Add images
==========
Images should not be checked into the Pigweed repo.
See :ref:`contrib-docs-website-images`.

.. _contrib-docs-build-files-remove:

----------------------------------------
Remove or change files in the docs build
----------------------------------------
Here's the general workflow:

#. Remove or change files that are used in the docs build.

#. :ref:`contrib-docs-build-build`.

#. When the docs build fails, Bazel's logs will tell you what you need to do
   next. If Bazel's logs aren't informative, try some of the tips described
   in :ref:`contrib-docs-build-debug`.

You may need to do some or all of these steps:

* In your module's ``BUILD.bazel`` files, update these rules:

  * ``sphinx_docs_library`` targets (usually named ``docs``)

  * ``filegroup`` targets named ``doxygen``

* Update ``//docs/sphinx/BUILD.bazel``.

* :ref:`redirects <contrib-docs-website-redirects>`.

.. _contrib-docs-build-build:

--------------
Build the docs
--------------
.. code-block:: console

   $ bazelisk build //docs

.. _contrib-docs-build-build-watch:

Watch the docs (automatically rebuild when files change)
========================================================
.. code-block:: console

   $ bazelisk run //:watch build //docs

.. tip::

   Try :ref:`locally previewing the docs <contrib-docs-build-preview>` in one console
   tab and watching the docs in another tab.

.. _contrib-docs-build-preview:

------------------------
Locally preview the docs
------------------------
.. code-block:: console

   $ bazelisk run //docs:serve

A message like this should get printed to ``stdout``:

.. code-block:: text

   Serving...
     Address: http://0.0.0.0:8000
     Serving directory: /home/kayce/pigweed/pigweed/bazel-out/k8-fastbuild/bin/docs/docs/_build/html
         url: file:///home/kayce/pigweed/pigweed/bazel-out/k8-fastbuild/bin/docs/docs/_build/html
     Server CWD: /home/kayce/.cache/bazel/_bazel_kayce/9659373b1552c281136de1c8eeb3080d/execroot/_main/bazel-out/k8-fastbuild/bin/docs/docs.serve.runfiles/_main

You can access the rendered docs at the URL that's printed next to
**Address** (``http://0.0.0.0:8000`` in the example).

.. _contrib-docs-build-list:

---------------------
List all docs sources
---------------------
.. _hermetic: https://bazel.build/basics/hermeticity

Bazel builds the docs in a `hermetic`_ environment. All inputs to the docgen
system must be copied into this hermetic environment. To check that you're
copying your files to the correct directory, run this command:

.. code-block:: console

   $ bazelisk build //docs/sphinx:_docs/_sources

.. _contrib-docs-build-debug:

--------------------
Debug the docs build
--------------------
.. inclusive-language: disable
.. _sphinx-build: https://www.sphinx-doc.org/en/master/man/sphinx-build.html
.. _--verbose: https://www.sphinx-doc.org/en/master/man/sphinx-build.html#cmdoption-sphinx-build-v
.. inclusive-language: enable

When things go wrong, run this command to build the docs in a
non-`hermetic`_ environment:

.. code-block:: console

   $ bazelisk run //docs/sphinx:docs.run

Also consider tweaking these ``extra_opts`` from the ``sphinx_docs`` rule in
``//docs/sphinx/BUILD.bazel``:

* Comment out the ``--silent`` warning to get more verbose logging output.
* Check `sphinx-build`_ to see what other options you might want to add or remove.
  ``sphinx-build`` is the underlying command that the ``sphinx_docs`` Bazel rule runs.

.. _contrib-docs-build-troubleshoot:

---------------
Troubleshooting
---------------

.. _contrib-docs-build-troubleshoot-autodoc:

autodoc: failed to import module
================================
See :ref:`contrib-docs-build-files-autodoc`.

.. _contrib-docs-build-appendix-architecture:

-------------------------------
Appendix: Architecture overview
-------------------------------
The outputs of some components of the docgen system are used as inputs
to other components.

.. mermaid::

   flowchart LR

     Doxygen --> Breathe
     Breathe --> reST
     reST --> Sphinx
     Rust --> Sphinx
     Python --> Sphinx

.. _Doxygen: https://www.doxygen.nl/
.. _Breathe: https://breathe.readthedocs.io/en/latest/
.. _reStructuredText: https://docutils.sourceforge.io/rst.html
.. _rustdoc: https://doc.rust-lang.org/rustdoc/what-is-rustdoc.html
.. inclusive-language: disable
.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. inclusive-language: enable
.. _static site generator: https://en.wikipedia.org/wiki/Static_site_generator

* **Doxygen**: We feed a bunch of C/C++ headers to `Doxygen`_. Doxygen parses each
  header and generates XML metadata for all of the classes, functions, structs,
  etc. that it finds. We also publish the Doxygen-generated HTML as a separate
  subsite. This subsite is available at
  `pigweed.dev/doxygen <https://pigweed.dev/doxygen>`_.

* **Breathe**: We provide the Doxygen XML metadata to `Breathe`_ so that C/C++ API
  reference content can be inserted into our reStructuredText files.

* **reST**: We gather up all the `reStructuredText`_ (reST) source files
  that are scattered across the Pigweed repository. Pigweed docs are authored in
  reST. We don't use Markdown.

* **Rust**: `rustdoc`_ generates Rust API reference content, similar to how
  Doxygen generates C/C++ API reference content. The Rust API references are output
  as HTML. It's essentially a separate documentation subsite that is not integrated
  with the rest of ``pigweed.dev`` (yet). This subsite is available at URLs like
  `pigweed.dev/rustdoc/pw_bytes/ <https://pigweed.dev/rustdoc/pw_bytes/>`_.

* **Python**: We use Sphinx's `autodoc`_ feature to auto-generate Python API
  reference content. In order for this to work, the Python modules must be
  listed as dependencies of the ``//docs/sphinx:docs`` target.

* **Sphinx**: Once all the other inputs are ready, we can use `Sphinx`_
  (essentially a `static site generator`_) to build the ``pigweed.dev``
  website.
