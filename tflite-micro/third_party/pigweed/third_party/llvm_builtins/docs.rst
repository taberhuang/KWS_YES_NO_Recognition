.. _module-pw_third_party_llvm_builtins:

================
LLVM compiler-rt
================
The ``$pw_external_llvm_builtins/`` directory provides the GN integration
necessary to link against LLVM compiler-rt project. The intention here is to use
the builtins provided by the LLVM compiler-rt project.

-------------------------------
Using upstream LLVM compiler-rt
-------------------------------
If you want to use LLVM compiler-rt, you must do the following:

Submodule
=========
Add LLVM compiler-rt to your workspace with the following command.

.. code-block:: console

   $ git submodule add \
   > https://llvm.googlesource.com/llvm-project/compiler-rt \
   > third_party/llvm_builtins

.. admonition:: Note

   This git repository is maintained by Google and is a slice of upstream
   LLVM including only the compiler-rt subdirectory.

GN
==
* Set the GN var ``dir_pw_third_party_llvm_builtins`` to the location of the
  LLVM compiler-rt source. If you used the command above, this will be
  ``//third_party/llvm_builtins``.

  This can be set in your ``args.gn`` or ``.gn`` file:

  .. code-block:: text

     dir_pw_third_party_llvm_builtins = "//third_party/llvm_builtins"

* Set the ``pw_third_party_llvm_builtins_TARGET_BUILTINS`` to the ``pw_source_set``
  that selectively adds the files required for the given architecture from the
  LLVM compiler-rt checkout directory.

  For example, you can add the following in your ``args.gn`` or ``.gn`` file to use the
  builtins for ARMv7-M targets.

  .. code-block:: text

     pw_third_party_llvm_builtins_TARGET_BUILTINS = "$pw_external_llvm_builtins:arm_builtins_armv7m"

* Set the optional ``pw_third_party_llvm_builtins_ignore_list`` variable to the list of
  files included in ``pw_source_set`` in ``$pw_external_llvm_builtins/BUILD.gn``.
