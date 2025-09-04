.. _target-rp2-upstream:

=====================================
Working with RP2s in upstream Pigweed
=====================================
RP2 is a family of microcontrollers (MCUs) from Raspberry Pi. The RP2040 MCU
(which powers the Pico 1) and the RP2350 MCU (which powers the Pico 2) are both
part of the RP2 family.

This guide shows :ref:`docs-glossary-upstream` maintainers how to do common
RP2-related development tasks in upstream Pigweed, such as building the upstream
repo for the RP2350, running on-device unit tests on a Pico 2, etc.

Most maintainers should use the newer Bazel-based workflows. If you need to use
the older GN-based workflows, see :ref:`target-rp2-upstream-gn`.

--------
Hardware
--------
.. _PicoPico: https://pigweed.googlesource.com/pigweed/hardware/picopico
.. _Debug Probe: https://www.raspberrypi.com/products/debug-probe/

Supported MCUs:

* RP2040
* RP2350

Supported Boards:

* Pico 1
* Pico 2

Unsupported boards:

* Pico 1W
* Pico 2W

The core Pigweed team uses `PicoPico`_, a custom development board that makes
parallel on-device testing easier.

If you don't have access to a PicoPico, the next best option is a Pico 2
and a `Debug Probe`_.

-------------------------------------------------
MCU selection flags (``--config`` and ``--chip``)
-------------------------------------------------
If your command requires a ``-config`` flag (e.g. ``--config=<mcu>``)
or a ``--chip`` flag (e.g. ``--chip <mcu>``) then you must replace
the ``<mcu>`` placeholder with one of these values:

* ``rp2040``
* ``rp2350``

For example, to build upstream Pigweed for a Pico 1 you run:

.. code-block:: console

   $ bazelisk build --config=rp2040 //...

Whereas to build upstream Pigweed for a Pico 2 you run:

.. code-block:: console

   $ bazelisk build --config=rp2350 //...

.. important::

   The target path should always be ``//targets/rp2040``, even if you're
   working with an RP2350. When we originally created this target, the
   RP2350 didn't exist yet, so we called the target ``rp2040`` and placed
   its code in the ``//targets/rp2040`` directory. The target now supports
   both RP2040 and RP2350. We just haven't got around to making the target
   name more general. If you try to reference ``//targets/rp2350`` you will
   see an error like this ``ERROR: no such package 'targets/rp2350'``
   because no directory exists at that location.

.. _target-rp2-upstream-build:

----------------
Build everything
----------------
.. code-block:: console

   $ bazelisk build --config=<mcu> //...

.. _target-rp2-upstream-tests:

-------------------
Run on-device tests
-------------------
.. _Updating the firmware on the Debug Probe: https://www.raspberrypi.com/documentation/microcontrollers/debug-probe.html#updating-the-firmware-on-the-debug-probe
.. _Getting Started: https://www.raspberrypi.com/documentation/microcontrollers/debug-probe.html#getting-started

#. Set up your hardware:

   .. tab-set::

      .. tab-item:: PicoPico
         :sync: picopico

         Connect the USB-Micro port of the **DEBUG PROBE** Pico
         to a USB port on your development host. Don't connect the
         **DEVICE UNDER TEST** Pico to anything.

      .. tab-item:: Debug Probe
         :sync: probe

         Make sure that your Debug Probe is running firmware version 2.0.1
         or later. See `Updating the firmware on the Debug Probe`_.

         Wire up your Pico to the Debug Probe as described in `Getting Started`_.

      .. tab-item:: Standalone
         :sync: standalone

         You can run Pigweed's tests with a single Pico and no additional
         hardware with the following limitations:

         * Tests will not be parallelized if more than one Pico is attached.

         * If the Pico crashes during a test, the failure will cascade into
           subsequent tests. You'll need to manually disconnect and re-connect
           the device to get it working again.

#. Start the test runner server:

   .. tab-set::

      .. tab-item:: PicoPico
         :sync: picopico

         .. code-block:: console

            $ bazelisk run //targets/rp2040/py:unit_test_server -- --chip <mcu>

      .. tab-item:: Debug Probe
         :sync: probe

         .. code-block:: console

            $ bazelisk run //targets/rp2040/py:unit_test_server -- --chip <mcu> --debug-probe-only

      .. tab-item:: Standalone
         :sync: standalone

         .. code-block:: console

            $ bazelisk run //targets/rp2040/py:unit_test_server -- --chip <mcu>


#. Open another terminal and run the tests:

   .. code-block:: console

      $ bazelisk test --config=<mcu> //...

.. _target-rp2-upstream-gn:

--------------------
GN (less maintained)
--------------------
The following guides may be outdated. We're keeping them around for Pigweed
contributors that are maintaining the upstream GN build system.

First-time setup
================

GN
==
To use this target, Pigweed must be set up to use FreeRTOS and the Pico SDK
HAL. When using Bazel, dependencies will be automatically installed.  For the GN
build, the supported repositories can be downloaded via ``pw package``, and then
the build must be manually configured to point to the locations the repositories
were downloaded to.

.. warning::

   The GN build does not distribute the libusb headers which are required by
   picotool.  If the picotool installation fails due to missing libusb headers,
   it can be fixed by installing them manually.

   .. tab-set::

      .. tab-item:: Linux
         :sync: linux

         .. code-block:: console

            $ sudo apt-get install libusb-1.0-0-dev

         .. admonition:: Note
            :class: tip

            These instructions assume a Debian/Ubuntu Linux distribution.

      .. tab-item:: macOS
         :sync: macos

         .. code-block:: console

            $ brew install libusb
            $ brew install pkg-config

         .. admonition:: Note
            :class: tip

            These instructions assume a brew is installed and used for package
            management.

.. code-block:: console

   $ pw package install freertos
   $ pw package install pico_sdk
   $ pw package install picotool

   $ gn gen out --export-compile-commands --args="
       dir_pw_third_party_freertos=\"//environment/packages/freertos\"
       PICO_SRC_DIR=\"//environment/packages/pico_sdk\"
     "

.. tip::

   Instead of the ``gn gen out`` with args set on the command line above you can
   run:

   .. code-block:: console

      $ gn args out

   Then add the following lines to that text file:

   .. code-block::

      dir_pw_third_party_freertos = getenv("PW_PACKAGE_ROOT") + "/freertos"
      PICO_SRC_DIR = getenv("PW_PACKAGE_ROOT") + "/pico_sdk"

.. _target-rp2040-udev:

Setting up udev rules
=====================
On Linux, you may need to update your udev rules to access the device as a
normal user (not root).

Add the following rules to ``/etc/udev/rules.d/49-pico.rules`` or
``/usr/lib/udev/rules.d/49-pico.rules``. Create the file if it doesn't exist.

.. literalinclude:: /targets/rp2040/49-pico.rules
   :language: linuxconfig
   :start-at: # Raspberry

Then reload the rules:

.. code-block:: console

   sudo udevadm control --reload-rules
   sudo udevadm trigger

Building
========

.. tab-set::

   .. tab-item:: GN
      :sync: GN

      Once the Pico SDK is configured, the Pi Pico will build as part of the default
      GN build:

      .. code-block:: console

         $ ninja -C out

      The pw_system example is available as a separate build target:

      .. code-block:: console

         $ ninja -C out pw_system_demo

Flashing
========

Using the mass-storage booloader
--------------------------------
Hold down the **BOOTSEL** button when plugging in the Pico and it will appear as a
mass storage device. Copy the UF2 firmware image (for example,
``out/rp2040.size_optimized/obj/pw_system/system_example.uf2``) to
your Pico when it is in USB bootloader mode.

.. tip::

   This is the simplest solution if you are fine with physically interacting
   with your Pico whenever you want to flash a new firmware image.

.. _target-rp2040-openocd:

Using OpenOCD
-------------
To flash using OpenOCD, you'll either need a
`Pico debug probe <https://www.raspberrypi.com/products/debug-probe/>`_ or a
second Raspberry Pi Pico to use as a debug probe. Also, on Linux you'll need to
follow the instructions for
:ref:`target-rp2040-udev`.

First-time setup
^^^^^^^^^^^^^^^^
First, flash your first Pi Pico with ``debugprobe_on_pico.uf2`` from `the
latest release of debugprobe <https://github.com/raspberrypi/debugprobe/releases/latest>`_.

Next, connect the two Pico boards as follows:

- Pico probe GND -> target Pico GND
- Pico probe GP2 -> target Pico SWCLK
- Pico probe GP3 -> target Pico SWDIO

If you do not jump VSYS -> VSYS, you'll need to connect both Pi Pico boards
to USB ports so that they have power.

For more detailed instructions on how how to connect two Pico boards, see
``Appendix A: Using Picoprobe`` of the `Getting started with Raspberry Pi Pico
<https://datasheets.raspberrypi.com/pico/getting-started-with-pico.pdf>`_
guide.

Flashing a new firmware
^^^^^^^^^^^^^^^^^^^^^^^
Once your Pico is all wired up, you'll be able to flash it using OpenOCD:

.. code-block:: console

   $ openocd -f interface/cmsis-dap.cfg \
         -f target/rp2040.cfg -c "adapter speed 5000" \
         -c "program out/rp2040.size_optimized/obj/pw_system/bin/system_example.elf verify reset exit"

Typical output:

.. code-block:: none

   xPack Open On-Chip Debugger 0.12.0+dev-01312-g18281b0c4-dirty (2023-09-05-01:33)
   Licensed under GNU GPL v2
   For bug reports, read
      http://openocd.org/doc/doxygen/bugs.html
   Info : Hardware thread awareness created
   Info : Hardware thread awareness created
   adapter speed: 5000 kHz
   Info : Using CMSIS-DAPv2 interface with VID:PID=0x2e8a:0x000c, serial=415032383337300B
   Info : CMSIS-DAP: SWD supported
   Info : CMSIS-DAP: Atomic commands supported
   Info : CMSIS-DAP: Test domain timer supported
   Info : CMSIS-DAP: FW Version = 2.0.0
   Info : CMSIS-DAP: Interface Initialised (SWD)
   Info : SWCLK/TCK = 0 SWDIO/TMS = 0 TDI = 0 TDO = 0 nTRST = 0 nRESET = 0
   Info : CMSIS-DAP: Interface ready
   Info : clock speed 5000 kHz
   Info : SWD DPIDR 0x0bc12477, DLPIDR 0x00000001
   Info : SWD DPIDR 0x0bc12477, DLPIDR 0x10000001
   Info : [rp2040.core0] Cortex-M0+ r0p1 processor detected
   Info : [rp2040.core0] target has 4 breakpoints, 2 watchpoints
   Info : [rp2040.core1] Cortex-M0+ r0p1 processor detected
   Info : [rp2040.core1] target has 4 breakpoints, 2 watchpoints
   Info : starting gdb server for rp2040.core0 on 3333
   Info : Listening on port 3333 for gdb connections
   Warn : [rp2040.core1] target was in unknown state when halt was requested
   [rp2040.core0] halted due to debug-request, current mode: Thread
   xPSR: 0xf1000000 pc: 0x000000ee msp: 0x20041f00
   [rp2040.core1] halted due to debug-request, current mode: Thread
   xPSR: 0xf1000000 pc: 0x000000ee msp: 0x20041f00
   ** Programming Started **
   Info : Found flash device 'win w25q16jv' (ID 0x001540ef)
   Info : RP2040 B0 Flash Probe: 2097152 bytes @0x10000000, in 32 sectors

   Info : Padding image section 1 at 0x10022918 with 232 bytes (bank write end alignment)
   Warn : Adding extra erase range, 0x10022a00 .. 0x1002ffff
   ** Programming Finished **
   ** Verify Started **
   ** Verified OK **
   ** Resetting Target **
   shutdown command invoked

.. tip::

   This is the most robust flashing solution if you don't want to physically
   interact with the attached devices every time you want to flash a Pico.

Running unit tests
==================
Unlike most other targets in Pigweed, the RP2040 uses RPC-based unit testing.
This makes it easier to fully automate on-device tests in a scalable and
maintainable way.

Step 1: Start test server
-------------------------
To allow Ninja to properly serialize tests to run on device, Ninja will send
test requests to a server running in the background. The first step is to launch
this server. By default, the script will attempt to automatically detect an
attached Pi Pico running an application with USB serial enabled or a Pi Debug
Probe, then use it for testing. To override this behavior, provide a custom
server configuration file with ``--server-config``.

.. code-block:: console

   $ python -m rp2040_utils.unit_test_server --chip RP2040

.. tip::

   If the server can't find any attached devices, ensure your Pi Pico is
   already running an application that utilizes USB serial.

.. Warning::

   If you connect or disconnect any boards, you'll need to restart the test
   server for hardware changes to take effect.

Step 2: Configure GN
--------------------
By default, this hardware target has incremental testing disabled. Enabling the
``pw_targets_ENABLE_RP2040_TEST_RUNNER`` build arg tells GN to send requests to
a running ``rp2040_utils.unit_test_server``.

.. code-block:: console

   $ gn args out
   # Modify and save the args file to use pw_target_runner.
   pw_targets_ENABLE_RP2040_TEST_RUNNER = true

Step 3: Build changes
---------------------
Now, whenever you run ``ninja -C out pi_pico``, all tests affected by changes
since the last build will be rebuilt and then run on the attached device.
Alternatively, you may use ``pw watch`` to set up Pigweed to trigger
builds/tests whenever changes to source files are detected.

Connect with pw_console
=======================
Once the board has been flashed, you can connect to it and send RPC commands
via the Pigweed console:

.. tab-set::

   .. tab-item:: Bazel
      :sync: bazel

      .. code-block:: console

         $ bazel run --config=rp2040 //pw_system:system_example_console

   .. tab-item:: GN
      :sync: gn

      .. code-block:: console

         $ pw-system-console --device /dev/{ttyX} --baudrate 115200 \
             --token-databases \
               out/rp2040.size_optimized/obj/pw_system/bin/system_example.elf

      Replace ``{ttyX}`` with the appropriate device on your machine. On Linux
      this may look like ``ttyACM0``, and on a Mac it may look like
      ``cu.usbmodem***``. If ``--device`` is omitted the first detected port
      will be used if there is only one. If multiple ports are detected an
      interactive prompt will be shown.

When the console opens, try sending an Echo RPC request. You should get back
the same message you sent to the device.

.. code-block:: pycon

   >>> device.rpcs.pw.rpc.EchoService.Echo(msg="Hello, Pigweed!")
   (Status.OK, pw.rpc.EchoMessage(msg='Hello, Pigweed!'))

You can also try out our thread snapshot RPC service, which should return a
stack usage overview of all running threads on the device in Host Logs.

.. code-block:: pycon

   >>> device.snapshot_peak_stack_usage()

Example output:

.. code-block::

   20220826 09:47:22  INF  PendingRpc(channel=1, method=pw.thread.ThreadSnapshotService.GetPeakStackUsage) completed: Status.OK
   20220826 09:47:22  INF  Thread State
   20220826 09:47:22  INF    5 threads running.
   20220826 09:47:22  INF
   20220826 09:47:22  INF  Thread (UNKNOWN): IDLE
   20220826 09:47:22  INF  Est CPU usage: unknown
   20220826 09:47:22  INF  Stack info
   20220826 09:47:22  INF    Current usage:   0x20002da0 - 0x???????? (size unknown)
   20220826 09:47:22  INF    Est peak usage:  390 bytes, 76.77%
   20220826 09:47:22  INF    Stack limits:    0x20002da0 - 0x20002ba4 (508 bytes)
   20220826 09:47:22  INF
   20220826 09:47:22  INF  ...

You are now up and running!

.. seealso::

   The :ref:`module-pw_console`
   :bdg-ref-primary-line:`module-pw_console-user_guide` for more info on using
   the pw_console UI.

Interactive debugging
=====================
To interactively debug a Pico, first ensure you are set up for
:ref:`target-rp2040-openocd`.

In one terminal window, start an OpenOCD GDB server with the following command:

.. code-block:: console

   $ openocd -f interface/cmsis-dap.cfg \
         -f target/rp2040.cfg -c "adapter speed 5000"

In a second terminal window, connect to the open GDB server, passing the binary
you will be debugging:

.. code-block:: console

   $ arm-none-eabi-gdb -ex "target remote :3333" \
     out/rp2040.size_optimized/obj/pw_system/bin/system_example.elf

Helpful GDB commands
--------------------
+---------------------------------------------------------+--------------------+
| Action                                                  | shortcut / command |
+=========================================================+====================+
| Reset the running device, stopping immediately          | ``mon reset halt`` |
+---------------------------------------------------------+--------------------+
| Continue execution until pause or breakpoint            |              ``c`` |
+---------------------------------------------------------+--------------------+
| Pause execution                                         |         ``ctrl+c`` |
+---------------------------------------------------------+--------------------+
| Show backtrace                                          |             ``bt`` |
+---------------------------------------------------------+--------------------+
