.. _showcase-sense-tutorial-prod:

===================================
15. Run the air quality monitor app
===================================
Your tour of Pigweed is almost done. Before you go, let's get you
familiar with the application described at
:ref:`showcase-sense-product-concept`. Within the Sense codebase this app
is called ``production``. The purpose of the ``production`` app is to
demonstrate what a medium-complexity application built on top of Pigweed's
software abstractions looks like. We're still perfecting the codebase
structure, but this application can begin to give you an idea of how the
Pigweed team thinks Pigweed-based projects should be structured.

.. important::

   This section requires the :ref:`full hardware
   setup <showcase-sense-tutorial-hardware>`.

First, let's get the app running on your Pico. Then we'll provide
an overview of the code.

.. warning::

   This is just a sample application. It is not suitable for real
   air quality monitoring.

.. _showcase-sense-tutorial-prod-hardware:

--------------------
Set up your hardware
--------------------
This part of the tutorial requires the
:ref:`full setup <showcase-sense-tutorial-hardware>`.

.. _showcase-sense-tutorial-prod-flash:

--------------
Flash the Pico
--------------
Flash the ``production`` app to your Pico:

.. tab-set::

   .. tab-item:: VS Code
      :sync: vsc

      .. tab-set::

         .. tab-item:: Pico 1 (RP2040)
            :sync: rp2040

            In **Bazel Targets** expand **//apps/production**, then right-click
            **:flash_rp2040 (native binary)**, then select **Run target**.

         .. tab-item:: Pico 2 (RP2350)
            :sync: rp2350

            In **Bazel Targets** expand **//apps/production**, then right-click
            **:flash_rp2350 (native binary)**, then select **Run target**.

   .. tab-item:: CLI
      :sync: cli

      .. tab-set::

         .. tab-item:: Pico 1 (RP2040)

            .. code-block:: console

               bazelisk run //apps/production:flash_rp2040

         .. tab-item:: Pico 2 (RP2350)

            .. code-block:: console

               bazelisk run //apps/production:flash_rp2350

.. _showcase-sense-tutorial-prod-logs:

----------------------
Monitor the app's logs
----------------------
The app prints out a lot of informational logs. These logs can
help you grok how the app works. Fire up ``pw_console`` again now:

.. tab-set::

   .. tab-item:: VS Code
      :sync: vsc

      .. tab-set::

         .. tab-item:: Pico 1 (RP2040)
            :sync: rp2040

            In **Bazel Targets** right-click the **:rp2040_console (native binary)**
            target (under **//apps/production**) then select **Run target**.

         .. tab-item:: Pico 2 (RP2350)
            :sync: rp2350

            In **Bazel Targets** right-click the **:rp2350_console (native binary)**
            target (under **//apps/production**) then select **Run target**.

   .. tab-item:: CLI
      :sync: cli

      Run the terminal-based console:

      .. tab-set::

         .. tab-item:: Pico 1 (RP2040)

            .. code-block:: console

               bazelisk run //apps/production:rp2040_console

         .. tab-item:: Pico 2 (RP2350)

            .. code-block:: console

               bazelisk run //apps/production:rp2350_console

See :ref:`showcase-sense-tutorial-sim` if you need a refresher
on how to use ``pw_console``.

.. _showcase-sense-tutorial-prod-alarm:

----------------------------
Trigger an air quality alarm
----------------------------
The default mode of the app is to continuously monitor air quality.
You should see the LED on your Enviro+ in one of the following
states:

* Blue/green: Excellent air quality
* Green: Good air quality
* Orange: Meh air quality
* Red: Bad air quality

.. admonition:: Troubleshooting

   **The LCD screen is blank**. This is expected because we haven't
   implemented display support in the app yet. Stay tuned!

Try triggering an air quality alarm now:

#. Hold a strong chemical such as rubbing alcohol close to the
   **BME688** sensor on your Enviro+ Pack.

   The LED on the Enviro+ Pack should change to orange (meh air quality) or
   red (bad air quality).

The next video is an example of what you should see.

.. raw:: html

   <video preload="metadata" style="width: 100%; height: auto;" controls muted>
     <source type="video/webm"
             src="https://storage.googleapis.com/pigweed-media/sense/20240802/production.mp4#t=0.5"/>
   </video>

.. _showcase-sense-tutorial-prod-thresholds:

----------------------------
Adjust the alarm sensitivity
----------------------------
You can adjust the sensitivity i.e. thresholds of the alarm with
the **A** and **B** buttons on your Enviro+ Pack:

* Press the **A** button repeatedly to increase the sensitivity
  of the alarm. In other words, with only a slight change in
  air quality the LED will shift to orange (meh air quality) or
  red (bad air quality).
* Press the **B** button repeatedly to decrease the sensitivity
  of the alarm. In other words, it takes a bigger change in
  air quality for the LED to shift to orange or red.

.. note::

   The "threshold adjustment" mode that you enter after pressing
   **A** or **B** will automatically exit after 3 seconds of
   inactivity.

In the **Device Logs** of ``pw_console`` you should see the
air quality thresholds change as you press **A** and **B**.
For example, if you quickly press **A** twice (i.e. in less
than a second) you should see a log like this:

.. code-block:: text

   19:38:23  INF  00:00:25.758  STATE  Air quality thresholds set: alarm at 384, silence at 512

That log is telling you that the LED will change to red and start
blinking when the air quality value is less than ``384``.

-------------------------------------------
View more information in the custom web app
-------------------------------------------
Now that your Pico is running the full ``production`` app,
the custom web app that was demonstrated in :ref:`showcase-sense-tutorial-webapp`
will show you more information if you fire it up again.

.. _showcase-sense-tutorial-prod-code:

-------------
Code overview
-------------
.. _Sense codebase: https://cs.opensource.google/pigweed/showcase/sense

As mentioned in the intro of this page, the ``production`` app
provides a good start for figuring out how to structure your
Pigweed-based project. It's not perfect yet, but it's a solid
start. We'll leave it up to you to study the code in-depth, but
here are some pointers on the relevant parts of the `Sense codebase`_:

* ``//apps/production/*``: The app's entrypoint code.
* ``//modules/*``: Portable business logic, algorithms, state handling, etc.
  Look at the header includes in ``//apps/production/main.cc`` to figure out
  what modules to study.
* ``//system/*``: System global accesors. Gives access to pre-created instances
  of portable system interfaces. For example, ``am::system::RpcServer()``
  returns the RPC server instance.

.. _showcase-sense-tutorial-prod-summary:

-------
Summary
-------
You now have a rudimentary but working air quality monitor. More
importantly, the code that powers your new air quality monitor is
a solid (but not perfect) starting point for learning how to structure
your own Pigweed-powered products.

Next, head over to :ref:`showcase-sense-tutorial-crash-handler` to learn about
the pigweed crash handler and crash snapshots.
