.. _showcase-sense-tutorial-webapp:

======================================================
12. Communicate with your Pico over the Web Serial API
======================================================
.. _Web Serial API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Serial_API

:ref:`module-pw_web` makes it possible to create custom web apps that
communicate with embedded devices over the `Web Serial API`_. Try
monitoring and controlling your Pico over the web now.

.. _NVM: https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating
.. _those browsers don't support: https://developer.mozilla.org/en-US/docs/Web/API/Web_Serial_API#browser_compatibility

#. If you're using a Debug Probe, disconnect it temporarily. See :bug:`419102596`.

#. Open a terminal.

#. Install `NVM`_. You'll need to close and re-open your terminal
   to get ``nvm`` on your command line path.

   .. note::

      You can skip the NVM steps if you've already got NPM installed
      on your machine. Also, you can install NPM however you want;
      NVM is just one fast option.

#. Use NVM to install the long-term support (LTS) release of Node.js:

   .. code-block:: console

      nvm install --lts

#. Set your working directory to ``web_app``:

   .. code-block:: console

      cd <path>/<to>/sense/web_app

#. Install the web app's dependencies:

   .. code-block:: console

      npm install

#. Run the web app:

   .. code-block:: console

      npm run dev

   In the output, note the line that looks like
   ``> Local: http://127.0.0.1:8000/`` (your actual URL may be different).
   This is the URL you'll need in the next step.

#. Open the web app in Google Chrome or Microsoft Edge.

   .. warning::

      **This application does not work in other browsers like Apple Safari
      or Mozilla Firefox** because `those browsers don't support`_ the Web
      Serial API.

   .. figure:: https://storage.googleapis.com/pigweed-media/sense/20240802/webapp_splash.png

#. Click **Connect**. Select **Pico** if you've only got a Pico connected to your development
   host. If you've got both a Pico and a Debug Probe connected, select **Debug Probe**.

   .. figure:: https://storage.googleapis.com/pigweed-media/sense/20240802/webapp_connect.png

   .. note::

      The Pico is not running a web server. The web app spins up its own local
      server and then communicates with the Pico by sending RPCs over the
      Web Serial API.

   You should see a chart of the Pico's onboard temperature getting updated
   every second.

   The **No Enviro+ Board Found!** warning is expected. Your Pico is currently
   running the ``blinky`` bringup app. In :ref:`showcase-sense-tutorial-prod` you will
   revisit this web app when your Pico is running the ``production`` app and you
   will see more information on the web app.

   .. figure:: https://storage.googleapis.com/pigweed-media/sense/20240802/webapp_home.png

#. Close the browser tab running the web app.

#. In the terminal where you launched the web app, press
   :kbd:`Control+C` to stop the web server.

#. Set your working directory back to the root directory of your Sense repository:

   .. code-block:: console

      cd ..

#. If you're using a Debug Probe, re-connect it now.

.. _showcase-sense-tutorial-webapp-summary:

-------
Summary
-------
Projects built on top of Pigweed often build themselves custom web apps
to make development, support, and manufacturing processes faster. Other
teams create web apps that let their end customers manage their own
devices.

Next, head over to :ref:`showcase-sense-tutorial-factory` to get
familiar with how Pigweed can make it easier to test newly
manufactured devices.
