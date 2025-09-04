.. _module-pw_web:

------
pw_web
------
.. pigweed-module::
   :name: pw_web

Pigweed provides an NPM package with modules to build web apps for Pigweed
devices.

Getting Started
===============

Easiest way to get started is to follow the :ref:`Sense tutorial <showcase-sense-tutorial-intro>`
and flash a Raspberry Pico board.

Once you have a device running Pigweed, you can connect to it using just your web browser.

Installation
-------------
If you have a bundler set up, you can install ``pigweedjs`` in your web application by:

.. code-block:: bash

   $ npm install --save pigweedjs


After installing, you can import modules from ``pigweedjs`` in this way:

.. code-block:: javascript

   import { pw_rpc, pw_tokenizer, Device, WebSerial } from 'pigweedjs';

Import Directly in HTML
^^^^^^^^^^^^^^^^^^^^^^^
If you don't want to set up a bundler, you can also load Pigweed directly in
your HTML page by:

.. code-block:: html

   <script src="https://unpkg.com/pigweedjs/dist/index.umd.js"></script>
   <script>
     const { pw_rpc, pw_hdlc, Device, WebSerial } from Pigweed;
   </script>

Modules
=======
.. _module-pw_web-device:

Device
------
Device class is a helper API to connect to a device over serial and call RPCs
easily.

To initialize device, it needs a ``ProtoCollection`` instance. ``pigweedjs``
includes a default one which you can use to get started, you can also generate
one from your own ``.proto`` files using ``pw_proto_compiler``.

``Device`` goes through all RPC methods in the provided ProtoCollection. For
each RPC, it reads all the fields in ``Request`` proto and generates a
JavaScript function to call that RPC and also a helper method to create a request.
It then makes this function available under ``rpcs.*`` namespaced by its package name.

Device has following public API:

- ``constructor(ProtoCollection, WebSerialTransport <optional>, channel <optional>, rpcAddress <optional>)``
- ``connect()`` - Shows browser's WebSerial connection dialog and let's user
  make device selection
- ``rpcs.*`` - Device API enumerates all RPC services and methods present in the
  provided proto collection and makes them available as callable functions under
  ``rpcs``. Example: If provided proto collection includes Pigweed's Echo
  service ie. ``pw.rpc.EchoService.Echo``, it can be triggered by calling
  ``device.rpcs.pw.rpc.EchoService.Echo.call(request)``. The functions return
  a ``Promise`` that resolves an array with status and response.

Using Device API with Sense
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sense project uses ``pw_log_rpc``; an RPC-based logging solution. Sense
also uses pw_tokenizer to tokenize strings and save device space. Below is an
example that streams logs using the ``Device`` API.

.. code-block:: html

   <h1>Hello Pigweed</h1>
   <button onclick="connect()">Connect</button>
   <br /><br />
   <code></code>
   <script src="https://unpkg.com/pigweedjs/dist/index.umd.js"></script>
   <script src="https://unpkg.com/pigweedjs/dist/protos/collection.umd.js"></script>
   <script>
     const { Device, pw_tokenizer } = Pigweed;
     const { ProtoCollection } = PigweedProtoCollection;
     const tokenDBCsv = `...` // Load token database here

     const device = new Device(new ProtoCollection());
     const detokenizer = new pw_tokenizer.Detokenizer(tokenDBCsv);

     async function connect(){
       await device.connect();
       const req = device.rpcs.pw.log.Logs.Listen.createRequest()
       const logs = device.rpcs.pw.log.Logs.Listen.call(req);
       for await (const msg of logs){
           msg.getEntriesList().forEach((entry) => {
             const frame = entry.getMessage();
             const detokenized = detokenizer.detokenizeUint8Array(frame);
             document.querySelector('code').innerHTML += detokenized + "<br/>";
           });
       }
       console.log("Log stream ended with status", logs.call.status);
     }
   </script>

The above example requires a token database in CSV format. You can generate one
from the Sense's ``.elf`` file by running:

.. code-block:: bash

   $ pw_tokenizer/py/pw_tokenizer/database.py create \
   --database db.csv bazel-bin/apps/blinky/rp2040_blinky.elf

You can then load this CSV in JavaScript using ``fetch()`` or by just copying
the contents into the ``tokenDBCsv`` variable in the above example.

WebSerialTransport
------------------
To help with connecting to WebSerial and listening for serial data, a helper
class is also included under ``WebSerial.WebSerialTransport``. Here is an
example usage:

.. code-block:: javascript

   import { WebSerial, pw_hdlc } from 'pigweedjs';

   const transport = new WebSerial.WebSerialTransport();
   const decoder = new pw_hdlc.Decoder();

   // Present device selection prompt to user
   await transport.connect();

   // Or connect to an existing `SerialPort`
   // await transport.connectPort(port);

   // Listen and decode HDLC frames
   transport.chunks.subscribe((item) => {
     const decoded = decoder.process(item);
     for (const frame of decoded) {
       if (frame.address === 1) {
         const decodedLine = new TextDecoder().decode(frame.data);
         console.log(decodedLine);
       }
     }
   });

   // Later, close all streams and close the port.
   transport.disconnect();

Individual Modules
==================
Following Pigweed modules are included in the NPM package:

- `pw_hdlc <https://pigweed.dev/pw_hdlc/#typescript>`_
- `pw_rpc <https://pigweed.dev/pw_rpc/ts/>`_
- `pw_tokenizer <https://pigweed.dev/pw_tokenizer/#typescript>`_
- `pw_transfer <https://pigweed.dev/pw_transfer/#typescript>`_

Log Viewer Component
====================
The NPM package also includes a log viewer component that can be embedded in any
webapp. The component works with Pigweed's RPC stack out-of-the-box but also
supports defining your own log source. See :ref:`module-pw_web-log-viewer` for
component interaction details.

The component is composed of the component itself and a log source. Here is a
simple example app that uses a mock log source:

.. code-block:: html

   <div id="log-viewer-container"></div>
   <script src="https://unpkg.com/pigweedjs/dist/logging.umd.js"></script>
   <script>

     const { createLogViewer, MockLogSource } = PigweedLogging;
     const logSource = new MockLogSource();
     const containerEl = document.querySelector(
       '#log-viewer-container'
     );

     let unsubscribe = createLogViewer(logSource, containerEl);
     logSource.start(); // Start producing mock logs

   </script>

The code above will render a working log viewer that just streams mock
log entries.

It also comes with an RPC log source with support for detokenization. Here is an
example app using that:

.. code-block:: html

   <div id="log-viewer-container"></div>
   <script src="https://unpkg.com/pigweedjs/dist/index.umd.js"></script>
   <script src="https://unpkg.com/pigweedjs/dist/protos/collection.umd.js"></script>
   <script src="https://unpkg.com/pigweedjs/dist/logging.umd.js"></script>
   <script>

     const { Device, pw_tokenizer } = Pigweed;
     const { ProtoCollection } = PigweedProtoCollection;
     const { createLogViewer, PigweedRPCLogSource } = PigweedLogging;

     const device = new Device(new ProtoCollection());
     const logSource = new PigweedRPCLogSource(device, "CSV TOKEN DB HERE");
     const containerEl = document.querySelector(
       '#log-viewer-container'
     );

     let unsubscribe = createLogViewer(logSource, containerEl);

   </script>

Custom Log Source
-----------------
You can define a custom log source that works with the log viewer component by
just extending the abstract `LogSource` class and emitting the `logEntry` events
like this:

.. code-block:: typescript

   import { LogSource, LogEntry, Level } from 'pigweedjs/logging';

   export class MockLogSource extends LogSource {
     constructor(){
       super();
       // Do any initializations here
       // ...
       // Then emit logs
       const log1: LogEntry = {

       }
       this.publishLogEntry({
         level: Level.INFO,
         timestamp: new Date(),
         fields: [
           { key: 'level', value: level }
           { key: 'timestamp', value: new Date().toISOString() },
           { key: 'source', value: "LEFT SHOE" },
           { key: 'message', value: "Running mode activated." }
         ]
       });
     }
   }

After this, you just need to pass your custom log source object
to `createLogViewer()`. See implementation of
`PigweedRPCLogSource <https://cs.opensource.google/pigweed/pigweed/+/main:ts/logging_source_rpc.ts>`_
for reference.

Column Order
------------
Column Order can be defined on initialization with the optional ``columnOrder`` parameter.
Only fields that exist in the Log Source will render as columns in the Log Viewer.

.. code-block:: typescript

   createLogViewer(logSource, root, { columnOrder })

``columnOrder`` accepts an ``string[]`` and defaults to ``[log_source, time, timestamp]``

.. code-block:: typescript

   createLogViewer(
    logSource,
    root,
    { columnOrder: ['log_source', 'time', 'timestamp'] }

  )

Note, columns will always start with ``level`` and end with ``message``, these fields do not need to be defined.
Columns are ordered in the following format:

1. ``level``
2. ``columnOrder``
3. Fields that exist in Log Source but not listed will be added here.
4. ``message``


Accessing and Modifying Log Views
---------------------------------

It can be challenging to access and manage log views directly through JavaScript or HTML due to the
shadow DOM boundaries generated by custom elements. To facilitate this, the ``Log Viewer``
component has a public property, ``logViews``, which returns an array containing all child log
views. Here is an example that modifies the ``viewTitle`` and ``searchText`` properties of two log
views:

.. code-block:: typescript

   const logViewer = containerEl.querySelector('log-viewer');
   const views = logViewer?.logViews;

   if (views) {
     views[0].viewTitle = 'Device A Logs';
     views[0].searchText = 'device:A';

     views[1].viewTitle = 'Device B Logs';
     views[1].searchText = 'device:B';
   }

Alternatively, you can define a state object containing nodes with their respective properties and
pass this state object to the ``Log Viewer`` during initialization. Here is how you can achieve
that:

.. code-block:: typescript

   const childNodeA: ViewNode = new ViewNode({
     type: NodeType.View,
     viewTitle: 'Device A Logs',
     searchText: 'device:A'
   });

   const childNodeB: ViewNode = new ViewNode({
     type: NodeType.View,
     viewTitle: 'Device B Logs',
     searchText: 'device:B'
   });

   const rootNode: ViewNode = new ViewNode({
     type: NodeType.Split,
     orientation: Orientation.Vertical,
     children: [childNodeA, childNodeB]
   });

   const options = { state: { rootNode: rootNode } };
   createLogViewer(logSources, containerEl, options);

Note that the relevant types and enums should be imported from
``log-viewer/src/shared/view-node.ts``.

Color Scheme
------------
The log viewer web component provides the ability to set the color scheme
manually, overriding any default or system preferences.

To set the color scheme, first obtain a reference to the ``log-viewer`` element
in the DOM. A common way to do this is by using ``querySelector()``:

.. code-block:: javascript

   const logViewer = document.querySelector('log-viewer');

You can then set the color scheme dynamically by updating the component's
`colorScheme` property or by setting a value for the `colorscheme` HTML attribute.

.. code-block:: javascript

   logViewer.colorScheme = 'dark';

.. code-block:: javascript

   logViewer.setAttribute('colorscheme', 'dark');

The color scheme can be set to ``'dark'``, ``'light'``, or the default ``'auto'``
which allows the component to adapt to the preferences in the operating system
settings.

Material Icon Font (Subsetting)
-------------------------------
.. inclusive-language: disable

The Log Viewer uses a subset of the Material Symbols Rounded icon font fetched via the `Google Fonts API <https://developers.google.com/fonts/docs/css2#forming_api_urls>`_. However, we also provide a subset of this font for offline usage at `GitHub <https://github.com/google/material-design-icons/blob/master/variablefont/MaterialSymbolsRounded%5BFILL%2CGRAD%2Copsz%2Cwght%5D.woff2>`_
with codepoints listed in the `codepoints <https://github.com/google/material-design-icons/blob/master/variablefont/MaterialSymbolsRounded%5BFILL%2CGRAD%2Copsz%2Cwght%5D.codepoints>`_ file.

(It's easiest to look up the codepoints at `fonts.google.com <https://fonts.google.com/icons?selected=Material+Symbols+Rounded>`_ e.g. see
the sidebar shows the Codepoint for `"home" <https://fonts.google.com/icons?selected=Material+Symbols+Rounded:home:FILL@0;wght@0;GRAD@0;opsz@NaN>`_ is e88a).

The following icons with codepoints are curently used:

* delete_sweep e16c
* error e000
* warning f083
* cancel e5c9
* bug_report e868
* info e88e
* view_column e8ec
* brightness_alert f5cf
* wrap_text e25b
* more_vert e5d4
* play_arrow e037
* stop e047

To save load time and bandwidth, we provide a pre-made subset of the font with
just the codepoints we need, which reduces the font size from 3.74MB to 12KB.

We use fonttools (https://github.com/fonttools/fonttools) to create the subset.
To create your own subset, find the codepoints you want to add and:

1. Start a python virtualenv and install fonttools

.. code-block:: bash

   virtualenv env
   source env/bin/activate
   pip install fonttools brotli

2. Download the the raw `MaterialSybmolsRounded woff2 file <https://github.com/google/material-design-icons/tree/master/variablefont>`_

.. code-block:: bash

   # line below for example, the url is not stable: e.g.
   curl -L -o MaterialSymbolsRounded.woff2 \
     "https://github.com/google/material-design-icons/raw/master/variablefont/MaterialSymbolsRounded%5BFILL,GRAD,opsz,wght%5D.woff2"

3. Run fonttools, passing in the unicode codepoints of the necessary glyphs.
   (The points for letters a-z, numbers 0-9 and underscore character are
   necessary for creating ligatures)

.. warning::  Ensure there are no spaces in the list of codepoints.
.. code-block:: bash

   fonttools subset MaterialSymbolsRounded.woff2 \
      --unicodes=5f-7a,30-39,e16c,e000,e002,e8b2,e5c9,e868,,e88e,e8ec,f083,f5cf,e25b,e5d4,e037,e047 \
      --no-layout-closure \
      --output-file=material_symbols_rounded_subset.woff2 \
      --flavor=woff2

4. Update ``material_symbols_rounded_subset.woff2`` in ``log_viewer/src/assets``
   with the new subset

.. inclusive-language: enable

Shoelace
--------
We currently use Split Panel from the `Shoelace <https://github.com/shoelace-style/shoelace>`_
library to enable resizable split views within the log viewer.

To provide flexibility in different environments, we've introduced a property ``useShoelaceFeatures``
in the ``LogViewer`` component. This flag allows developers to enable or disable the import and
usage of Shoelace components based on their needs.

By default, the ``useShoelaceFeatures`` flag is set to ``true``, meaning Shoelace components will
be used and resizable split views are made available. To disable Shoelace components, set this
property to ``false`` as shown below:

.. code-block:: javascript

   const logViewer = document.querySelector('log-viewer');
   logViewer.useShoelaceFeatures = false;

When ``useShoelaceFeatures`` is set to ``false``, the  <sl-split-panel> component from Shoelace will
not be imported or used within the log viewer.

Guides
======

.. toctree::
  :maxdepth: 1

  testing
  log_viewer
  repl
