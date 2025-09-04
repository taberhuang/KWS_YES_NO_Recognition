.. _module-pw_spi:

======
pw_spi
======
.. pigweed-module::
   :name: pw_spi

Pigweed's SPI module provides a set of interfaces for communicating with SPI
responders attached to a target. It also provides an interface for implementing
SPI responders.

--------
Overview
--------
The ``pw_spi`` module provides a series of interfaces that facilitate the
development of SPI responder drivers that are abstracted from the target's
SPI hardware implementation.  The interface consists of these main classes:

- ``pw::spi::Initiator`` - Interface for configuring a SPI bus, and using it
  to transmit and receive data.
- ``pw::spi::ChipSelector`` - Interface for enabling/disabling a SPI
  responder attached to the bus.
- ``pw::spi::Device`` - primary HAL interface used to interact with a SPI
  responder.
- ``pw::spi::Responder`` - Interface for implementing a SPI responder.

``pw_spi`` relies on a target-specific implementations of
``pw::spi::Initiator`` and ``pw::spi::ChipSelector`` to be defined, and
injected into ``pw::spi::Device`` objects which are used to communicate with a
given responder attached to a target's SPI bus.

--------
Examples
--------

Constructing a SPI Device
=========================

.. code-block:: cpp

   constexpr pw::spi::Config kConfig = {
       .polarity = pw::spi::ClockPolarity::kActiveHigh,
       .phase = pw::spi::ClockPhase::kRisingEdge,
       .bits_per_word = pw::spi::BitsPerWord(8),
       .bit_order = pw::spi::BitOrder::kLsbFirst,
   };

   auto initiator = pw::spi::MyInitator();
   auto mutex = pw::sync::VirtualMutex();
   auto selector = pw::spi::MyChipSelector();

   auto device = pw::spi::Device(
      pw::sync::Borrowable<Initiator>(initiator, mutex), kConfig, selector);

This example demonstrates the construction of a ``pw::spi::Device`` from its
object dependencies and configuration data; where ``MyDevice`` and
``MyChipSelector`` are concrete implementations of the ``pw::spi::Initiator``
and ``pw::spi::ChipSelector`` interfaces, respectively.

The use of ``pw::sync::Borrowable`` in the interface provides a
mutual-exclusion wrapper for the injected ``pw::spi::Initiator``, ensuring
that transactions cannot be interrupted or corrupted by other concurrent
workloads making use of the same SPI bus.

Once constructed, the ``device`` object can then be passed to functions used to
perform SPI transfers with a target responder.

Performing a Transfer
=====================

.. code-block:: cpp

   pw::Result<SensorData> ReadSensorData(pw::spi::Device& device) {
     std::array<std::byte, 16> raw_sensor_data;
     constexpr std::array<std::byte, 2> kAccelReportCommand = {
         std::byte{0x13}, std::byte{0x37}};

     // This device supports full-duplex transfers
     PW_TRY(device.WriteRead(kAccelReportCommand, raw_sensor_data));
     return UnpackSensorData(raw_sensor_data);
   }

The ``ReadSensorData()`` function implements a driver function for a contrived
SPI accelerometer.  The function performs a full-duplex transfer with the
device to read its current data.

As this function relies on the ``device`` object that abstracts the details
of bus-access and chip-selection, the function is portable to any target
that implements its underlying interfaces.

Performing a Multi-part Transaction
===================================

.. code-block:: cpp

   pw::Result<SensorData> ReadSensorData(pw::spi::Device& device) {
     std::array<std::byte, 16> raw_sensor_data;
     constexpr std::array<std::byte, 2> kAccelReportCommand = {
         std::byte{0x13}, std::byte{0x37}};

     // Creation of the RAII `transaction` acquires exclusive access to the bus
     pw::spi::Device::Transaction transaction =
       device.StartTransaction(pw::spi::ChipSelectBehavior::kPerTransaction);

     // This device only supports half-duplex transfers
     PW_TRY(transaction.Write(kAccelReportCommand));
     PW_TRY(transaction.Read(raw_sensor_data))

     return UnpackSensorData(raw_sensor_data);

     // Destruction of RAII `transaction` object releases lock on the bus
   }

The code above is similar to the previous example, but makes use of the
``Transaction`` API in ``pw::spi::Device`` to perform separate, half-duplex
``Write()`` and ``Read()`` transfers, as is required by the sensor in this
example.

The use of the RAII ``transaction`` object in this example guarantees that
no other thread can perform transfers on the same SPI bus
(``pw::spi::Initiator``) until it goes out-of-scope.

------------------
pw::spi Interfaces
------------------
The SPI API consists of the following components:

- The ``pw::spi::Initiator`` interface, and its associated configuration data
  structs.
- The ``pw::spi::ChipSelector`` interface.
- The ``pw::spi::Device`` class.
- The ``pw::spi::Responder`` interface.

pw::spi::Initiator
==================
The common interface for configuring a SPI bus, and initiating transfers using
it.

A concrete implementation of this interface class *must* be defined in order
to use ``pw_spi`` with a specific target.

The ``spi::Initiator`` object configures the SPI bus to communicate with a
defined set of common bus parameters that include:

- clock polarity/phase
- bits-per-word (between 3-32 bits)
- bit ordering (LSB or MSB first)

These bus configuration parameters are aggregated in the ``pw::spi::Config``
structure, and passed to the ``pw::spi::Initiator`` via its ``Configure()``
method.

.. inclusive-language: disable

.. Note:

   Throughout ``pw_spi``, the terms "initiator" and "responder" are used to
   describe the two roles SPI devices can implement.  These terms correspond
   to the  "master" and "slave" roles described in legacy documentation
   related to the SPI protocol.

.. inclusive-language: enable

.. cpp:class:: pw::spi::Initiator

   .. cpp:function:: Status Configure(const Config& config)

      Configure the SPI bus to communicate using a specific set of properties,
      including the clock polarity, clock phase, bit-order, and bits-per-word.

      Returns OkStatus() on success, and implementation-specific values on
      failure conditions

   .. cpp:function:: Status WriteRead(ConstByteSpan write_buffer, ByteSpan read_buffer) = 0;

      Perform a synchronous read/write operation on the SPI bus.  Data from the
      `write_buffer` object is written to the bus, while the `read_buffer` is
      populated with incoming data on the bus.  The operation will ensure that
      all requested data is written-to and read-from the bus. In the event the
      read buffer is smaller than the write buffer (or zero-size), any
      additional input bytes are discarded. In the event the write buffer is
      smaller than the read buffer (or zero size), the output is padded with
      0-bits for the remainder of the transfer.

      Returns OkStatus() on success, and implementation-specific values on
      failure.

pw::spi::ChipSelector
=====================
.. doxygenclass:: pw::spi::ChipSelector
   :members:

pw::spi::DigitalOutChipSelector
===============================
.. doxygenclass:: pw::spi::DigitalOutChipSelector
   :members:

pw::spi::Device
===============
This is primary object used by a client to interact with a target SPI device.
It provides a wrapper for an injected ``pw::spi::Initiator`` object, using
its methods to configure the bus and perform individual SPI transfers.  The
injected ``pw::spi::ChipSelector`` object is used internally to activate and
de-activate the device on-demand from within the data transfer methods.

The ``Read()``/``Write()``/``WriteRead()`` methods provide support for
performing individual transfers:  ``Read()`` and ``Write()`` perform
half-duplex operations, where ``WriteRead()`` provides support for
full-duplex transfers.

The ``StartTransaction()`` method provides support for performing multi-part
transfers consisting of a series of ``Read()``/``Write()``/``WriteRead()``
calls, during which the caller is guaranteed exclusive access to the
underlying bus.  The ``Transaction`` objects returned from this method
implements the RAII layer providing exclusive access to the bus; exclusive
access locking is released when the ``Transaction`` object is destroyed/goes
out of scope.

Mutual-exclusion to the ``pw::spi::Initiator`` object is provided by the use of
the ``pw::sync::Borrowable`` object, where the ``pw::spi::Initiator`` object is
"borrowed" for the duration of a transaction.

.. cpp:class:: pw::spi::Device

   .. cpp:function:: Status Read(Bytespan read_buffer)

      Synchronously read data from the SPI responder until the provided
      `read_buffer` is full.
      This call will configure the bus and activate/deactivate chip select
      for the transfer

      Note: This call will block in the event that other clients are currently
      performing transactions using the same SPI Initiator.

      Returns OkStatus() on success, and implementation-specific values on
      failure.

   .. cpp:function:: Status Write(ConstByteSpan write_buffer)

      Synchronously write the contents of `write_buffer` to the SPI responder.
      This call will configure the bus and activate/deactivate chip select
      for the transfer

      Note: This call will block in the event that other clients are currently
      performing transactions using the same SPI Initiator.

      Returns OkStatus() on success, and implementation-specific values on
      failure.

   .. cpp:function:: Status WriteRead(ConstByteSpan write_buffer, ByteSpan read_buffer)

      Perform a synchronous read/write transfer with the SPI responder. Data
      from the `write_buffer` object is written to the bus, while the
      `read_buffer` is populated with incoming data on the bus.  In the event
      the read buffer is smaller than the write buffer (or zero-size), any
      additional input bytes are discarded. In the event the write buffer is
      smaller than the read buffer (or zero size), the output is padded with
      0-bits for the remainder of the transfer.
      This call will configure the bus and activate/deactivate chip select
      for the transfer

      Note: This call will block in the event that other clients are currently
      performing transactions using the same SPI Initiator.

      Returns OkStatus() on success, and implementation-specific values on
      failure.

   .. cpp:function:: Transaction StartTransaction(ChipSelectBehavior behavior)

      Begin a transaction with the SPI device.  This creates an RAII
      `Transaction` object that ensures that only one entity can access the
      underlying SPI bus (Initiator) for the object's duration. The `behavior`
      parameter provides a means for a client to select how the chip-select
      signal will be applied on Read/Write/WriteRead calls taking place with
      the Transaction object. A value of `kPerWriteRead` will activate/deactivate
      chip-select on each operation, while `kPerTransaction` will hold the
      chip-select active for the duration of the Transaction object.

.. cpp:class:: pw::spi::Device::Transaction

   .. cpp:function:: Status Read(Bytespan read_buffer)

      Synchronously read data from the SPI responder until the provided
      `read_buffer` is full.

      Returns OkStatus() on success, and implementation-specific values on
      failure.

   .. cpp:function:: Status Write(ConstByteSpan write_buffer)

      Synchronously write the contents of `write_buffer` to the SPI responder

      Returns OkStatus() on success, and implementation-specific values on
      failure.

   .. cpp:function:: Status WriteRead(ConstByteSpan write_buffer, ByteSpan read_buffer)

      Perform a synchronous read/write transfer on the SPI bus.  Data from the
      `write_buffer` object is written to the bus, while the `read_buffer` is
      populated with incoming data on the bus.  The operation will ensure that
      all requested data is written-to and read-from the bus. In the event the
      read buffer is smaller than the write buffer (or zero-size), any
      additional input bytes are discarded. In the event the write buffer is
      smaller than the read buffer (or zero size), the output is padded with
      0-bits for the remainder of the transfer.

      Returns OkStatus() on success, and implementation-specific values on
      failure.

pw::spi::MockInitiator
======================
A generic mocked backend for ``pw::spi::Initiator``. This is specifically
intended for use when developing drivers for SPI devices. This is structured
around a set of 'transactions' where each transaction contains a write, read and
a status. A transaction list can then be passed to the ``MockInitiator``, where
each consecutive call to read/write will iterate to the next transaction in the
list. An example of this is shown below:

.. code-block:: cpp

   using pw::spi::MakeExpectedTransactionlist;
   using pw::spi::MockInitiator;
   using pw::spi::MockWriteTransaction;

   constexpr auto kExpectWrite1 = pw::bytes::Array<1, 2, 3, 4, 5>();
   constexpr auto kExpectWrite2 = pw::bytes::Array<3, 4, 5>();
   auto expected_transactions = MakeExpectedTransactionArray(
       {MockWriteTransaction(pw::OkStatus(), kExpectWrite1),
        MockWriteTransaction(pw::OkStatus(), kExpectWrite2)});
   MockInitiator spi_mock(expected_transactions);

   // Begin driver code
   ConstByteSpan write1 = kExpectWrite1;
   // write1 is ok as spi_mock expects {1, 2, 3, 4, 5} == {1, 2, 3, 4, 5}
   Status status = spi_mock.WriteRead(write1, ConstByteSpan());

   // Takes the first two bytes from the expected array to build a mismatching
   // span to write.
   ConstByteSpan write2 = pw::span(kExpectWrite2).first(2);
   // write2 fails as spi_mock expects {3, 4, 5} != {3, 4}
   status = spi_mock.WriteRead(write2, ConstByteSpan());
   // End driver code

   // Optionally check if the mocked transaction list has been exhausted.
   // Alternatively this is also called from MockInitiator::~MockInitiator().
   EXPECT_EQ(spi_mock.Finalize(), OkStatus());

pw::spi::Responder
==================
The common interface for implementing a SPI responder. It provides a way to
respond to SPI transactions coming from a SPI initiator in a non-target specific
way. A concrete implementation of the ``Responder`` class should be provided for
the target hardware. Applications can then use it to implement their specific
protocols.

.. code-block:: cpp

   MyResponder responder;
   responder.SetCompletionHandler([](ByteSpan rx_data, Status status) {
     // Handle incoming data from initiator.
     // ...
     // Prepare data to send back to initiator during next SPI transaction.
     responder.WriteReadAsync(tx_data, rx_data);
   });

   // Prepare data to send back to initiator during next SPI transaction.
   responder.WriteReadAsync(tx_data, rx_data)

.. toctree::
   :hidden:
   :maxdepth: 1

   backends
