// Copyright 2024 The Pigweed Authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package dev.pigweed.pw_transfer;

import static dev.pigweed.pw_transfer.TransferProgress.UNKNOWN_TRANSFER_SIZE;
import static java.lang.Math.max;
import static java.lang.Math.min;

import dev.pigweed.pw_log.Logger;
import dev.pigweed.pw_rpc.Status;
import dev.pigweed.pw_transfer.TransferEventHandler.TransferInterface;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

class ReadTransfer extends Transfer<byte[]> {
  private static final Logger logger = Logger.forClass(ReadTransfer.class);

  // The fractional position within a window at which a receive transfer should
  // extend its window size to minimize the amount of time the transmitter
  // spends blocked.
  //
  // For example, a divisor of 2 will extend the window when half of the
  // requested data has been received, a divisor of three will extend at a third
  // of the window, and so on.
  private static final int EXTEND_WINDOW_DIVISOR = 2;

  // To minimize copies, store the ByteBuffers directly from the chunk protos in a
  // list.
  private final List<ByteBuffer> dataChunks = new ArrayList<>();
  private int totalDataSize = 0;

  private final TransferParameters parameters;

  private long remainingTransferSize = UNKNOWN_TRANSFER_SIZE;

  private int windowEndOffset = 0;

  private int windowSize = 0;
  private int windowSizeMultiplier = 1;
  private TransmitPhase transmitPhase = TransmitPhase.SLOW_START;

  private int lastReceivedOffset = 0;

  // Slow start and congestion avoidance are analogues to the equally named phases
  // in TCP congestion
  // control.
  private enum TransmitPhase {
    SLOW_START,
    CONGESTION_AVOIDANCE,
  }

  // The type of data transmission the transfer is requesting.
  private enum TransmitAction {
    // Immediate parameters sent at the start of a new transfer for legacy
    // compatibility.
    BEGIN,

    // Initial parameters chunk following the opening handshake.
    FIRST_PARAMETERS,

    // Extend the current transmission window.
    EXTEND,

    // Rewind the transfer to a certain offset following data loss.
    RETRANSMIT,
  }

  ReadTransfer(int resourceId,
      int sessionId,
      ProtocolVersion desiredProtocolVersion,
      TransferInterface transferManager,
      TransferTimeoutSettings timeoutSettings,
      TransferParameters transferParameters,
      Consumer<TransferProgress> progressCallback,
      BooleanSupplier shouldAbortCallback,
      int initialOffset) {
    super(resourceId,
        sessionId,
        desiredProtocolVersion,
        transferManager,
        timeoutSettings,
        progressCallback,
        shouldAbortCallback,
        initialOffset);
    this.parameters = transferParameters;
    this.windowEndOffset = parameters.maxChunkSizeBytes();
    this.windowSize = parameters.maxChunkSizeBytes();
  }

  final TransferParameters getParametersForTest() {
    return parameters;
  }

  @Override
  State getWaitingForDataState() {
    return new ReceivingData();
  }

  @Override
  void prepareInitialChunk(VersionedChunk.Builder chunk) {
    chunk.setInitialOffset(getOffset());
    setTransferParameters(chunk);
  }

  @Override
  VersionedChunk getChunkForRetry() {
    VersionedChunk chunk = getLastChunkSent();
    // If the last chunk sent was transfer parameters, send an updated RETRANSMIT
    // chunk.
    if (chunk.type() == Chunk.Type.PARAMETERS_CONTINUE
        || chunk.type() == Chunk.Type.PARAMETERS_RETRANSMIT) {
      return prepareTransferParameters(TransmitAction.RETRANSMIT);
    }
    return chunk;
  }

  private class ReceivingData extends ActiveState {
    @Override
    public void handleDataChunk(VersionedChunk chunk) throws TransferAbortedException {
      // Track the last seen offset so the DropRecovery state can detect retried
      // packets.
      lastReceivedOffset = chunk.offset();

      if (chunk.offset() != getOffset()) {
        // If the chunk's data has already been received, don't go through a full
        // recovery cycle to avoid shrinking the window size and potentially
        // thrashing. The expected data may already be in-flight, so just allow
        // the transmitter to keep going with a CONTINUE parameters chunk.
        if (chunk.offset() + chunk.data().size() <= getOffset()) {
          logger.atFine().log("%s received duplicate chunk with offset offset %d",
              ReadTransfer.this,
              chunk.offset());
          sendChunk(prepareTransferParameters(TransmitAction.EXTEND, false));
        } else {
          logger.atFine().log("%s expected offset %d, received %d; resending transfer parameters",
              ReadTransfer.this,
              getOffset(),
              chunk.offset());

          // For now, only in-order transfers are supported. If data is received out of
          // order, discard this data and retransmit from the last received offset.
          sendChunk(prepareTransferParameters(TransmitAction.RETRANSMIT));
          changeState(new DropRecovery());
        }
        setNextChunkTimeout();
        return;
      }

      // Add the underlying array(s) to a list to avoid making copies of the data.
      dataChunks.addAll(chunk.data().asReadOnlyByteBufferList());
      totalDataSize += chunk.data().size();

      setOffset(getOffset() + chunk.data().size());

      if (chunk.remainingBytes().isPresent()) {
        if (chunk.remainingBytes().getAsLong() == 0) {
          setStateTerminatingAndSendFinalChunk(Status.OK);
          return;
        }

        remainingTransferSize = chunk.remainingBytes().getAsLong();
      } else if (remainingTransferSize != UNKNOWN_TRANSFER_SIZE) {
        // If remaining size was not specified, update based on the most recent
        // estimate, if any.
        remainingTransferSize = max(remainingTransferSize - chunk.data().size(), 0);
      }

      if (remainingTransferSize == UNKNOWN_TRANSFER_SIZE || remainingTransferSize == 0) {
        updateProgress(getOffset(), getOffset(), UNKNOWN_TRANSFER_SIZE);
      } else {
        updateProgress(getOffset(), getOffset(), getOffset() + remainingTransferSize);
      }

      int remainingWindowSize = windowEndOffset - getOffset();
      boolean extendWindow = remainingWindowSize <= windowSize / EXTEND_WINDOW_DIVISOR;

      if (remainingWindowSize == 0) {
        logger.atFinest().log(
            "%s received all pending bytes; sending transfer parameters update", ReadTransfer.this);
        sendChunk(prepareTransferParameters(TransmitAction.EXTEND));
      } else if (extendWindow) {
        sendChunk(prepareTransferParameters(TransmitAction.EXTEND));
      }
      setNextChunkTimeout();
    }
  }

  /** State for recovering from dropped packets. */
  private class DropRecovery extends ActiveState {
    @Override
    public void handleDataChunk(VersionedChunk chunk) throws TransferAbortedException {
      if (chunk.offset() == getOffset()) {
        logger.atFine().log(
            "%s received expected offset %d, resuming transfer", ReadTransfer.this, getOffset());
        changeState(new ReceivingData()).handleDataChunk(chunk);
        return;
      }

      // To avoid a flood of identical parameters packets, only send one if a retry is
      // detected.
      if (chunk.offset() == lastReceivedOffset) {
        logger.atFiner().log(
            "%s received repeated offset %d: retry detected, resending transfer parameters",
            ReadTransfer.this,
            lastReceivedOffset);
        sendChunk(prepareTransferParameters(TransmitAction.RETRANSMIT));
      } else {
        lastReceivedOffset = chunk.offset();
        logger.atFiner().log("%s expecting offset %d, ignoring received offset %d",
            ReadTransfer.this,
            getOffset(),
            chunk.offset());
      }
      setNextChunkTimeout();
    }
  }

  @Override
  void setFutureResult() {
    updateProgress(totalDataSize, totalDataSize, totalDataSize);

    ByteBuffer result = ByteBuffer.allocate(totalDataSize);
    dataChunks.forEach(result::put);
    set(result.array());
  }

  private VersionedChunk prepareTransferParameters(TransmitAction action) {
    return prepareTransferParameters(action, true);
  }

  private VersionedChunk prepareTransferParameters(TransmitAction action, boolean update) {
    Chunk.Type type;

    switch (action) {
      case BEGIN:
        // Initial window is always one chunk. No special handling required.
        type = Chunk.Type.START;
        break;

      case FIRST_PARAMETERS:
        // Initial window is always one chunk. No special handling required.
        type = Chunk.Type.PARAMETERS_RETRANSMIT;
        break;

      case EXTEND:
        // Window was received successfully without packet loss and should grow. Double
        // the window
        // size during slow start, or increase it by a single chunk in congestion
        // avoidance.
        type = Chunk.Type.PARAMETERS_CONTINUE;

        if (update) {
          if (transmitPhase == TransmitPhase.SLOW_START) {
            windowSizeMultiplier *= 2;
          } else {
            windowSizeMultiplier += 1;
          }

          // The window size can never exceed the user-specified maximum bytes. If it
          // does, reduce
          // the multiplier to the largest size that fits.
          windowSizeMultiplier = min(windowSizeMultiplier, parameters.maxChunksInWindow());
        }
        break;

      case RETRANSMIT:
        // A packet was lost: shrink the window size. Additionally, after the first
        // packet loss,
        // transition from the slow start to the congestion avoidance phase of the
        // transfer.
        type = Chunk.Type.PARAMETERS_RETRANSMIT;
        if (update) {
          windowSizeMultiplier = max(windowSizeMultiplier / 2, 1);
          if (transmitPhase == TransmitPhase.SLOW_START) {
            transmitPhase = TransmitPhase.CONGESTION_AVOIDANCE;
          }
        }
        break;

      default:
        throw new AssertionError("Invalid transmit action");
    }

    if (update) {
      windowSize = windowSizeMultiplier * parameters.maxChunkSizeBytes();
      windowEndOffset = getOffset() + windowSize;
    }

    return setTransferParameters(newChunk(type)).build();
  }

  private VersionedChunk.Builder setTransferParameters(VersionedChunk.Builder chunk) {
    chunk.setMaxChunkSizeBytes(parameters.maxChunkSizeBytes())
        .setOffset(getOffset())
        .setWindowEndOffset(windowEndOffset);
    if (parameters.chunkDelayMicroseconds() > 0) {
      chunk.setMinDelayMicroseconds(parameters.chunkDelayMicroseconds());
    }
    return chunk;
  }
}
