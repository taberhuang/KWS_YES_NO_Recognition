// Copyright 2021 The Pigweed Authors
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

package dev.pigweed.pw_rpc;

import com.google.errorprone.annotations.concurrent.GuardedBy;
import com.google.protobuf.ByteString;
import com.google.protobuf.MessageLite;
import dev.pigweed.pw_log.Logger;
import dev.pigweed.pw_rpc.internal.Packet.RpcPacket;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Tracks the state of service method invocations.
 *
 * The RPC endpoint handles all RPC-related events and actions. It synchronizes interactions between
 * the endpoint and any threads interacting with RPC call objects.
 *
 * The Endpoint's intrinsic lock is held when updating the channels or pending calls lists. Call
 * objects only make updates to their own state through function calls made from the Endpoint, which
 * ensures their states are also guarded by the Endpoint's lock. Updates to call objects are
 * enqueued while the lock is held and processed after releasing the lock. This ensures updates
 * occur in order without needing to hold the Endpoint's lock while possibly executing user code.
 */
class Endpoint {
  private static final Logger logger = Logger.forClass(Endpoint.class);

  // Call IDs are varint encoded. Limit the varint size to 2 bytes (14 usable bits).
  private static final int MAX_CALL_ID = 1 << 14;
  static final int FIRST_CALL_ID = 1;
  // These call Ids are specifically reserved for open call ids.
  static final int LEGACY_OPEN_CALL_ID = 0;
  static final int OPEN_CALL_ID = -1;

  private final Packets packets;
  private final Map<Integer, Channel> channels;
  private final Map<PendingRpc, AbstractCall<?, ?>> pending = new HashMap<>();
  private final BlockingQueue<Runnable> callUpdates = new LinkedBlockingQueue<>();
  private final int maxCallId;

  @GuardedBy("this") private int nextCallId = FIRST_CALL_ID;

  Endpoint(CallIdMode callIdMode, List<Channel> channels) {
    this(callIdMode, channels, MAX_CALL_ID);
  }

  /** Create endpoint with {@code maxCallId} possible call_ids for testing purposes */
  Endpoint(CallIdMode callIdMode, List<Channel> channels, int maxCallId) {
    this.packets = new Packets(callIdMode);
    this.channels = channels.stream().collect(Collectors.toMap(Channel::id, c -> c));
    this.maxCallId = maxCallId;
  }

  /**
   * Creates an RPC call object and invokes the RPC
   *
   * @param channelId the channel to use
   * @param method the service method to invoke
   * @param createCall function that creates the call object
   * @param request the request proto; null if this is a client streaming RPC
   * @throws InvalidRpcChannelException if channelId is invalid
   */
  <RequestT extends MessageLite, CallT extends AbstractCall<?, ?>> CallT invokeRpc(int channelId,
      Method method,
      BiFunction<Endpoint, PendingRpc, CallT> createCall,
      @Nullable RequestT request) throws ChannelOutputException {
    CallT call = createCall(channelId, method, createCall);

    // Attempt to start the call.
    logger.atFiner().log("Starting %s", call);

    try {
      // If sending the packet fails, the RPC is never considered pending.
      call.rpc().channel().send(packets.request(call.rpc(), request));
    } catch (ChannelOutputException e) {
      call.handleExceptionOnInitialPacket(e);
    }
    registerCall(call);
    return call;
  }

  /**
   * Starts listening to responses for an RPC locally, but does not send any packets.
   *
   * <p>The RPC remains open until it is closed by the server (either from a response or error
   * packet) or cancelled.
   */
  <CallT extends AbstractCall<?, ?>> CallT openRpc(
      int channelId, Method method, BiFunction<Endpoint, PendingRpc, CallT> createCall) {
    CallT call = createCall(channelId, method, createCall, OPEN_CALL_ID);
    logger.atFiner().log("Opening %s", call);
    registerCall(call);
    return call;
  }

  private <CallT extends AbstractCall<?, ?>> CallT createCall(
      int channelId, Method method, BiFunction<Endpoint, PendingRpc, CallT> createCall) {
    return createCall(channelId, method, createCall, getNewCallId());
  }

  private <CallT extends AbstractCall<?, ?>> CallT createCall(int channelId,
      Method method,
      BiFunction<Endpoint, PendingRpc, CallT> createCall,
      int callId) {
    Channel channel = channels.get(channelId);
    if (channel == null) {
      throw InvalidRpcChannelException.unknown(channelId);
    }

    // Use 0 for call ID when IDs are disabled, which is equivalent to an unset ID in the packet.
    PendingRpc pendingRpc =
        PendingRpc.create(channel, method, packets.callIdsEnabled() ? callId : 0);
    return createCall.apply(this, pendingRpc);
  }

  private void registerCall(AbstractCall<?, ?> call) {
    pending.put(call.rpc(), call);
  }

  /** Enqueues call object updates to make after release the Endpoint's lock. */
  private void enqueueCallUpdate(Runnable callUpdate) {
    while (!callUpdates.add(callUpdate)) {
      // Retry until added successfully
    }
  }

  /** Processes all enqueued call updates; the lock must NOT be held when this is called. */
  private void processCallUpdates() {
    while (true) {
      Runnable callUpdate = callUpdates.poll();
      if (callUpdate == null) {
        break;
      }
      callUpdate.run();
    }
  }

  /** Cancels an ongoing RPC */
  public boolean cancel(AbstractCall<?, ?> call) throws ChannelOutputException {
    try {
      synchronized (this) {
        if (pending.remove(call.rpc()) == null) {
          return false;
        }

        enqueueCallUpdate(() -> call.handleError(Status.CANCELLED));
        call.sendPacket(packets.cancel(call.rpc()));
      }
    } finally {
      logger.atFiner().log("Cancelling %s", call);
      processCallUpdates();
    }
    return true;
  }

  /** Cancels an ongoing RPC without sending a cancellation packet. */
  public boolean abandon(AbstractCall<?, ?> call) {
    synchronized (this) {
      if (pending.remove(call.rpc()) == null) {
        return false;
      }
      enqueueCallUpdate(() -> call.handleError(Status.CANCELLED));
    }
    logger.atFiner().log("Abandoning %s", call);
    processCallUpdates();
    return true;
  }

  public synchronized boolean clientStream(AbstractCall<?, ?> call, MessageLite payload)
      throws ChannelOutputException {
    return sendPacket(call, packets.clientStream(call.rpc(), payload));
  }

  public synchronized boolean clientStreamEnd(AbstractCall<?, ?> call)
      throws ChannelOutputException {
    return sendPacket(call, packets.clientStreamEnd(call.rpc()));
  }

  private boolean sendPacket(AbstractCall<?, ?> call, byte[] packet) throws ChannelOutputException {
    if (!pending.containsKey(call.rpc())) {
      return false;
    }
    // TODO(hepler): Consider aborting the call if sending the packet fails.
    call.sendPacket(packet);
    return true;
  }

  public synchronized void openChannel(Channel channel) {
    if (channels.putIfAbsent(channel.id(), channel) != null) {
      throw InvalidRpcChannelException.duplicate(channel.id());
    }
  }

  public boolean closeChannel(int id) {
    synchronized (this) {
      if (channels.remove(id) == null) {
        return false;
      }
      pending.values().stream().filter(call -> call.getChannelId() == id).forEach(call -> {
        enqueueCallUpdate(() -> call.handleError(Status.ABORTED));
      });
    }
    processCallUpdates();
    return true;
  }

  private boolean handleNext(PendingRpc rpc, ByteString payload) {
    AbstractCall<?, ?> call = getCall(rpc);
    if (call == null) {
      return false;
    }
    logger.atFiner().log("%s received server stream with %d B payload", call, payload.size());
    enqueueCallUpdate(() -> call.handleNext(payload));
    return true;
  }

  private boolean handleUnaryCompleted(PendingRpc rpc, ByteString payload, Status status) {
    PendingRpc rpcToRemove = getRpc(rpc);
    if (rpcToRemove == null) {
      return false;
    }

    AbstractCall<?, ?> call = pending.remove(rpcToRemove);
    if (call == null) {
      return false;
    }
    logger.atFiner().log(
        "%s completed with status %s and %d B payload", call, status, payload.size());
    enqueueCallUpdate(() -> call.handleUnaryCompleted(payload, status));
    return true;
  }

  private boolean handleStreamCompleted(PendingRpc rpc, Status status) {
    PendingRpc rpcToRemove = getRpc(rpc);
    if (rpcToRemove == null) {
      return false;
    }
    AbstractCall<?, ?> call = pending.remove(rpcToRemove);
    if (call == null) {
      return false;
    }
    logger.atFiner().log("%s completed with status %s", call, status);
    enqueueCallUpdate(() -> call.handleStreamCompleted(status));
    return true;
  }

  private boolean handleError(PendingRpc rpc, Status status) {
    AbstractCall<?, ?> call = pending.remove(rpc);
    if (call == null) {
      return false;
    }
    logger.atFiner().log("%s failed with error %s", call, status);
    enqueueCallUpdate(() -> call.handleError(status));
    return true;
  }

  public boolean processClientPacket(@Nullable Method method, RpcPacket packet) {
    synchronized (this) {
      Channel channel = channels.get(packet.getChannelId());
      if (channel == null) {
        logger.atWarning().log(
            "Received packet for unrecognized channel %d", packet.getChannelId());
        return false;
      }

      if (method == null) {
        logger.atFine().log("Ignoring packet for unknown service method");
        sendError(channel, packet, Status.NOT_FOUND);
        return true; // true since the packet was handled, even though it was invalid.
      }

      // Use 0 for call ID when IDs are disabled, which is equivalent to an unset ID in the packet.
      int callId = packets.callIdsEnabled() ? packet.getCallId() : 0;
      PendingRpc rpc = PendingRpc.create(channel, method, callId);
      if (!updateCall(packet, rpc)) {
        logger.atFine().log("Ignoring packet for %s, which isn't pending", rpc);
        sendError(channel, packet, Status.FAILED_PRECONDITION);
        return true;
      }
    }

    processCallUpdates();
    return true;
  }

  /** Returns true if the packet was forwarded to an active RPC call; false if no call was found. */
  private boolean updateCall(RpcPacket packet, PendingRpc rpc) {
    switch (packet.getType()) {
      case SERVER_ERROR: {
        Status status = decodeStatus(packet);
        return handleError(rpc, status);
      }
      case RESPONSE: {
        Status status = decodeStatus(packet);
        // Client streaming and unary RPCs include a payload with their response packet.
        if (rpc.method().isServerStreaming()) {
          return handleStreamCompleted(rpc, status);
        }
        return handleUnaryCompleted(rpc, packet.getPayload(), status);
      }
      case SERVER_STREAM:
        return handleNext(rpc, packet.getPayload());
      default:
        logger.atWarning().log(
            "%s received unexpected PacketType %d", rpc, packet.getType().getNumber());
    }

    return true;
  }

  private void sendError(Channel channel, RpcPacket packet, Status status) {
    try {
      channel.send(packets.error(packet, status));
    } catch (ChannelOutputException e) {
      logger.atWarning().withCause(e).log("Failed to send error packet");
    }
  }

  private static Status decodeStatus(RpcPacket packet) {
    Status status = Status.fromCode(packet.getStatus());
    if (status == null) {
      logger.atWarning().log(
          "Illegal status code %d in packet; using Status.UNKNOWN ", packet.getStatus());
      return Status.UNKNOWN;
    }
    return status;
  }

  @Nullable
  private Map.Entry<PendingRpc, AbstractCall<?, ?>> getRpcCallPair(PendingRpc rpc) {
    if (packets.callIdsEnabled()
        && (rpc.callId() == LEGACY_OPEN_CALL_ID || rpc.callId() == OPEN_CALL_ID)) {
      Optional<Map.Entry<PendingRpc, AbstractCall<?, ?>>> openCall =
          pending.entrySet()
              .stream()
              .filter(entry -> entry.getKey().equalsExceptCallId(rpc))
              .findFirst();

      if (openCall.isEmpty()) {
        return null;
      }

      PendingRpc newRpc = PendingRpc.withCallId(rpc, openCall.get().getKey().callId());
      return Map.entry(newRpc, openCall.get().getValue());
    }

    AbstractCall<?, ?> call = pending.get(rpc);
    return call == null ? null : Map.entry(rpc, pending.get(rpc));
  }

  /**
   * Gets the correct pending rpc.
   *
   * If call ids are not enabled, this should always be the original rpc, however, if call_ids are
   * enabled and the call_id is set to an open call id, then the rpc should be the first pending
   * call to the corresponding <channel, service, method> tuple. If there aren't any pending calls
   * that match, then null is returned.
   */
  @Nullable
  private PendingRpc getRpc(PendingRpc rpc) {
    Map.Entry<PendingRpc, AbstractCall<?, ?>> call = getRpcCallPair(rpc);
    return call == null ? null : call.getKey();
  }

  /**
   * Gets the correct pending call.
   *
   * If call ids are not enabled, this should always be the original call, however, if call_ids are
   * enabled and the call_id is set to an open call id, then the call should be the first pending
   * call to the corresponding <channel, service, method> tuple. If there aren't any pending calls
   * that match, then null is returned.
   */
  @Nullable
  private AbstractCall<?, ?> getCall(PendingRpc rpc) {
    Map.Entry<PendingRpc, AbstractCall<?, ?>> call = getRpcCallPair(rpc);
    return call == null ? null : call.getValue();
  }

  /** Gets the next available call id and increments internal count for next call. */
  private synchronized int getNewCallId() {
    int callId = nextCallId;
    nextCallId = (nextCallId + 1) % maxCallId;

    // Skip call_id `0` to avoid confusion with legacy servers which use call_id `0` as
    // an open call id or which do not provide call_id at all.
    if (nextCallId == 0) {
      nextCallId = FIRST_CALL_ID;
    }
    return callId;
  }

  /** Expose the Packets object for internal use by TestClient. */
  Packets getPackets() {
    return packets;
  }
}
