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
#pragma once

#include <atomic>

#include "fsl_dma.h"
#include "fsl_inputmux.h"
#include "fsl_usart_dma.h"
#include "pw_bytes/byte_builder.h"
#include "pw_bytes/span.h"
#include "pw_clock_tree/clock_tree.h"
#include "pw_dma_mcuxpresso/dma.h"
#include "pw_status/status.h"
#include "pw_sync/interrupt_spin_lock.h"
#include "pw_sync/timed_thread_notification.h"
#include "pw_uart/uart.h"

namespace pw::uart {

class DmaUartMcuxpresso final : public Uart {
 public:
  // Configuration structure
  struct Config {
    USART_Type* usart_base;     // Base of USART control struct
    uint32_t baud_rate;         // Desired communication speed
    bool flow_control = false;  // Hardware flow control setting
    usart_parity_mode_t parity = kUSART_ParityDisabled;  // Parity setting
    usart_stop_bit_count_t stop_bits =
        kUSART_OneStopBit;                 // Number of stop bits to use
    dma::McuxpressoDmaChannel& rx_dma_ch;  // Receive DMA channel
    dma::McuxpressoDmaChannel& tx_dma_ch;  // Transmit DMA channel
    inputmux_signal_t rx_input_mux_dmac_ch_request_en;  // Rx input mux signal
    inputmux_signal_t tx_input_mux_dmac_ch_request_en;  // Tx input mux signal
    ByteSpan buffer;                                    // Receive ring buffer
    pw::clock_tree::Element*
        clock_tree_element{};  // Optional clock tree element
  };

  DmaUartMcuxpresso(const Config& config)
      : rx_data_{.ring_buffer = config.buffer},
        config_(config),
        clock_tree_element_(config.clock_tree_element),
        initialized_(false) {}

  ~DmaUartMcuxpresso();

  DmaUartMcuxpresso(const DmaUartMcuxpresso& other) = delete;

  DmaUartMcuxpresso& operator=(const DmaUartMcuxpresso& other) = delete;

 private:
  // Usart DMA TX data structure
  struct UsartDmaTxData {
    void Init();

    ConstByteSpan buffer;         // TX transaction buffer
    size_t tx_idx{};              // Position within TX transaction
    usart_transfer_t transfer{};  // USART TX transfer structure
    std::atomic_uint8_t
        busy{};  // Flag to prevent concurrent access to TX queue
    pw::sync::TimedThreadNotification
        notification;  // TX completion notification
  };

  // Usart DMA RX data structure
  struct UsartDmaRxData {
    void Init();

    ByteSpan ring_buffer;            // Receive ring buffer
    size_t ring_buffer_read_idx{};   // ring buffer reader index
    size_t ring_buffer_write_idx{};  // ring buffer writer index
    size_t data_received{};  // data received and acknowledged by completion
                             // callback
    size_t data_copied{};    // data copied out to receiver
    // completion callback will be executed when completion size decreases to 0
    // bytes
    size_t completion_size{};
    usart_transfer_t transfer{};  // USART RX transfer structure
    std::atomic_uint8_t
        busy{};  // Flag to prevent concurrent access to RX ring buffer
    pw::sync::TimedThreadNotification
        notification{};  // RX completion notification
  };

  // Since we are calling USART_TransferGetReceiveCountDMA we may only
  // transfer DMA_MAX_TRANSFER_COUNT - 1 bytes per DMA transfer.
  static constexpr size_t kUsartDmaMaxTransferCount =
      DMA_MAX_TRANSFER_COUNT - 1;

  // A reader may at most wait for 25% of the ring buffer size before data
  // needs to be copied out to the caller.
  static constexpr size_t kUsartRxRingBufferSplitCount = 4;

  // Should not be called while read/write is active.
  Status DoEnable(bool enable) override;
  Status DoSetBaudRate(uint32_t baud_rate) override;
  Status DoSetFlowControl(bool enable) override;

  // Will return an error if the internal ring buffer is overflowed.
  StatusWithSize DoTryReadFor(
      ByteSpan rx_buffer,
      size_t min_bytes,
      std::optional<chrono::SystemClock::duration> timeout) override;
  StatusWithSize DoTryWriteFor(
      ConstByteSpan tx_buffer,
      std::optional<chrono::SystemClock::duration> timeout) override;
  size_t DoConservativeReadAvailable() override;
  Status DoClearPendingReceiveBytes() override;
  Status DoFlushOutput() override;

  // Helper functions
  static IRQn_Type GetInterrupt(const DMA_Type* base);
  Status Init();
  void Deinit();
  void TriggerReadDma() PW_EXCLUSIVE_LOCKS_REQUIRED(interrupt_lock_);
  void TriggerWriteDma();

  StatusWithSize TransferGetReceiveDMACount()
      PW_LOCKS_EXCLUDED(interrupt_lock_);
  StatusWithSize TransferGetReceiveDMACountLockHeld()
      PW_EXCLUSIVE_LOCKS_REQUIRED(interrupt_lock_);
  size_t GetReceiveTransferRemainingBytes();
  static void TxRxCompletionCallback(USART_Type* base,
                                     usart_dma_handle_t* state,
                                     status_t status,
                                     void* param);
  Status WaitForReceiveBytes(
      size_t bytes_needed,
      std::optional<chrono::SystemClock::time_point> deadline);
  void CopyReceiveData(ByteBuilder& bb, size_t copy_size);

  pw::sync::InterruptSpinLock
      interrupt_lock_;  // Lock to synchronize with interrupt handler and to
                        // guarantee exclusive access to DMA control registers
  usart_dma_handle_t uart_dma_handle_{};  // USART DMA Handle
  UsartDmaTxData tx_data_;                // TX data
  UsartDmaRxData rx_data_;                // RX data

  Config config_;  // USART DMA configuration
  pw::clock_tree::OptionalElement clock_tree_element_;
  bool
      initialized_;  // Whether the USART and DMA channels have been initialized
  uint32_t flexcomm_clock_freq_{};
};

}  // namespace pw::uart
