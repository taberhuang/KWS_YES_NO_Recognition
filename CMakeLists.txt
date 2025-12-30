cmake_minimum_required(VERSION 3.10)
project(kws_0_9 VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard - match working config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compilation options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# MinGW64 compiler setup (Windows)
if(WIN32)
    # Compiler paths are now set by build script
    # CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set via command line
    message(STATUS "Using MinGW64 compilers from build script")
endif()

# Add critical definitions that the working config has
add_definitions(-DFLATBUFFERS_LOCALE_INDEPENDENT=0)

# Include directories based on build.bat configuration
include_directories(
    .
    tflite-micro
    tflite-micro/third_party/flatbuffers/include
    tflite-micro/third_party/gemmlowp
    tflite-micro/third_party/kissfft
    tflite-micro/third_party/kissfft/tools
    tflite-micro/third_party/ruy
    tflite-micro/signal/src
)

# Main source files
set(MAIN_SOURCES
    wav_file_reader.c
    audio_preprocessor_int8_model_data.c
    micro_speech_quantized_model_data.c
    main.cpp
)

# Critical: Add microfrontend C sources (essential for correct recognition!)
set(MICROFRONTEND_C_SOURCES
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/window.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/window_util.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/filterbank.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/log_scale.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/log_lut.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/frontend.c
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/frontend_util.c
)

# TFLite Micro Core
set(TFLM_CORE_SOURCES
    # Use standard debug_log
    tflite-micro/tensorflow/lite/micro/debug_log.cc
    tflite-micro/tensorflow/lite/micro/micro_log.cc
    tflite-micro/tensorflow/lite/micro/micro_time.cc
    tflite-micro/tensorflow/lite/micro/system_setup.cc
    tflite-micro/tensorflow/lite/micro/micro_utils.cc
    tflite-micro/tensorflow/lite/micro/flatbuffer_utils.cc
    tflite-micro/tensorflow/lite/micro/memory_helpers.cc
    tflite-micro/tensorflow/lite/micro/micro_allocator.cc
    tflite-micro/tensorflow/lite/micro/micro_allocation_info.cc
    tflite-micro/tensorflow/lite/micro/micro_interpreter.cc
    tflite-micro/tensorflow/lite/micro/micro_context.cc
    tflite-micro/tensorflow/lite/micro/micro_interpreter_graph.cc
    tflite-micro/tensorflow/lite/micro/micro_interpreter_context.cc
    tflite-micro/tensorflow/lite/micro/micro_op_resolver.cc
    tflite-micro/tensorflow/lite/micro/micro_resource_variable.cc
    tflite-micro/tensorflow/lite/micro/micro_profiler.cc

    # Arena Allocators
    tflite-micro/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc
    tflite-micro/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc
    tflite-micro/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc

    # Memory Planners
    tflite-micro/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
    tflite-micro/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc

    # Tensor Utils
    tflite-micro/tensorflow/lite/micro/kernels/micro_tensor_utils.cc
    tflite-micro/tensorflow/lite/kernels/internal/portable_tensor_utils.cc
    tflite-micro/tensorflow/lite/core/c/common.cc
    tflite-micro/tensorflow/lite/kernels/internal/quantization_util.cc
    tflite-micro/tensorflow/lite/kernels/internal/common.cc
    tflite-micro/tensorflow/lite/core/api/flatbuffer_conversions.cc
    tflite-micro/tensorflow/lite/core/api/tensor_utils.cc
    tflite-micro/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc
    tflite-micro/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
    tflite-micro/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc
    tflite-micro/tensorflow/lite/kernels/kernel_util.cc
    tflite-micro/tensorflow/compiler/mlir/lite/schema/schema_utils.cc
    tflite-micro/tensorflow/lite/array.cc
    tflite-micro/tensorflow/lite/kernels/internal/tensor_ctypes.cc
)

# Essential Kernels
set(TFLM_KERNEL_SOURCES
    tflite-micro/tensorflow/lite/micro/kernels/kernel_util.cc
    tflite-micro/tensorflow/lite/micro/kernels/reshape.cc
    tflite-micro/tensorflow/lite/micro/kernels/cast.cc
    tflite-micro/tensorflow/lite/micro/kernels/strided_slice.cc
    tflite-micro/tensorflow/lite/micro/kernels/concatenation.cc
    tflite-micro/tensorflow/lite/micro/kernels/mul.cc
    tflite-micro/tensorflow/lite/micro/kernels/add.cc
    tflite-micro/tensorflow/lite/micro/kernels/div.cc
    tflite-micro/tensorflow/lite/micro/kernels/fully_connected.cc
    tflite-micro/tensorflow/lite/micro/kernels/depthwise_conv.cc
    tflite-micro/tensorflow/lite/micro/kernels/conv.cc
    tflite-micro/tensorflow/lite/micro/kernels/softmax.cc
    tflite-micro/tensorflow/lite/micro/kernels/quantize.cc
    tflite-micro/tensorflow/lite/micro/kernels/quantize_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/elementwise.cc

    # Kernel Common Functions
    tflite-micro/tensorflow/lite/micro/kernels/mul_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/add_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/strided_slice_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/reshape_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/fully_connected_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/depthwise_conv_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/softmax_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/conv_common.cc
    tflite-micro/tensorflow/lite/micro/kernels/maximum_minimum.cc
)

# Signal processing from working config
set(TFLM_SIGNAL_KERNEL_SOURCES
    tflite-micro/signal/micro/kernels/window.cc
    tflite-micro/signal/micro/kernels/fft_auto_scale_kernel.cc
    tflite-micro/signal/micro/kernels/rfft.cc
    tflite-micro/signal/micro/kernels/energy.cc
    tflite-micro/signal/micro/kernels/filter_bank.cc
    tflite-micro/signal/micro/kernels/filter_bank_square_root.cc
    tflite-micro/signal/micro/kernels/filter_bank_square_root_common.cc
    tflite-micro/signal/micro/kernels/filter_bank_spectral_subtraction.cc
    tflite-micro/signal/micro/kernels/filter_bank_log.cc
    tflite-micro/signal/micro/kernels/pcan.cc
    tflite-micro/signal/micro/kernels/fft_auto_scale_common.cc
)

# Signal processing source files - matching working config exactly
set(TFLM_SIGNAL_SOURCES
    tflite-micro/signal/src/window.cc
    tflite-micro/signal/src/fft_auto_scale.cc
    tflite-micro/signal/src/irfft_int16.cc
    tflite-micro/signal/src/irfft_int32.cc
    tflite-micro/signal/src/irfft_float.cc
    tflite-micro/signal/src/rfft_int16.cc
    tflite-micro/signal/src/rfft_int32.cc
    tflite-micro/signal/src/rfft_float.cc
    tflite-micro/signal/src/energy.cc
    tflite-micro/signal/src/filter_bank.cc
    tflite-micro/signal/src/filter_bank_square_root.cc
    tflite-micro/signal/src/filter_bank_spectral_subtraction.cc
    tflite-micro/signal/src/filter_bank_log.cc
    tflite-micro/signal/src/log.cc
    tflite-micro/signal/src/kiss_fft_wrappers/kiss_fft_int16.cc
    tflite-micro/signal/src/msb_32.cc
    tflite-micro/signal/src/max_abs.cc
    tflite-micro/signal/src/square_root_32.cc
    tflite-micro/signal/src/square_root_64.cc
    tflite-micro/signal/src/pcan_argc_fixed.cc
    tflite-micro/signal/src/msb_64.cc
)

# Critical: Add microfrontend FFT support for working recognition
set(MICROFRONTEND_FFT_SOURCES
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/fft.cc
    tflite-micro/tensorflow/lite/experimental/microfrontend/lib/fft_util.cc
)

# Flatbuffers support
set(FLATBUFFERS_SOURCES
    tflite-micro/third_party/flatbuffers/src/util.cpp
)

# Combine all source files - includes all new source file sets
set(ALL_SOURCES
    ${MAIN_SOURCES}
    ${MICROFRONTEND_C_SOURCES}     # Critical for recognition!
    ${FLATBUFFERS_SOURCES}
    ${TFLM_CORE_SOURCES}
    ${TFLM_KERNEL_SOURCES}
    ${TFLM_SIGNAL_KERNEL_SOURCES}
    ${TFLM_SIGNAL_SOURCES}
    ${MICROFRONTEND_FFT_SOURCES}   # Critical FFT support!
)


# Create executable
add_executable(${PROJECT_NAME} ${ALL_SOURCES})

# Set compile definitions (if needed)
# target_compile_definitions(${PROJECT_NAME} PRIVATE ...)

# Set link options (Windows MinGW full static linking)
if(MINGW)
    target_link_options(${PROJECT_NAME} PRIVATE
        -static
        -static-libgcc
        -static-libstdc++
    )
endif()

# Install target (optional)
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Output build information
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Source files: ${ALL_SOURCES}")

# Display include directories
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
    message(STATUS "Include dir: ${dir}")
endforeach()
