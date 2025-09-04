# KWS Yes/No Recognition - TensorFlow Lite Micro

A Windows-native keyword spotting (KWS) application built with TensorFlow Lite Micro, implementing a complete dual-model pipeline for accurate recognition of "yes", "no", "unknown", and "silence" audio commands.

## 🎯 Project Highlights

- **Dual-Model Architecture**: AudioPreprocessor (feature extraction) + MicroSpeech (classification)
- **Modern Build System**: CMake + Ninja with automated build scripts
- **Complete TFLM Integration**: Based on official micro_speech example implementation
- **Automated Testing**: Built-in test cases with audio validation
- **Cross-Platform Support**: Windows MinGW and MSVC compilation support

## 📋 System Requirements

- Windows 10/11 (MinGW64 or Visual Studio 2022)
- CMake 3.10+
- Ninja build system
- MinGW64 GCC 15.2.0+ or MSVC v143+
- 16kHz mono WAV audio files for testing

## 📁 Project Structure

```
kws/
├── CMakeLists.txt                          # Main CMake build configuration
├── build_cmake.bat                         # Windows build script
├── main.cpp                                # Main application (dual-model pipeline)
├── wav_file_reader.c                       # WAV file reading implementation
├── wav_reader.h                            # WAV reader header
├── audio_preprocessor.h                    # Audio preprocessor interface
├── audio_preprocessor.c                    # Audio preprocessor implementation
├── model_settings.h                        # Model configuration constants
├── audio_preprocessor_int8_model_data.h/c  # AudioPreprocessor model data
├── micro_speech_quantized_model_data.h/c   # MicroSpeech classifier model data
├── yes_1000ms.wav                          # Test audio file (yes command)
├── no_1000ms.wav                           # Test audio file (no command)
├── build_cmake/                            # Build output directory
│   └── kws_yesno.exe                       # Generated executable
└── tflite-micro/                           # TensorFlow Lite Micro source code
    ├── tensorflow/lite/micro/              # Core TFLM implementation
    ├── signal/                             # Signal processing library
    ├── third_party/                        # Third-party dependencies
    │   ├── flatbuffers/                    # FlatBuffers library
    │   ├── gemmlowp/                       # Low-precision matrix library
    │   ├── kissfft/                        # FFT implementation
    │   └── ruy/                            # Matrix multiplication library
    └── tensorflow/lite/experimental/microfrontend/  # Audio frontend library
```

## 🚀 Quick Start

### 1. Build the Application

#### Windows (Recommended)
```cmd
# Build and test automatically
.\build_cmake.bat

# Build only
.\build_cmake.bat build

# Run tests only (after building)
.\build_cmake.bat run

# Clean build
.\build_cmake.bat clean
```

#### Manual CMake Build
```cmd
mkdir build_cmake
cd build_cmake
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

### 2. Run Keyword Recognition

```cmd
# Test with provided audio files
.\build_cmake\kws_yesno.exe yes_1000ms.wav
.\build_cmake\kws_yesno.exe no_1000ms.wav

# Test with your own 16kHz mono WAV files
.\build_cmake\kws_yesno.exe your_audio.wav
```

## 🔧 Build System

### Toolchain Requirements

- **MinGW64**: Primary development toolchain
  - Location: `C:\msys64\mingw64\`
  - GCC version: 15.2.0+
  - Includes: CMake, Ninja, GCC, G++

- **MSVC** (Alternative): Visual Studio 2022 Build Tools
  - Requires: MSVC v143 compiler toolset
  - Windows SDK 10.0+

### CMake Configuration Features

- **C++ Standard**: C++17 (required for TFLM compatibility)
- **C Standard**: C11
- **Build Type**: Release (optimized for performance)
- **Static Linking**: Full static linking on Windows
- **Compiler Flags**: `-O2 -Wall` for optimization and warnings

### Build Script Options

The `build_cmake.bat` script supports multiple commands:

```cmd
build_cmake.bat [command]

Commands:
  (empty)     - Build and run tests (default)
  build       - Build only
  rebuild     - Clean and rebuild
  build_run   - Build and run tests
  rebuild_run - Clean, rebuild and run tests
  run         - Run tests only
  clean       - Clean build directory
```

## 🎵 Audio Requirements

### Input Format
- **Sample Rate**: 16 kHz
- **Channels**: Mono (single channel)
- **Format**: 16-bit PCM WAV
- **Duration**: Any length (processed in 1-second windows)

### Processing Pipeline
1. **Audio Loading**: WAV file reading with format validation
2. **Feature Extraction**: 30ms windows with 20ms stride
3. **Spectrogram Generation**: MFCC-like features (49 frames × 40 features)
4. **Classification**: Four-class output (silence, unknown, yes, no)

## 🧠 Model Architecture

### AudioPreprocessor Model
- **Input**: 480 samples (30ms at 16kHz)
- **Output**: 40 INT8 features per frame
- **Purpose**: Convert raw audio to mel-scale spectrogram features
- **Implementation**: TensorFlow Lite Micro with signal processing ops

### MicroSpeech Classifier
- **Input**: 49×40 INT8 feature matrix (1960 elements)
- **Output**: 4-class probability distribution
- **Classes**: [silence, unknown, yes, no]
- **Architecture**: Convolutional neural network optimized for microcontrollers

## 📊 Expected Results

### Successful Recognition Example
```
=== TFLite Micro Keyword Spotting (Dual Model Pipeline) ===

Step 1: Read WAV file
Successfully loaded WAV file:
  Sample rate: 16000 Hz
  Sample count: 16000
  Duration: 1.00 seconds

Step 2: Generate features using AudioPreprocessor
AudioPreprocessor arena used: 10568 bytes
Generated 49 features

Step 3: Run classification using MicroSpeech
MicroSpeech arena used: 7496 bytes

MicroSpeech predictions:
  0.0000 silence
  0.0000 unknown
  0.9961 yes
  0.0000 no

>>> FINAL RESULT: yes (confidence: 0.9961)
✅ Prediction matches expected label!
```

## 🛠️ Development

### Key Source Files

- **`main.cpp`**: Implements the complete dual-model pipeline
- **`wav_file_reader.c`**: Handles WAV file parsing and audio data extraction
- **`audio_preprocessor.c`**: Audio preprocessing utilities (currently unused)
- **Model Data Files**: Contains the compiled TensorFlow Lite models as C arrays

### Signal Processing Operations

The project includes custom TFLM signal processing operations:
- **SignalWindow**: Windowing function for audio frames
- **SignalFFTAutoScale**: FFT with automatic scaling
- **SignalRFFT**: Real-valued FFT
- **SignalEnergy**: Energy computation
- **SignalFilterBank**: Mel-scale filter bank
- **SignalFilterBankSquareRoot**: Square root post-processing
- **SignalPCAN**: Per-channel auto-normalization

### Memory Usage

- **Total Arena Size**: 28,584 bytes
- **AudioPreprocessor**: ~10,568 bytes
- **MicroSpeech**: ~7,496 bytes
- **Remaining**: Available for tensor operations

## 🔍 Troubleshooting

### Common Build Issues

1. **Ninja not found**: Install Ninja build system and add to PATH
2. **CMake not found**: Install CMake 3.10+ and add to PATH
3. **Compiler not found**: Ensure MinGW64 is installed and in PATH
4. **Build failures**: Try `build_cmake.bat clean` followed by `build_cmake.bat rebuild`

### Runtime Issues

1. **WAV file not found**: Ensure audio files are in the correct directory
2. **Incorrect recognition**: Verify audio format (16kHz mono WAV)
3. **Low confidence**: Check audio quality and background noise levels

## 📝 License

This project is based on TensorFlow Lite Micro and follows the Apache 2.0 License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the provided audio files
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your build environment matches the requirements
3. Test with the provided audio samples first
4. Open an issue with detailed error messages and system information