# KWS 关键词识别系统 - TensorFlow Lite Micro

基于 TensorFlow Lite Micro 的嵌入式关键词识别（Keyword Spotting）系统，实现了完整的双模型流水线，可识别 "yes"、"no"、"unknown" 和 "silence" 四类语音指令。

## 📋 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [嵌入式平台移植指南](#嵌入式平台移植指南)
- [模型训练与自定义](#模型训练与自定义)
- [常见问题](#常见问题)

---

## 项目概述

本项目是一个完整的语音关键词识别系统，专为资源受限的嵌入式设备设计。系统采用双模型架构：

1. **音频预处理模型 (AudioPreprocessor)**：将原始音频转换为 MFCC 特征
2. **语音分类模型 (MicroSpeech)**：对特征进行分类，输出识别结果

### 核心优势

- 🎯 **轻量级设计**：总内存占用约 28KB，适合 MCU 部署
- ⚡ **实时处理**：支持流式音频处理
- 🔧 **易于移植**：基于标准 C/C++，无外部依赖
- 📦 **完整工具链**：包含构建脚本、测试用例

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        KWS 系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ WAV 文件 │───▶│ 音频预处理模型   │───▶│  语音分类模型    │   │
│  │ 16kHz    │    │ AudioPreprocessor│    │  MicroSpeech     │   │
│  │ Mono     │    │                  │    │                  │   │
│  └──────────┘    └─────────────────┘    └──────────────────┘   │
│       │                  │                       │              │
│       ▼                  ▼                       ▼              │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │ 16000个  │    │ 49帧 × 40特征   │    │ 4类概率分布      │   │
│  │ 采样点   │    │ INT8 量化       │    │ silence/unknown  │   │
│  │ (1秒)    │    │                  │    │ yes/no           │   │
│  └──────────┘    └─────────────────┘    └──────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流说明

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 音频读取 | WAV 文件 | 16000 个 INT16 采样点 | 1秒 16kHz 单声道音频 |
| 特征提取 | 480 采样点/帧 | 40 个 INT8 特征/帧 | 30ms 窗口，20ms 步进 |
| 分类推理 | 1960 个 INT8 特征 | 4 个浮点概率值 | 49帧 × 40特征 |

---

## 功能特性

### 已实现功能

- ✅ WAV 文件读取与格式验证
- ✅ 自动立体声转单声道
- ✅ 自动重采样至 16kHz
- ✅ MFCC 特征提取（基于 TFLite 信号处理库）
- ✅ INT8 量化推理
- ✅ 四分类输出（silence/unknown/yes/no）
- ✅ Windows MinGW/MSVC 编译支持
- ✅ CMake + Ninja 构建系统

### 信号处理算子

项目使用 TFLite Micro Signal 库提供的专用算子：

| 算子名称 | 功能 |
|----------|------|
| SignalWindow | 音频帧加窗 |
| SignalFFTAutoScale | 自动缩放 FFT |
| SignalRFFT | 实数 FFT |
| SignalEnergy | 能量计算 |
| SignalFilterBank | Mel 滤波器组 |
| SignalFilterBankSquareRoot | 平方根后处理 |
| SignalPCAN | 通道自动归一化 |

---

## 环境要求

### 硬件要求

- **开发机**：Windows 10/11 或 Linux
- **内存**：至少 4GB RAM
- **存储**：约 500MB（含 tflite-micro 源码）

### 软件要求

| 工具 | 版本要求 | 用途 |
|------|----------|------|
| CMake | 3.10+ | 构建配置 |
| Ninja | 最新版 | 构建执行 |
| GCC/G++ | 15.2.0+ (MinGW64) | 编译器 |
| Python | 3.8+ | 模型训练（可选） |
| TensorFlow | 2.x | 模型训练（可选） |

### Windows 环境配置

```cmd
# 推荐使用 MSYS2 安装 MinGW64
# 下载地址: https://www.msys2.org/

# 在 MSYS2 终端中安装工具链
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-ninja
```

---

## 快速开始

### 1. 克隆项目

```cmd
git clone <repository_url>
cd kws
```

### 2. 编译项目

```cmd
# 方式一：使用构建脚本（推荐）
.\build_cmake.bat

# 方式二：手动 CMake 构建
mkdir build_cmake
cd build_cmake
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

### 3. 运行测试

```cmd
# 测试 "yes" 识别
.\build_cmake\kws_yesno.exe yes_1000ms.wav

# 测试 "no" 识别
.\build_cmake\kws_yesno.exe no_1000ms.wav

# 测试自定义音频（需 16kHz 单声道 WAV）
.\build_cmake\kws_yesno.exe your_audio.wav
```

### 4. 预期输出

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
```

---

## 项目结构

```
kws/
├── CMakeLists.txt                          # CMake 构建配置
├── build_cmake.bat                         # Windows 构建脚本
├── main.cpp                                # 主程序（双模型流水线）
│
├── wav_file_reader.c                       # WAV 文件读取实现
├── wav_reader.h                            # WAV 读取接口
│
├── audio_preprocessor.c                    # 音频预处理实现（备用）
├── audio_preprocessor.h                    # 音频预处理接口
│
├── model_settings.h                        # 模型参数配置
│
├── audio_preprocessor_int8.tflite          # 音频预处理模型（TFLite 格式）
├── audio_preprocessor_int8_model_data.c    # 模型数据（C 数组）
├── audio_preprocessor_int8_model_data.h    # 模型数据头文件
│
├── micro_speech_quantized.tflite           # 语音分类模型（TFLite 格式）
├── micro_speech_quantized_model_data.c     # 模型数据（C 数组）
├── micro_speech_quantized_model_data.h     # 模型数据头文件
│
├── yes_1000ms.wav                          # 测试音频（yes）
├── no_1000ms.wav                           # 测试音频（no）
│
└── tflite-micro/                           # TensorFlow Lite Micro 源码
    ├── tensorflow/lite/micro/              # TFLM 核心实现
    ├── signal/                             # 信号处理库
    └── third_party/                        # 第三方依赖
        ├── flatbuffers/                    # FlatBuffers 序列化库
        ├── gemmlowp/                       # 低精度矩阵运算库
        ├── kissfft/                        # FFT 实现
        └── ruy/                            # 矩阵乘法库
```

---

## 技术细节

### 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| kAudioSampleFrequency | 16000 | 采样率 (Hz) |
| kFeatureSize | 40 | 每帧特征数 |
| kFeatureCount | 49 | 总帧数 |
| kFeatureDurationMs | 30 | 帧长度 (ms) |
| kFeatureStrideMs | 20 | 帧步进 (ms) |
| kCategoryCount | 4 | 分类数量 |

### 内存占用

| 组件 | 内存 (bytes) |
|------|-------------|
| Tensor Arena 总大小 | 28,584 |
| AudioPreprocessor 使用 | ~10,568 |
| MicroSpeech 使用 | ~7,496 |
| 特征缓冲区 | 1,960 |

### 音频格式要求

- **采样率**：16 kHz（其他采样率会自动重采样）
- **声道数**：单声道（立体声会自动转换）
- **位深度**：16-bit PCM
- **格式**：WAV (RIFF)

---

## 嵌入式平台移植指南

### 移植概述

本项目基于 TensorFlow Lite Micro，可移植到多种嵌入式平台：

| 平台类型 | 示例 | 最低要求 |
|----------|------|----------|
| ARM Cortex-M | STM32F4/F7/H7, nRF52 | 64KB RAM, 256KB Flash |
| ESP32 | ESP32, ESP32-S3 | 内置 PSRAM 推荐 |
| RISC-V | ESP32-C3, GD32VF103 | 64KB RAM |
| Arduino | Arduino Nano 33 BLE | 256KB RAM |

### 移植步骤

#### 步骤 1：准备目标平台工具链

```bash
# 以 ARM Cortex-M 为例（使用 arm-none-eabi-gcc）
# 下载地址: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm
```

#### 步骤 2：修改 CMakeLists.txt

```cmake
# 添加交叉编译工具链配置
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)

# 添加 MCU 特定编译选项
set(MCU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MCU_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MCU_FLAGS}")
```

#### 步骤 3：实现平台相关接口

需要实现以下平台相关函数：

```cpp
// 1. 调试日志输出（可选）
// 文件: tflite-micro/tensorflow/lite/micro/debug_log.cc
void DebugLog(const char* s) {
    // 替换为目标平台的串口输出
    UART_SendString(s);
}

// 2. 系统时间（用于性能分析，可选）
// 文件: tflite-micro/tensorflow/lite/micro/micro_time.cc
uint32_t tflite::GetCurrentTimeTicks() {
    return HAL_GetTick();  // 使用 HAL 库示例
}
```

#### 步骤 4：替换音频输入

```cpp
// 替换 WAV 文件读取为麦克风输入
// 示例：使用 I2S 接口读取 PDM 麦克风数据

#include "i2s_driver.h"

int16_t audio_buffer[16000];  // 1秒音频缓冲

void CaptureAudio() {
    I2S_Read(audio_buffer, 16000);
}

// 在 main 函数中调用
CaptureAudio();
GenerateFeatures(audio_buffer, 16000, &g_features);
RunMicroSpeechClassifier(g_features);
```

#### 步骤 5：优化内存使用

```cpp
// 减小 Arena 大小（根据实际需求调整）
constexpr size_t kArenaSize = 20000;  // 尝试更小的值

// 使用静态分配避免堆碎片
static uint8_t tensor_arena[kArenaSize] __attribute__((aligned(16)));
```

### 平台特定优化

#### ARM Cortex-M (CMSIS-NN)

```cmake
# 启用 CMSIS-NN 优化内核
add_definitions(-DCMSIS_NN)
include_directories(${CMSIS_PATH}/CMSIS/NN/Include)
```

#### ESP32 (ESP-IDF)

```cmake
# 使用 ESP-IDF 组件系统
idf_component_register(
    SRCS "main.cpp" "wav_file_reader.c" ...
    INCLUDE_DIRS "." "tflite-micro" ...
)
```

---

## 模型训练与自定义

### 训练新的关键词模型（如识别 "hello"）

#### 概述流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 1. 数据准备 │───▶│ 2. 模型训练 │───▶│ 3. 模型转换 │───▶│ 4. 部署测试 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

#### 步骤 1：准备训练数据

**方式一：使用 Google Speech Commands 数据集**

```bash
# 下载官方数据集
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C data/

# 数据集包含 35 个关键词，每个约 2000+ 样本
# 包括: yes, no, up, down, left, right, on, off, stop, go 等
```

**方式二：录制自定义关键词（如 "hello"）**

```bash
# 创建数据目录结构
mkdir -p data/hello data/unknown data/silence data/_background_noise_

# 录制要求：
# - 格式: 16kHz, 16-bit, 单声道 WAV
# - 时长: 1秒
# - 数量: 每个关键词至少 1000+ 样本
# - 多样性: 不同说话人、不同环境、不同语速

# 使用 Python 脚本批量录制
python scripts/record_audio.py --keyword hello --count 1000
```

**数据增强（提高模型鲁棒性）**

```python
import librosa
import numpy as np

def augment_audio(audio, sr=16000):
    augmented = []
    
    # 1. 时间拉伸
    for rate in [0.9, 1.1]:
        augmented.append(librosa.effects.time_stretch(audio, rate=rate))
    
    # 2. 音调变换
    for steps in [-2, 2]:
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps))
    
    # 3. 添加噪声
    noise = np.random.randn(len(audio)) * 0.005
    augmented.append(audio + noise)
    
    # 4. 音量变化
    for gain in [0.8, 1.2]:
        augmented.append(audio * gain)
    
    return augmented
```

#### 步骤 2：训练模型

**安装依赖**

```bash
pip install tensorflow==2.15.0
pip install tensorflow-model-optimization
```

**训练脚本**

```python
# train_micro_speech.py
import tensorflow as tf
from tensorflow.keras import layers, models

# 模型参数（与 model_settings.h 保持一致）
FEATURE_SIZE = 40
FEATURE_COUNT = 49
NUM_CLASSES = 4  # silence, unknown, hello, other

# 定义模型架构（与原 micro_speech 一致）
def create_model():
    model = models.Sequential([
        layers.Input(shape=(FEATURE_COUNT, FEATURE_SIZE, 1)),
        
        # 第一层卷积
        layers.DepthwiseConv2D(
            kernel_size=(8, 10),
            strides=(2, 2),
            padding='same',
            depth_multiplier=1
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# 编译模型
model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练
model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

#### 步骤 3：模型量化与转换

```python
# convert_model.py
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model('best_model.h5')

# 创建转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 配置 INT8 量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 提供代表性数据集（用于量化校准）
def representative_dataset():
    for data in calibration_data:
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset

# 转换模型
tflite_model = converter.convert()

# 保存模型
with open('micro_speech_hello.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"模型大小: {len(tflite_model)} bytes")
```

#### 步骤 4：生成 C 数组

```bash
# 使用 xxd 工具转换
xxd -i micro_speech_hello.tflite > micro_speech_hello_model_data.c

# 或使用 Python 脚本
python -c "
import sys
data = open('micro_speech_hello.tflite', 'rb').read()
print('const int8_t micro_speech_hello_tflite[] = {')
print(','.join(f'0x{b:02x}' for b in data))
print('};')
print(f'const size_t micro_speech_hello_tflite_len = {len(data)};')
" > micro_speech_hello_model_data.c
```

#### 步骤 5：更新代码

```cpp
// 1. 修改 model_settings.h
#define kCategoryCount 4
extern const char* kCategoryLabels[kCategoryCount];
// 在 .c 文件中定义:
// const char* kCategoryLabels[] = {"silence", "unknown", "hello", "other"};

// 2. 替换模型数据头文件
#include "micro_speech_hello_model_data.h"

// 3. 更新模型加载代码
const tflite::Model* model = tflite::GetModel(micro_speech_hello_tflite);
```

### 音频预处理模型训练

音频预处理模型（AudioPreprocessor）通常不需要重新训练，它是一个固定的信号处理流水线。如果需要修改特征提取参数：

```python
# 修改 MFCC 参数
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 30
FRAME_STRIDE_MS = 20
NUM_MEL_BINS = 40
NUM_MFCC = 40
FFT_SIZE = 512
```

---

## 常见问题

### Q1: 编译报错 "Ninja not found"

```cmd
# 解决方案：安装 Ninja
# Windows (使用 MSYS2):
pacman -S mingw-w64-x86_64-ninja

# 或下载预编译版本:
# https://github.com/ninja-build/ninja/releases
```

### Q2: 识别准确率低

可能原因及解决方案：

1. **音频格式不正确**
   - 确保是 16kHz 单声道 WAV
   - 使用 Audacity 等工具转换格式

2. **音频质量差**
   - 减少背景噪声
   - 确保说话清晰

3. **音频时长不足**
   - 关键词应在 1 秒音频的中间位置
   - 前后留有适当静音

### Q3: 内存不足（嵌入式平台）

```cpp
// 尝试减小 Arena 大小
constexpr size_t kArenaSize = 20000;

// 使用更激进的量化
// 在模型转换时使用 INT8 全量化

// 考虑使用更小的模型架构
```

### Q4: 如何添加更多关键词？

1. 收集新关键词的训练数据
2. 修改 `kCategoryCount` 和 `kCategoryLabels`
3. 重新训练分类模型
4. 注意：更多类别可能需要更大的模型

### Q5: 如何实现实时流式识别？

```cpp
// 使用环形缓冲区实现流式处理
class StreamingRecognizer {
    int16_t audio_buffer[32000];  // 2秒缓冲
    int write_index = 0;
    
public:
    void AddSamples(int16_t* samples, int count) {
        // 添加新采样到缓冲区
        for (int i = 0; i < count; i++) {
            audio_buffer[write_index] = samples[i];
            write_index = (write_index + 1) % 32000;
        }
    }
    
    const char* Recognize() {
        // 每隔一定时间运行一次识别
        // 使用最近 1 秒的音频
        GenerateFeatures(...);
        return RunClassifier(...);
    }
};
```

---

## 参考资源

- [TensorFlow Lite Micro 官方文档](https://www.tensorflow.org/lite/microcontrollers)
- [Speech Commands 数据集](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [TFLite Micro GitHub](https://github.com/tensorflow/tflite-micro)
- [Micro Speech 示例](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech)

---

## 许可证

本项目基于 TensorFlow Lite Micro，遵循 Apache 2.0 许可证。

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/new-keyword`)
3. 提交更改 (`git commit -am 'Add new keyword support'`)
4. 推送分支 (`git push origin feature/new-keyword`)
5. 创建 Pull Request
