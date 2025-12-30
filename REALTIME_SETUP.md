# 实时麦克风关键词识别设置指南

## 概述

这个版本将原来的 WAV 文件输入改为实时麦克风流媒体输入，可以持续监听并自动识别 "yes" 和 "no" 关键词。

## 依赖安装

### 方法 1: 使用 MSYS2 (推荐)

1. 安装 MSYS2: https://www.msys2.org/

2. 打开 MSYS2 MinGW64 终端，安装 PortAudio:
```bash
pacman -S mingw-w64-x86_64-portaudio
```

3. 确保 MinGW64 在 PATH 中:
```
C:\msys64\mingw64\bin
```

### 方法 2: 手动安装 PortAudio

1. 下载 PortAudio: http://www.portaudio.com/download.html

2. 编译或下载预编译的库

3. 设置环境变量 `PORTAUDIO_DIR` 指向安装目录

## 编译

```batch
build_realtime.bat
```

或手动:

```batch
mkdir build_realtime
cd build_realtime
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -f ../CMakeLists_realtime.txt ..
cmake --build . -j
```

## 运行

```batch
REM 使用默认麦克风
kws_realtime.exe

REM 指定麦克风设备号
kws_realtime.exe 1
```

## 使用说明

1. 运行程序后会显示可用的音频输入设备列表
2. 程序开始监听后，会显示实时音量条
3. 对着麦克风说 "yes" 或 "no"
4. 当检测到关键词且置信度超过阈值时，会显示检测结果
5. 按 Ctrl+C 退出

## 参数调整

在 `main_realtime.cpp` 中可以调整以下参数:

```cpp
constexpr int kInferenceIntervalMs = 200;     // 推理间隔 (毫秒)
constexpr float kDetectionThreshold = 0.6f;   // 检测置信度阈值 (0-1)
constexpr int kSuppressRepeatMs = 1000;       // 重复检测抑制时间 (毫秒)
```

## 故障排除

### 找不到 portaudio.h
确保 PortAudio 已正确安装，并且 include 路径正确设置。

### 没有检测到麦克风
- 检查系统音频设置中麦克风是否启用
- 尝试指定不同的设备号运行

### 识别不准确
- 确保环境安静
- 说话清晰，音量适中
- 可以降低 `kDetectionThreshold` 值

## 文件说明

- `main_realtime.cpp` - 实时识别主程序
- `CMakeLists_realtime.txt` - CMake 配置文件
- `build_realtime.bat` - Windows 构建脚本
- `model_settings.c` - 模型标签定义
