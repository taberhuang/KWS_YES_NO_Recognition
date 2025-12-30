/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

// Audio settings derived from the micro_speech model training
#define kAudioSampleFrequency 16000
#define kFeatureSize 40
#define kFeatureCount 49
#define kFeatureElementCount (kFeatureSize * kFeatureCount)
#define kFeatureStrideMs 20
#define kFeatureDurationMs 30

// Model output categories
#define kCategoryCount 12
extern const char* kCategoryLabels[kCategoryCount];

// Audio processing settings
#define kMaxAudioSampleSize 16000  // 1 second at 16kHz
#define kAudioChannels 1           // Mono audio
#define kAudioBitsPerSample 16     // 16-bit PCM

// TensorFlow Lite Micro settings
#define kTensorArenaSize (60 * 1024)  // 60KB tensor arena

#endif  // MODEL_SETTINGS_H_