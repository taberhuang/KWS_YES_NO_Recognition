// main.cpp - TFLite Micro Keyword Spotting with Dual Model Pipeline
// Implements audio keyword detection using two models:
// 1. Audio Preprocessor: Converts raw audio to MFCC features
// 2. MicroSpeech Classifier: Classifies features into keywords (yes/no/silence/unknown)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <iterator>

// Project headers
#include "wav_reader.h"
#include "model_settings.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"

// Model data headers
#include "audio_preprocessor_int8_model_data.h"
#include "micro_speech_quantized_model_data.h"

// Signal processing operation registrations
namespace tflite { 
namespace tflm_signal {
TFLMRegistration* Register_PCAN();
TFLMRegistration* Register_FILTER_BANK();
TFLMRegistration* Register_FILTER_BANK_LOG();
TFLMRegistration* Register_FILTER_BANK_SQUARE_ROOT();
TFLMRegistration* Register_FILTER_BANK_SPECTRAL_SUBTRACTION();
TFLMRegistration* Register_WINDOW();
TFLMRegistration* Register_FFT_AUTO_SCALE();
TFLMRegistration* Register_RFFT();
TFLMRegistration* Register_ENERGY();
}
}

// Audio sample parameters from model_settings.h
constexpr int kAudioSampleDurationCount = kFeatureDurationMs * kAudioSampleFrequency / 1000;  // 480
constexpr int kAudioSampleStrideCount = kFeatureStrideMs * kAudioSampleFrequency / 1000;      // 320

// Arena configuration (size determined from micro_speech_test.cc)
constexpr size_t kArenaSize = 28584;
alignas(16) uint8_t tensor_arena[kArenaSize];

// Feature storage
using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

// Operation resolver type definitions
using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

// Register operations for MicroSpeech model
TfLiteStatus RegisterMicroSpeechOps(MicroSpeechOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    return kTfLiteOk;
}

// Register operations for AudioPreprocessor model
TfLiteStatus RegisterAudioPreprocessorOps(AudioPreprocessorOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
    TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
    TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
    TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
    TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
    TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
    return kTfLiteOk;
}

// Generate a single feature from audio window
TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
    TfLiteTensor* input = interpreter->input(0);
    if (input == nullptr) return kTfLiteError;
    
    if (audio_data_size != kAudioSampleDurationCount) {
        printf("ERROR: Expected %d audio samples, got %d\n", kAudioSampleDurationCount, audio_data_size);
        return kTfLiteError;
    }
    
    TfLiteTensor* output = interpreter->output(0);
    if (output == nullptr) return kTfLiteError;
    
    // Copy audio data to input tensor
    std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: Feature generation inference failed\n");
        return kTfLiteError;
    }
    
    // Copy output features
    std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize, feature_output);
    
    return kTfLiteOk;
}

// Generate all features from audio data
TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
    printf("\n=== Generating Features with Audio Preprocessor ===\n");
    
    // Load the audio preprocessor model
    const tflite::Model* model = tflite::GetModel(__audio_preprocessor_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: AudioPreprocessor model schema version mismatch\n");
        return kTfLiteError;
    }
    
    AudioPreprocessorOpResolver op_resolver;
    if (RegisterAudioPreprocessorOps(op_resolver) != kTfLiteOk) {
        printf("ERROR: Failed to register AudioPreprocessor ops\n");
        return kTfLiteError;
    }
    
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate AudioPreprocessor tensors\n");
        return kTfLiteError;
    }
    
    printf("AudioPreprocessor arena used: %zu bytes\n", interpreter.arena_used_bytes());
    
    // Process audio with sliding window
    size_t remaining_samples = audio_data_size;
    size_t feature_index = 0;
    const int16_t* current_audio = audio_data;
    
    printf("Processing audio: %zu samples, window size: %d, stride: %d\n", 
           audio_data_size, kAudioSampleDurationCount, kAudioSampleStrideCount);
    
    while (remaining_samples >= kAudioSampleDurationCount && feature_index < kFeatureCount) {

        
        if (GenerateSingleFeature(current_audio, kAudioSampleDurationCount,
                                  (*features_output)[feature_index], &interpreter) != kTfLiteOk) {
            printf("ERROR: Failed to generate feature %zu\n", feature_index);
            return kTfLiteError;
        }
        
        feature_index++;
        current_audio += kAudioSampleStrideCount;
        remaining_samples -= kAudioSampleStrideCount;
    }
    
    printf("Generated %zu features\n", feature_index);
    return kTfLiteOk;
}

// Run MicroSpeech classifier on features
TfLiteStatus RunMicroSpeechClassifier(const Features& features, const char* expected_label = nullptr) {
    printf("\n=== Running MicroSpeech Classifier ===\n");
    
    // Load the MicroSpeech model
    const tflite::Model* model = tflite::GetModel(micro_speech_quantized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: MicroSpeech model schema version mismatch\n");
        return kTfLiteError;
    }
    
    MicroSpeechOpResolver op_resolver;
    if (RegisterMicroSpeechOps(op_resolver) != kTfLiteOk) {
        printf("ERROR: Failed to register MicroSpeech ops\n");
        return kTfLiteError;
    }
    
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate MicroSpeech tensors\n");
        return kTfLiteError;
    }
    
    printf("MicroSpeech arena used: %zu bytes\n", interpreter.arena_used_bytes());
    
    TfLiteTensor* input = interpreter.input(0);
    if (input == nullptr) return kTfLiteError;
    
    if (kFeatureElementCount != input->dims->data[input->dims->size - 1]) {
        printf("ERROR: Input size mismatch. Expected %d, got %d\n",
               kFeatureElementCount, input->dims->data[input->dims->size - 1]);
        return kTfLiteError;
    }
    
    TfLiteTensor* output = interpreter.output(0);
    if (output == nullptr) return kTfLiteError;
    
    if (kCategoryCount != output->dims->data[output->dims->size - 1]) {
        printf("ERROR: Output size mismatch. Expected %d, got %d\n",
               kCategoryCount, output->dims->data[output->dims->size - 1]);
        return kTfLiteError;
    }
    
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    // Copy features to input tensor
    std::copy_n(&features[0][0], kFeatureElementCount, tflite::GetTensorData<int8_t>(input));
    
    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("ERROR: MicroSpeech inference failed\n");
        return kTfLiteError;
    }
    
    // Process output - dequantize predictions
    float category_predictions[kCategoryCount];
    printf("\nMicroSpeech predictions:\n");
    for (int i = 0; i < kCategoryCount; i++) {
        category_predictions[i] = 
            (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) * output_scale;
        printf("  %.4f %s\n", static_cast<double>(category_predictions[i]), kCategoryLabels[i]);
    }
    
    // Find the class with highest probability
    int prediction_index = 
        std::distance(std::begin(category_predictions),
                      std::max_element(std::begin(category_predictions),
                                       std::end(category_predictions)));
    
    printf("\n>>> FINAL RESULT: %s (confidence: %.4f)\n", 
           kCategoryLabels[prediction_index], 
           static_cast<double>(category_predictions[prediction_index]));
    
    if (expected_label != nullptr) {
        if (strcmp(expected_label, kCategoryLabels[prediction_index]) == 0) {
            printf("Prediction matches expected label!\n");
        } else {
            printf("Prediction does not match expected label (%s)\n", expected_label);
        }
    }
    
    return kTfLiteOk;
}

// Main function
int main(int argc, char* argv[]) {
    printf("=== TFLite Micro Keyword Spotting (Dual Model Pipeline) ===\n\n");
    
    if (argc != 2) {
        printf("Usage: %s <wav_file>\n", argv[0]);
        return 1;
    }
    
    const char* wav_file = argv[1];
    
    // Step 1: Read WAV file
    printf("Step 1: Read WAV file\n");
    uint32_t sample_count, sample_rate;
    int16_t* audio_data = ReadWavFile(wav_file, &sample_count, &sample_rate);
    if (audio_data == nullptr) {
        printf("Failed to read WAV file: %s\n", wav_file);
        return 1;
    }
    
    printf("Successfully loaded WAV file:\n");
    printf("  Sample rate: %u Hz\n", sample_rate);
    printf("  Sample count: %u\n", sample_count);
    printf("  Duration: %.2f seconds\n", (float)sample_count / sample_rate);
    
    // Step 2: Generate features using AudioPreprocessor
    printf("\nStep 2: Generate features using AudioPreprocessor\n");
    if (GenerateFeatures(audio_data, sample_count, &g_features) != kTfLiteOk) {
        printf("ERROR: Failed to generate features\n");
        free(audio_data);
        return 1;
    }
    
    // Step 3: Run classification using MicroSpeech
    printf("\nStep 3: Run classification using MicroSpeech\n");
    const char* expected_label = nullptr;
    
    // Simple filename-based expectation check
    if (strstr(wav_file, "zero") != nullptr) {
        expected_label = "zero";
    } else if (strstr(wav_file, "one") != nullptr) {
        expected_label = "one";
    } else if (strstr(wav_file, "two") != nullptr) {
        expected_label = "two";
    } else if (strstr(wav_file, "three") != nullptr) {
        expected_label = "three";
    } else if (strstr(wav_file, "four") != nullptr) {
        expected_label = "four";
    } else if (strstr(wav_file, "five") != nullptr) {
        expected_label = "five";
    } else if (strstr(wav_file, "six") != nullptr) {
        expected_label = "six";
    } else if (strstr(wav_file, "seven") != nullptr) {
        expected_label = "seven";
    } else if (strstr(wav_file, "eight") != nullptr) {
        expected_label = "eight";
    } else if (strstr(wav_file, "nine") != nullptr) {
        expected_label = "nine";
    } else if (strstr(wav_file, "silence") != nullptr) {
        expected_label = "silence";
    }
    
    if (expected_label != nullptr) {
        printf("Expected label from filename: %s\n", expected_label);
    }
    
    if (RunMicroSpeechClassifier(g_features, expected_label) != kTfLiteOk) {
        printf("ERROR: Failed to run classification\n");
        free(audio_data);
        return 1;
    }
    
    // Cleanup
    free(audio_data);
    printf("\n=== Processing Complete ===\n");
    return 0;
}