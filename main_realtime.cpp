// main_realtime.cpp - TFLite Micro Keyword Spotting with Real-time Microphone Input
// Implements streaming audio keyword detection using PortAudio for microphone capture
// Detects: yes/no/silence/unknown in real-time

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <atomic>
#include <thread>
#include <chrono>

// PortAudio for microphone input
#include "portaudio.h"

// Project headers
#include "model_settings.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/kernels/quantize.h"

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

// Debug flags (set to false to reduce output)
constexpr bool kDebugVad = false;
constexpr bool kDebugInference = false;

// Audio gain - use automatic gain normalization instead of fixed gain
constexpr bool kUseAutoGain = false;  // Disabled to match offline testing
constexpr float kTargetRmsEnergy = 3000.0f;
constexpr float kMaxGain = 5.0f;
constexpr float kMinGain = 1.0f;

// Streaming audio buffer configuration
constexpr int kAudioBufferSize = kAudioSampleFrequency * 2;  // 2 second circular buffer
constexpr float kDetectionThreshold = 0.5f;  // Confidence threshold for 0-9 digits
constexpr int kSuppressRepeatMs = 800;  // Suppress repeated detections

// VAD (Voice Activity Detection) parameters
constexpr int kVadFrameSizeMs = 30;  // VAD frame size in ms
constexpr int kVadFrameSize = kAudioSampleFrequency * kVadFrameSizeMs / 1000;  // 480 samples
constexpr float kVadEnergyThreshold = 300.0f;  // Minimum RMS energy for speech
constexpr float kVadSilenceThreshold = 100.0f;  // Below this is definitely silence
constexpr int kVadSpeechFramesRequired = 3;  // Need N consecutive speech frames to trigger
constexpr int kVadSilenceFramesRequired = 6;  // Need N consecutive silence frames to end utterance
constexpr int kMinUtteranceDurationMs = 250;  // Minimum utterance duration
constexpr int kMaxUtteranceDurationMs = 1200;  // Maximum utterance duration

// Arena configuration
constexpr size_t kArenaSize = 28584;
alignas(16) uint8_t tensor_arena[kArenaSize];

// Feature storage
using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

// Circular audio buffer for streaming
int16_t g_audio_buffer[kAudioBufferSize];
std::atomic<int> g_audio_write_index(0);
std::atomic<bool> g_running(true);

// VAD state machine
enum VadState {
    VAD_SILENCE,      // Waiting for speech
    VAD_SPEECH_START, // Speech detected, collecting
    VAD_SPEECH,       // In speech
    VAD_SPEECH_END    // Speech ended, ready for inference
};

struct VadContext {
    VadState state;
    int speech_frame_count;
    int silence_frame_count;
    int utterance_start_idx;
    int utterance_sample_count;
    float background_energy;
    bool utterance_ready;
};

VadContext g_vad = {VAD_SILENCE, 0, 0, 0, 0, 0.0f, false};

// Operation resolver type definitions
using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<6>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

// Register operations for MicroSpeech model
TfLiteStatus RegisterMicroSpeechOps(MicroSpeechOpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
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

// PortAudio callback for microphone input
static int AudioCallback(const void* inputBuffer, void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo* timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void* userData) {
    (void)outputBuffer;
    (void)timeInfo;
    (void)statusFlags;
    (void)userData;
    
    const int16_t* input = static_cast<const int16_t*>(inputBuffer);
    
    if (input != nullptr) {
        int write_idx = g_audio_write_index.load();
        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            g_audio_buffer[write_idx] = input[i];
            write_idx = (write_idx + 1) % kAudioBufferSize;
        }
        g_audio_write_index.store(write_idx);
    }
    
    return g_running.load() ? paContinue : paComplete;
}

// Get audio samples from circular buffer
void GetAudioSamples(int16_t* output, int sample_count) {
    int read_idx = (g_audio_write_index.load() - sample_count + kAudioBufferSize) % kAudioBufferSize;
    
    for (int i = 0; i < sample_count; i++) {
        output[i] = g_audio_buffer[read_idx];
        read_idx = (read_idx + 1) % kAudioBufferSize;
    }
}

// Get audio samples from specific position in circular buffer
void GetAudioSamplesFrom(int16_t* output, int start_idx, int sample_count) {
    int read_idx = (start_idx + kAudioBufferSize) % kAudioBufferSize;
    
    for (int i = 0; i < sample_count; i++) {
        output[i] = g_audio_buffer[read_idx];
        read_idx = (read_idx + 1) % kAudioBufferSize;
    }
}

// Calculate RMS energy of audio frame
float CalculateRmsEnergy(const int16_t* audio, int count) {
    int64_t sum_sq = 0;
    for (int i = 0; i < count; i++) {
        sum_sq += (int64_t)audio[i] * audio[i];
    }
    return sqrtf((float)sum_sq / count);
}

// Apply gain to audio buffer (with clipping protection)
void ApplyGain(int16_t* audio, int count, float gain) {
    for (int i = 0; i < count; i++) {
        int32_t sample = (int32_t)(audio[i] * gain);
        // Clip to int16 range
        if (sample > 32767) sample = 32767;
        if (sample < -32768) sample = -32768;
        audio[i] = (int16_t)sample;
    }
}

// Apply automatic gain to normalize audio level
void ApplyAutoGain(int16_t* audio, int count) {
    if (!kUseAutoGain) return;
    
    // Calculate current RMS
    float rms = CalculateRmsEnergy(audio, count);
    if (rms < 10.0f) return;  // Too quiet, don't amplify noise
    
    // Calculate gain needed
    float gain = kTargetRmsEnergy / rms;
    if (gain > kMaxGain) gain = kMaxGain;
    if (gain < kMinGain) gain = kMinGain;
    
    ApplyGain(audio, count, gain);
}

// Calculate zero crossing rate (helps distinguish speech from noise)
float CalculateZeroCrossingRate(const int16_t* audio, int count) {
    int crossings = 0;
    for (int i = 1; i < count; i++) {
        if ((audio[i] >= 0 && audio[i-1] < 0) || (audio[i] < 0 && audio[i-1] >= 0)) {
            crossings++;
        }
    }
    return (float)crossings / count;
}

// Update VAD state machine
void UpdateVad(VadContext* vad, const int16_t* frame, int frame_size) {
    float energy = CalculateRmsEnergy(frame, frame_size);
    float zcr = CalculateZeroCrossingRate(frame, frame_size);
    
    // Update background energy estimate (slow adaptation)
    if (vad->state == VAD_SILENCE) {
        vad->background_energy = vad->background_energy * 0.95f + energy * 0.05f;
    }
    
    // Dynamic threshold based on background noise
    float dynamic_threshold = fmaxf(kVadEnergyThreshold, vad->background_energy * 3.0f);
    
    // Speech detection: primarily based on energy, ZCR as secondary check
    // Relaxed ZCR range to allow more speech patterns
    bool is_speech = (energy > dynamic_threshold);
    bool is_silence = (energy < kVadSilenceThreshold);
    
    int current_idx = g_audio_write_index.load();
    
    switch (vad->state) {
        case VAD_SILENCE:
            if (is_speech) {
                vad->speech_frame_count++;
                if (vad->speech_frame_count >= kVadSpeechFramesRequired) {
                    vad->state = VAD_SPEECH_START;
                    // Mark utterance start (go back a bit to capture onset)
                    vad->utterance_start_idx = (current_idx - frame_size * (kVadSpeechFramesRequired + 2) + kAudioBufferSize) % kAudioBufferSize;
                    vad->utterance_sample_count = frame_size * (kVadSpeechFramesRequired + 2);
                    vad->silence_frame_count = 0;
                }
            } else {
                vad->speech_frame_count = 0;
            }
            break;
            
        case VAD_SPEECH_START:
        case VAD_SPEECH: {
            vad->state = VAD_SPEECH;
            vad->utterance_sample_count += frame_size;
            
            if (is_silence) {
                vad->silence_frame_count++;
                if (vad->silence_frame_count >= kVadSilenceFramesRequired) {
                    // Check minimum duration
                    int duration_ms = vad->utterance_sample_count * 1000 / kAudioSampleFrequency;
                    if (duration_ms >= kMinUtteranceDurationMs) {
                        vad->state = VAD_SPEECH_END;
                        vad->utterance_ready = true;
                    } else {
                        // Too short, reset
                        vad->state = VAD_SILENCE;
                        vad->speech_frame_count = 0;
                        vad->silence_frame_count = 0;
                    }
                }
            } else {
                vad->silence_frame_count = 0;
            }
            
            // Check maximum duration
            int max_samples = kMaxUtteranceDurationMs * kAudioSampleFrequency / 1000;
            if (vad->utterance_sample_count >= max_samples) {
                vad->state = VAD_SPEECH_END;
                vad->utterance_ready = true;
                vad->utterance_sample_count = max_samples;
            }
            break;
        }
            
        case VAD_SPEECH_END:
            // Wait for processing
            break;
    }
}

// Reset VAD after processing utterance
void ResetVad(VadContext* vad) {
    vad->state = VAD_SILENCE;
    vad->speech_frame_count = 0;
    vad->silence_frame_count = 0;
    vad->utterance_ready = false;
}

// Generate a single feature from audio window
TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
    TfLiteTensor* input = interpreter->input(0);
    if (input == nullptr) return kTfLiteError;
    
    if (audio_data_size != kAudioSampleDurationCount) {
        return kTfLiteError;
    }
    
    TfLiteTensor* output = interpreter->output(0);
    if (output == nullptr) return kTfLiteError;
    
    std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));
    
    if (interpreter->Invoke() != kTfLiteOk) {
        return kTfLiteError;
    }
    
    std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize, feature_output);
    
    return kTfLiteOk;
}

// Generate all features from audio data
TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output,
                              tflite::MicroInterpreter* preprocessor) {
    size_t remaining_samples = audio_data_size;
    size_t feature_index = 0;
    const int16_t* current_audio = audio_data;
    
    while (remaining_samples >= kAudioSampleDurationCount && feature_index < kFeatureCount) {
        if (GenerateSingleFeature(current_audio, kAudioSampleDurationCount,
                                  (*features_output)[feature_index], preprocessor) != kTfLiteOk) {
            return kTfLiteError;
        }
        
        feature_index++;
        current_audio += kAudioSampleStrideCount;
        remaining_samples -= kAudioSampleStrideCount;
    }
    
    return kTfLiteOk;
}

// Run MicroSpeech classifier on features
int RunClassifier(const Features& features, 
                  tflite::MicroInterpreter* classifier,
                  float* confidence) {
    TfLiteTensor* input = classifier->input(0);
    if (input == nullptr) return -1;
    
    TfLiteTensor* output = classifier->output(0);
    if (output == nullptr) return -1;
    
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    std::copy_n(&features[0][0], kFeatureElementCount, tflite::GetTensorData<int8_t>(input));
    
    if (classifier->Invoke() != kTfLiteOk) {
        return -1;
    }
    
    float category_predictions[kCategoryCount];
    for (int i = 0; i < kCategoryCount; i++) {
        category_predictions[i] = 
            (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) * output_scale;
    }
    
    int prediction_index = 
        std::distance(std::begin(category_predictions),
                      std::max_element(std::begin(category_predictions),
                                       std::end(category_predictions)));
    
    *confidence = category_predictions[prediction_index];
    
    // Debug: print all category scores
    if (kDebugInference) {
        printf("\n[INF] Scores: ");
        for (int i = 0; i < kCategoryCount; i++) {
            printf("%s:%.1f%% ", kCategoryLabels[i], category_predictions[i] * 100);
        }
        printf("\n[INF] Winner: %s (%.1f%%)\n", kCategoryLabels[prediction_index], *confidence * 100);
    }
    
    return prediction_index;
}

// Print audio level meter
void PrintAudioLevel(const int16_t* audio, int count, float* rms_energy) {
    *rms_energy = CalculateRmsEnergy(audio, count);
    int level = (int)(*rms_energy / 328);  // Scale to 0-100
    if (level > 100) level = 100;
    
    printf("\r[");
    for (int i = 0; i < 50; i++) {
        if (i < level / 2) printf("#");
        else printf(" ");
    }
    printf("] %3d%% E:%.0f ", level, *rms_energy);
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    printf("=== TFLite Micro Real-time Keyword Spotting ===\n");
    printf("Listening for: yes, no\n");
    printf("Press Ctrl+C to exit\n\n");
    
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return 1;
    }
    
    // List available input devices
    int numDevices = Pa_GetDeviceCount();
    printf("Available audio input devices:\n");
    int defaultInput = Pa_GetDefaultInputDevice();
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            printf("  [%d] %s%s\n", i, deviceInfo->name, 
                   (i == defaultInput) ? " (default)" : "");
        }
    }
    printf("\n");
    
    // Select input device
    int inputDevice = defaultInput;
    if (argc > 1) {
        inputDevice = atoi(argv[1]);
        printf("Using device %d\n", inputDevice);
    } else {
        printf("Using default input device. Run with device number to select different device.\n");
    }
    
    // Print device info
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputDevice);
    printf("\nDevice: %s\n", deviceInfo->name);
    printf("Default sample rate: %.0f Hz\n", deviceInfo->defaultSampleRate);
    printf("Max input channels: %d\n", deviceInfo->maxInputChannels);
    printf("Requested sample rate: %d Hz\n", kAudioSampleFrequency);
    
    // Check if device supports 16kHz
    PaStreamParameters testParams;
    testParams.device = inputDevice;
    testParams.channelCount = 1;
    testParams.sampleFormat = paInt16;
    testParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
    testParams.hostApiSpecificStreamInfo = nullptr;
    
    err = Pa_IsFormatSupported(&testParams, nullptr, kAudioSampleFrequency);
    if (err == paFormatIsSupported) {
        printf("16kHz format: SUPPORTED\n\n");
    } else {
        printf("16kHz format: NOT SUPPORTED (%s)\n", Pa_GetErrorText(err));
        printf("WARNING: Device may not support 16kHz, audio quality may be affected!\n\n");
    }
    
    // Initialize audio buffer
    memset(g_audio_buffer, 0, sizeof(g_audio_buffer));
    
    // Setup audio stream parameters
    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = 1;  // Mono
    inputParameters.sampleFormat = paInt16;
    inputParameters.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;
    
    // Open audio stream
    PaStream* stream;
    err = Pa_OpenStream(&stream,
                        &inputParameters,
                        nullptr,  // No output
                        kAudioSampleFrequency,  // 16000 Hz
                        256,  // Frames per buffer
                        paClipOff,
                        AudioCallback,
                        nullptr);
    
    if (err != paNoError) {
        printf("Failed to open audio stream: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        return 1;
    }
    
    // Get actual stream info
    const PaStreamInfo* streamInfo = Pa_GetStreamInfo(stream);
    if (streamInfo) {
        printf("Actual sample rate: %.0f Hz\n", streamInfo->sampleRate);
        printf("Input latency: %.1f ms\n\n", streamInfo->inputLatency * 1000);
    }
    
    // Load Audio Preprocessor model
    printf("Loading Audio Preprocessor model...\n");
    const tflite::Model* preprocessor_model = tflite::GetModel(__audio_preprocessor_int8_tflite);
    if (preprocessor_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: AudioPreprocessor model schema version mismatch\n");
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    
    AudioPreprocessorOpResolver preprocessor_resolver;
    if (RegisterAudioPreprocessorOps(preprocessor_resolver) != kTfLiteOk) {
        printf("ERROR: Failed to register AudioPreprocessor ops\n");
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    
    tflite::MicroInterpreter preprocessor(preprocessor_model, preprocessor_resolver, 
                                          tensor_arena, kArenaSize);
    if (preprocessor.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate AudioPreprocessor tensors\n");
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    printf("AudioPreprocessor ready (arena: %zu bytes)\n", preprocessor.arena_used_bytes());
    
    // Load MicroSpeech Classifier model - use separate arena
    alignas(16) uint8_t classifier_arena[kArenaSize];
    
    printf("Loading MicroSpeech Classifier model...\n");
    const tflite::Model* classifier_model = tflite::GetModel(micro_speech_quantized_tflite);
    if (classifier_model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: MicroSpeech model schema version mismatch\n");
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    
    MicroSpeechOpResolver classifier_resolver;
    if (RegisterMicroSpeechOps(classifier_resolver) != kTfLiteOk) {
        printf("ERROR: Failed to register MicroSpeech ops\n");
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    
    tflite::MicroInterpreter classifier(classifier_model, classifier_resolver, 
                                        classifier_arena, kArenaSize);
    if (classifier.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Failed to allocate MicroSpeech tensors\n");
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    printf("MicroSpeech Classifier ready (arena: %zu bytes)\n", classifier.arena_used_bytes());
    
    // Start audio stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        printf("Failed to start audio stream: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    
    printf("\n*** Listening... Speak digits 0-9 ***\n");
    printf("Say a digit clearly, then pause.\n\n");
    
    // Audio buffers
    int16_t audio_1sec[kAudioSampleFrequency];
    int16_t speech_buffer[kAudioSampleFrequency * 2];
    int speech_buffer_len = 0;
    
    // Pre-buffer to capture speech onset (ring buffer of last 500ms)
    const int kPreBufferSize = kAudioSampleFrequency / 2;  // 500ms
    int16_t pre_buffer[kPreBufferSize];
    int pre_buffer_pos = 0;
    bool pre_buffer_full = false;
    
    int last_detection = -1;
    auto last_detection_time = std::chrono::steady_clock::now();
    float background_energy = 0.0f;
    
    // Speech detection state
    bool speech_active = false;
    int speech_frames = 0;
    int silence_frames = 0;
    const int kSpeechFramesNeeded = 2;
    const int kSilenceFramesNeeded = 4;
    const int kFrameSize = 1600;  // 100ms at 16kHz
    
    // Initialize
    memset(pre_buffer, 0, sizeof(pre_buffer));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    GetAudioSamples(audio_1sec, kAudioSampleFrequency);
    background_energy = CalculateRmsEnergy(audio_1sec, kAudioSampleFrequency);
    printf("Background: %.0f\n\n", background_energy);
    
    // Main inference loop
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Get latest frame of audio
        int16_t frame[kFrameSize];
        GetAudioSamples(frame, kFrameSize);
        float energy = CalculateRmsEnergy(frame, kFrameSize);
        
        // Always update pre-buffer (ring buffer)
        for (int i = 0; i < kFrameSize; i++) {
            pre_buffer[pre_buffer_pos] = frame[i];
            pre_buffer_pos = (pre_buffer_pos + 1) % kPreBufferSize;
            if (pre_buffer_pos == 0) pre_buffer_full = true;
        }
        
        // Dynamic threshold
        float threshold = fmaxf(kVadEnergyThreshold, background_energy * 5.0f);
        bool is_speech = (energy > threshold);
        
        // Update background when quiet
        if (!is_speech && !speech_active) {
            background_energy = background_energy * 0.9f + energy * 0.1f;
        }
        
        // Simple audio level display
        int level = (int)(energy / 500);
        if (level > 30) level = 30;
        printf("\r[");
        for (int i = 0; i < 30; i++) printf(i < level ? "=" : " ");
        printf("] E:%4.0f T:%4.0f %s", energy, threshold, speech_active ? "* " : "  ");
        fflush(stdout);
        
        // State machine
        if (!speech_active) {
            if (is_speech) {
                speech_frames++;
                silence_frames = 0;
                if (speech_frames >= kSpeechFramesNeeded) {
                    speech_active = true;
                    speech_buffer_len = 0;
                    
                    // Copy pre-buffer first (to capture speech onset)
                    int pre_len = pre_buffer_full ? kPreBufferSize : pre_buffer_pos;
                    if (pre_buffer_full) {
                        int first_part = kPreBufferSize - pre_buffer_pos;
                        memcpy(speech_buffer, pre_buffer + pre_buffer_pos, first_part * sizeof(int16_t));
                        memcpy(speech_buffer + first_part, pre_buffer, pre_buffer_pos * sizeof(int16_t));
                    } else {
                        memcpy(speech_buffer, pre_buffer, pre_buffer_pos * sizeof(int16_t));
                    }
                    speech_buffer_len = pre_len;
                }
            } else {
                speech_frames = 0;
            }
        } else {
            // Recording speech - append frame to buffer
            if (speech_buffer_len + kFrameSize <= kAudioSampleFrequency * 2) {
                memcpy(speech_buffer + speech_buffer_len, frame, kFrameSize * sizeof(int16_t));
                speech_buffer_len += kFrameSize;
            }
            
            if (!is_speech) {
                silence_frames++;
                if (silence_frames >= kSilenceFramesNeeded) {
                    // Speech ended
                    speech_active = false;
                    speech_frames = 0;
                    silence_frames = 0;
                    
                    // Prepare 1 second audio for inference
                    int samples_to_use = speech_buffer_len;
                    if (samples_to_use > kAudioSampleFrequency) {
                        samples_to_use = kAudioSampleFrequency;
                    }
                    
                    memset(audio_1sec, 0, sizeof(audio_1sec));
                    memcpy(audio_1sec, speech_buffer, samples_to_use * sizeof(int16_t));
                    
                    // Apply automatic gain normalization before inference
                    ApplyAutoGain(audio_1sec, kAudioSampleFrequency);
                    
                    // Save first 10 samples for debugging (same data used for inference)
                    static int save_count = 0;
                    if (save_count < 10) {
                        char filename[64];
                        snprintf(filename, sizeof(filename), "test_%d.wav", save_count);
                        FILE* fp = fopen(filename, "wb");
                        if (fp) {
                            int data_size = kAudioSampleFrequency * 2;
                            int file_size = data_size + 36;
                            unsigned char header[44] = {
                                'R','I','F','F',
                                (unsigned char)(file_size & 0xff), (unsigned char)((file_size >> 8) & 0xff),
                                (unsigned char)((file_size >> 16) & 0xff), (unsigned char)((file_size >> 24) & 0xff),
                                'W','A','V','E', 'f','m','t',' ',
                                16, 0, 0, 0, 1, 0, 1, 0,
                                0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0,
                                2, 0, 16, 0,
                                'd','a','t','a',
                                (unsigned char)(data_size & 0xff), (unsigned char)((data_size >> 8) & 0xff),
                                (unsigned char)((data_size >> 16) & 0xff), (unsigned char)((data_size >> 24) & 0xff)
                            };
                            fwrite(header, 1, 44, fp);
                            fwrite(audio_1sec, 2, kAudioSampleFrequency, fp);
                            fclose(fp);
                            printf("[saved %s] ", filename);
                        }
                        save_count++;
                    }
                    
                    // Generate features and classify
                    // Reset preprocessor tensors to ensure consistent results
                    preprocessor.Reset();
                    
                    if (GenerateFeatures(audio_1sec, kAudioSampleFrequency, &g_features, &preprocessor) == kTfLiteOk) {
                        float confidence;
                        int prediction = RunClassifier(g_features, &classifier, &confidence);
                        
                        if (prediction >= 0) {
                            auto now = std::chrono::steady_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_detection_time).count();
                            
                            bool is_keyword = (prediction >= 2);
                            bool above_threshold = (confidence >= kDetectionThreshold);
                            bool not_suppressed = (prediction != last_detection || elapsed > kSuppressRepeatMs);
                            
                            // Always show result
                            printf("\r[%s %.0f%%]                         \n", kCategoryLabels[prediction], confidence * 100);
                            
                            if (is_keyword && above_threshold && not_suppressed) {
                                printf(">>> %s (%.0f%%) <<<\n", kCategoryLabels[prediction], confidence * 100);
                                last_detection = prediction;
                                last_detection_time = now;
                            }
                        }
                    }
                }
            } else {
                silence_frames = 0;
            }
        }
    }
    
    // Cleanup
    printf("\nShutting down...\n");
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    
    printf("Done.\n");
    return 0;
}
