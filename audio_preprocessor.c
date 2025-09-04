/* Audio Preprocessor Implementation
 * Simplified MFCC feature extraction for keyword spotting
 */

#include "audio_preprocessor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Mel filter bank parameters
#define kNumMelFilters 40
#define kFFTSize 512
#define kNumFFTBins (kFFTSize / 2 + 1)

// Quantization parameters (matched to micro_speech model)
#define kFeatureScale 0.0039215686f  // 1.0f / 255.0f
#define kFeatureOffset -128

int InitializePreprocessor(AudioPreprocessor* preprocessor) {
    if (!preprocessor) {
        return -1;
    }
    
    memset(preprocessor, 0, sizeof(AudioPreprocessor));
    preprocessor->initialized = 1;
    
    printf("Audio preprocessor initialized\n");
    printf("  Frame size: %d ms\n", kFeatureDurationMs);
    printf("  Frame stride: %d ms\n", kFeatureStrideMs);
    printf("  Feature size: %d\n", kFeatureSize);
    printf("  Feature count: %d\n", kFeatureCount);
    
    return 0;
}

void NormalizeAudio(int16_t* audio_data, uint32_t sample_count) {
    if (!audio_data || sample_count == 0) {
        return;
    }
    
    // Find max absolute value
    int16_t max_val = 0;
    for (uint32_t i = 0; i < sample_count; i++) {
        int16_t abs_val = audio_data[i] < 0 ? -audio_data[i] : audio_data[i];
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    // Normalize if needed
    if (max_val > 0) {
        float scale = 32767.0f / max_val;
        for (uint32_t i = 0; i < sample_count; i++) {
            audio_data[i] = (int16_t)(audio_data[i] * scale);
        }
    }
}

void ApplyHannWindow(float* frame, int frame_size) {
    for (int i = 0; i < frame_size; i++) {
        float window_val = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frame_size - 1)));
        frame[i] *= window_val;
    }
}

// Simplified FFT implementation for power spectrum
static void ComputePowerSpectrum(const float* input, int input_size, float* output) {
    // This is a simplified implementation
    // In a real implementation, you would use a proper FFT library
    
    int output_size = kNumFFTBins;
    
    // Initialize output
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
    }
    
    // Simplified frequency domain conversion
    // This approximates the power spectrum using a sliding window approach
    float window_size = (float)input_size / output_size;
    
    for (int i = 0; i < output_size; i++) {
        int start_idx = (int)(i * window_size);
        int end_idx = (int)((i + 1) * window_size);
        if (end_idx > input_size) end_idx = input_size;
        
        float power = 0.0f;
        for (int j = start_idx; j < end_idx; j++) {
            power += input[j] * input[j];
        }
        
        output[i] = power / (end_idx - start_idx);
    }
}

// Convert frequency to mel scale
static float FreqToMel(float freq) {
    return 2595.0f * log10f(1.0f + freq / 700.0f);
}

// Convert mel scale to frequency
static float MelToFreq(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Apply mel filter bank
static void ApplyMelFilterBank(const float* power_spectrum, float* mel_output) {
    float sample_rate = kAudioSampleFrequency;
    float nyquist = sample_rate / 2.0f;
    
    // Mel scale boundaries
    float mel_low = FreqToMel(0.0f);
    float mel_high = FreqToMel(nyquist);
    float mel_step = (mel_high - mel_low) / (kNumMelFilters + 1);
    
    // Initialize output
    for (int i = 0; i < kNumMelFilters; i++) {
        mel_output[i] = 0.0f;
    }
    
    // Apply triangular filters
    for (int i = 0; i < kNumMelFilters; i++) {
        float mel_center = mel_low + (i + 1) * mel_step;
        float mel_left = mel_low + i * mel_step;
        float mel_right = mel_low + (i + 2) * mel_step;
        
        float freq_center = MelToFreq(mel_center);
        float freq_left = MelToFreq(mel_left);
        float freq_right = MelToFreq(mel_right);
        
        // Map frequencies to FFT bins
        int bin_left = (int)(freq_left * kNumFFTBins / nyquist);
        int bin_center = (int)(freq_center * kNumFFTBins / nyquist);
        int bin_right = (int)(freq_right * kNumFFTBins / nyquist);
        
        // Clamp to valid range
        if (bin_left < 0) bin_left = 0;
        if (bin_right >= kNumFFTBins) bin_right = kNumFFTBins - 1;
        
        // Left slope
        for (int j = bin_left; j <= bin_center && j < kNumFFTBins; j++) {
            if (bin_center > bin_left) {
                float weight = (float)(j - bin_left) / (bin_center - bin_left);
                mel_output[i] += power_spectrum[j] * weight;
            }
        }
        
        // Right slope
        for (int j = bin_center; j <= bin_right && j < kNumFFTBins; j++) {
            if (bin_right > bin_center) {
                float weight = (float)(bin_right - j) / (bin_right - bin_center);
                mel_output[i] += power_spectrum[j] * weight;
            }
        }
    }
}

int ComputeMFCC(const float* audio_frame, int frame_size, float* mfcc_output) {
    if (!audio_frame || !mfcc_output || frame_size <= 0) {
        return -1;
    }
    
    // Allocate temporary buffers
    float* power_spectrum = (float*)malloc(kNumFFTBins * sizeof(float));
    float* mel_energies = (float*)malloc(kNumMelFilters * sizeof(float));
    
    if (!power_spectrum || !mel_energies) {
        free(power_spectrum);
        free(mel_energies);
        return -1;
    }
    
    // Compute power spectrum
    ComputePowerSpectrum(audio_frame, frame_size, power_spectrum);
    
    // Apply mel filter bank
    ApplyMelFilterBank(power_spectrum, mel_energies);
    
    // Apply log and DCT (simplified)
    for (int i = 0; i < kFeatureSize; i++) {
        if (i < kNumMelFilters) {
            // Add small epsilon to avoid log(0)
            float energy = mel_energies[i] + 1e-10f;
            mfcc_output[i] = logf(energy);
        } else {
            mfcc_output[i] = 0.0f;
        }
    }
    
    // Cleanup
    free(power_spectrum);
    free(mel_energies);
    
    return 0;
}

void QuantizeFeatures(const float* float_features, int8_t* int8_features, int count) {
    for (int i = 0; i < count; i++) {
        // Apply quantization: int8 = (float - offset) / scale
        float scaled = float_features[i] / kFeatureScale + kFeatureOffset;
        
        // Clamp to int8 range
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        
        int8_features[i] = (int8_t)scaled;
    }
}

int ProcessAudioToFeatures(AudioPreprocessor* preprocessor,
                          const int16_t* audio_data,
                          uint32_t sample_count,
                          int8_t* output_features) {
    if (!preprocessor || !preprocessor->initialized || 
        !audio_data || !output_features) {
        printf("Error: Invalid parameters for feature extraction\n");
        return -1;
    }
    
    printf("Processing audio to features...\n");
    printf("  Input samples: %d\n", sample_count);
    
    // Frame parameters
    int frame_length = kFeatureDurationMs * kAudioSampleFrequency / 1000;  // 30ms
    int frame_stride = kFeatureStrideMs * kAudioSampleFrequency / 1000;   // 20ms
    
    printf("  Frame length: %d samples\n", frame_length);
    printf("  Frame stride: %d samples\n", frame_stride);
    
    // Process frames
    for (int frame_idx = 0; frame_idx < kFeatureCount; frame_idx++) {
        int start_sample = frame_idx * frame_stride;
        
        // Check if we have enough samples
        if (start_sample + frame_length > sample_count) {
            // Pad with zeros if needed
            printf("Warning: Padding frame %d with zeros\n", frame_idx);
            
            // Fill remaining frames with zeros
            for (int remaining = frame_idx; remaining < kFeatureCount; remaining++) {
                for (int i = 0; i < kFeatureSize; i++) {
                    output_features[remaining * kFeatureSize + i] = kFeatureOffset;
                }
            }
            break;
        }
        
        // Copy frame to float buffer
        for (int i = 0; i < frame_length; i++) {
            preprocessor->window[i] = (float)audio_data[start_sample + i] / 32768.0f;
        }
        
        // Apply window function
        ApplyHannWindow(preprocessor->window, frame_length);
        
        // Compute MFCC features
        float frame_features[kFeatureSize];
        if (ComputeMFCC(preprocessor->window, frame_length, frame_features) != 0) {
            printf("Error: MFCC computation failed for frame %d\n", frame_idx);
            return -1;
        }
        
        // Quantize and store features
        QuantizeFeatures(frame_features, 
                        &output_features[frame_idx * kFeatureSize], 
                        kFeatureSize);
    }
    
    printf("Feature extraction completed: %d frames x %d features\n", 
           kFeatureCount, kFeatureSize);
    
    return 0;
}