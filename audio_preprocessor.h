/* Audio Preprocessor for Keyword Spotting
 * Converts raw audio to MFCC features for TensorFlow Lite Micro
 */

#ifndef AUDIO_PREPROCESSOR_H_
#define AUDIO_PREPROCESSOR_H_

#include <stdint.h>
#include "model_settings.h"

#ifdef __cplusplus
extern "C" {
#endif

// Audio preprocessor structure
typedef struct {
    float window[kFeatureDurationMs * kAudioSampleFrequency / 1000];
    float features[kFeatureElementCount];
    int initialized;
} AudioPreprocessor;

/**
 * Initialize audio preprocessor
 * @param preprocessor Pointer to preprocessor structure
 * @return 0 on success, -1 on error
 */
int InitializePreprocessor(AudioPreprocessor* preprocessor);

/**
 * Process audio data to extract features
 * @param preprocessor Pointer to initialized preprocessor
 * @param audio_data Raw audio samples (16-bit signed)
 * @param sample_count Number of audio samples
 * @param output_features Output features (int8 quantized)
 * @return 0 on success, -1 on error
 */
int ProcessAudioToFeatures(AudioPreprocessor* preprocessor,
                          const int16_t* audio_data,
                          uint32_t sample_count,
                          int8_t* output_features);

/**
 * Normalize audio data
 * @param audio_data Input/output audio samples
 * @param sample_count Number of samples
 */
void NormalizeAudio(int16_t* audio_data, uint32_t sample_count);

/**
 * Apply Hann window to audio frame
 * @param frame Audio frame data
 * @param frame_size Size of frame
 */
void ApplyHannWindow(float* frame, int frame_size);

/**
 * Compute simplified MFCC features
 * @param audio_frame Windowed audio frame
 * @param frame_size Size of frame
 * @param mfcc_output Output MFCC coefficients
 * @return 0 on success, -1 on error
 */
int ComputeMFCC(const float* audio_frame, int frame_size, float* mfcc_output);

/**
 * Quantize float features to int8
 * @param float_features Input float features
 * @param int8_features Output int8 features
 * @param count Number of features
 */
void QuantizeFeatures(const float* float_features, int8_t* int8_features, int count);

#ifdef __cplusplus
}
#endif

#endif  // AUDIO_PREPROCESSOR_H_