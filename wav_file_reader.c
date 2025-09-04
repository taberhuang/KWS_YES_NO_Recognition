/* wav_file_reader.c - Robust WAV File Reader Implementation
 * Handles various WAV formats and edge cases for keyword spotting
 */

 #include "wav_reader.h"
 #include "model_settings.h"
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 
 // Category labels for model output
 const char* kCategoryLabels[kCategoryCount] = {
     "silence",
     "unknown", 
     "yes",
     "no"
 };
 
 // Helper function to read a 4-byte chunk ID
 static int ReadChunkId(FILE* file, char* chunk_id) {
     return fread(chunk_id, 4, 1, file) == 1;
 }
 
 // Helper function to read a 4-byte size value
 static int ReadUint32(FILE* file, uint32_t* value) {
     return fread(value, 4, 1, file) == 1;
 }
 
 // Helper function to skip bytes in file
 static int SkipBytes(FILE* file, uint32_t bytes) {
     return fseek(file, bytes, SEEK_CUR) == 0;
 }
 
 // Find a specific chunk in the WAV file
 static long FindChunk(FILE* file, const char* target_chunk, uint32_t* chunk_size) {
     char chunk_id[4];
     uint32_t size;
     
     while (ReadChunkId(file, chunk_id)) {
         if (!ReadUint32(file, &size)) {
             return -1;
         }
         
         if (memcmp(chunk_id, target_chunk, 4) == 0) {
             *chunk_size = size;
             return ftell(file);  // Return position of chunk data
         }
         
         // Skip this chunk (align to word boundary if odd size)
         uint32_t skip_size = (size + 1) & ~1;
         if (!SkipBytes(file, skip_size)) {
             return -1;
         }
     }
     
     return -1;  // Chunk not found
 }
 
 // Normalize audio data to prevent clipping
 static void NormalizeAudioData(int16_t* data, uint32_t sample_count) {
     // Find peak absolute value
     int32_t peak = 0;
     for (uint32_t i = 0; i < sample_count; i++) {
         int32_t abs_val = abs((int)data[i]);
         if (abs_val > peak) {
             peak = abs_val;
         }
     }
     
     if (peak == 0) {
         printf("Warning: Audio is silent (all zeros)\n");
         return;
     }
     
     // Check if normalization is needed
     const int16_t target_peak = 16384;  // 50% of int16 range
     if (peak > target_peak) {
         float scale = (float)target_peak / peak;
         printf("Normalizing audio: peak=%d, scale=%.3f\n", peak, scale);
         
         for (uint32_t i = 0; i < sample_count; i++) {
             data[i] = (int16_t)(data[i] * scale);
         }
     } else {
         printf("Audio peak: %d (no normalization needed)\n", peak);
     }
 }
 
 // Convert stereo to mono by averaging channels
 static void ConvertStereoToMono(int16_t* stereo_data, int16_t* mono_data, 
                                 uint32_t sample_count) {
     for (uint32_t i = 0; i < sample_count; i++) {
         int32_t left = stereo_data[i * 2];
         int32_t right = stereo_data[i * 2 + 1];
         mono_data[i] = (int16_t)((left + right) / 2);
     }
 }
 
 // Resample audio if necessary (simple linear interpolation)
 static int16_t* ResampleAudio(int16_t* input, uint32_t input_samples, 
                               uint32_t input_rate, uint32_t output_rate,
                               uint32_t* output_samples) {
     if (input_rate == output_rate) {
         *output_samples = input_samples;
         return input;  // No resampling needed
     }
     
     *output_samples = (uint32_t)((uint64_t)input_samples * output_rate / input_rate);
     int16_t* output = (int16_t*)malloc(*output_samples * sizeof(int16_t));
     if (!output) {
         return NULL;
     }
     
     float ratio = (float)input_rate / output_rate;
     
     for (uint32_t i = 0; i < *output_samples; i++) {
         float src_idx = i * ratio;
         uint32_t idx0 = (uint32_t)src_idx;
         uint32_t idx1 = idx0 + 1;
         
         if (idx1 >= input_samples) {
             output[i] = input[input_samples - 1];
         } else {
             float frac = src_idx - idx0;
             output[i] = (int16_t)(input[idx0] * (1.0f - frac) + 
                                  input[idx1] * frac);
         }
     }
     
     printf("Resampled from %u Hz to %u Hz (%u -> %u samples)\n",
            input_rate, output_rate, input_samples, *output_samples);
     
     return output;
 }
 
 int ValidateWavFile(const char* filename) {
     FILE* file = fopen(filename, "rb");
     if (!file) {
         printf("Error: Cannot open file %s\n", filename);
         return 0;
     }
     
     // Read RIFF header
     char riff[4];
     uint32_t file_size;
     char wave[4];
     
     if (!ReadChunkId(file, riff) || 
         !ReadUint32(file, &file_size) ||
         !ReadChunkId(file, wave)) {
         printf("Error: Cannot read file header\n");
         fclose(file);
         return 0;
     }
     
     // Validate signatures
     if (memcmp(riff, "RIFF", 4) != 0) {
         printf("Error: Not a RIFF file\n");
         fclose(file);
         return 0;
     }
     
     if (memcmp(wave, "WAVE", 4) != 0) {
         printf("Error: Not a WAVE file\n");
         fclose(file);
         return 0;
     }
     
     // Find and validate fmt chunk
     uint32_t fmt_size;
     long fmt_pos = FindChunk(file, "fmt ", &fmt_size);
     if (fmt_pos < 0) {
         printf("Error: No fmt chunk found\n");
         fclose(file);
         return 0;
     }
     
     fseek(file, fmt_pos, SEEK_SET);
     
     uint16_t audio_format, channels, bits_per_sample;
     uint32_t sample_rate;
     
     fread(&audio_format, 2, 1, file);
     fread(&channels, 2, 1, file);
     fread(&sample_rate, 4, 1, file);
     fseek(file, 6, SEEK_CUR);  // Skip byte_rate and block_align
     fread(&bits_per_sample, 2, 1, file);
     
     // Check format
     if (audio_format != 1) {
         printf("Error: Only PCM format supported (got format %d)\n", audio_format);
         fclose(file);
         return 0;
     }
     
     if (bits_per_sample != 16) {
         printf("Error: Only 16-bit audio supported (got %d bits)\n", bits_per_sample);
         fclose(file);
         return 0;
     }
     
     // Check for data chunk
     uint32_t data_size;
     if (FindChunk(file, "data", &data_size) < 0) {
         printf("Error: No data chunk found\n");
         fclose(file);
         return 0;
     }
     
     fclose(file);
     
     printf("WAV file validated:\n");
     printf("  Format: PCM\n");
     printf("  Channels: %d\n", channels);
     printf("  Sample rate: %u Hz\n", sample_rate);
     printf("  Bits per sample: %d\n", bits_per_sample);
     printf("  Data size: %u bytes\n", data_size);
     
     return 1;
 }
 
 int16_t* ReadWavFile(const char* filename, 
                      uint32_t* sample_count,
                      uint32_t* sample_rate) {
     if (!filename || !sample_count || !sample_rate) {
         printf("Error: Invalid parameters\n");
         return NULL;
     }
     
     FILE* file = fopen(filename, "rb");
     if (!file) {
         printf("Error: Cannot open file %s\n", filename);
         return NULL;
     }
     
     // Skip RIFF header
     fseek(file, 12, SEEK_SET);
     
     // Find and read fmt chunk
     uint32_t fmt_size;
     long fmt_pos = FindChunk(file, "fmt ", &fmt_size);
     if (fmt_pos < 0) {
         printf("Error: No fmt chunk found\n");
         fclose(file);
         return NULL;
     }
     
     fseek(file, fmt_pos, SEEK_SET);
     
     // Read format information
     uint16_t audio_format, channels, bits_per_sample;
     uint32_t file_sample_rate, byte_rate;
     uint16_t block_align;
     
     fread(&audio_format, 2, 1, file);
     fread(&channels, 2, 1, file);
     fread(&file_sample_rate, 4, 1, file);
     fread(&byte_rate, 4, 1, file);
     fread(&block_align, 2, 1, file);
     fread(&bits_per_sample, 2, 1, file);
     
     // Validate format
     if (audio_format != 1 || bits_per_sample != 16) {
         printf("Error: Unsupported audio format\n");
         fclose(file);
         return NULL;
     }
     
     // Reset to start for chunk search
     fseek(file, 12, SEEK_SET);
     
     // Find data chunk
     uint32_t data_size;
     long data_pos = FindChunk(file, "data", &data_size);
     if (data_pos < 0) {
         printf("Error: No data chunk found\n");
         fclose(file);
         return NULL;
     }
     
     fseek(file, data_pos, SEEK_SET);
     
     // Calculate sample count
     uint32_t total_samples = data_size / (bits_per_sample / 8) / channels;
     
     // Limit to maximum size
     if (total_samples > kMaxAudioSampleSize) {
         printf("Truncating audio from %u to %u samples\n", 
                total_samples, kMaxAudioSampleSize);
         total_samples = kMaxAudioSampleSize;
     }
     
     // Allocate memory for raw audio
     int16_t* raw_data = (int16_t*)malloc(total_samples * channels * sizeof(int16_t));
     if (!raw_data) {
         printf("Error: Cannot allocate memory\n");
         fclose(file);
         return NULL;
     }
     
     // Read audio data
     size_t samples_read = fread(raw_data, sizeof(int16_t), 
                                 total_samples * channels, file);
     fclose(file);
     
     if (samples_read != total_samples * channels) {
         printf("Warning: Expected %u samples, read %zu\n", 
                total_samples * channels, samples_read);
         total_samples = samples_read / channels;
     }
     
     // Convert to mono if necessary
     int16_t* mono_data;
     if (channels == 1) {
         mono_data = raw_data;
     } else {
         printf("Converting %d-channel audio to mono\n", channels);
         mono_data = (int16_t*)malloc(total_samples * sizeof(int16_t));
         if (!mono_data) {
             free(raw_data);
             return NULL;
         }
         ConvertStereoToMono(raw_data, mono_data, total_samples);
         free(raw_data);
     }
     
     // Resample if necessary
     int16_t* final_data = mono_data;
     uint32_t final_samples = total_samples;
     
     if (file_sample_rate != kAudioSampleFrequency) {
         printf("Resampling from %u Hz to %u Hz\n", 
                file_sample_rate, kAudioSampleFrequency);
         
         final_data = ResampleAudio(mono_data, total_samples, 
                                   file_sample_rate, kAudioSampleFrequency,
                                   &final_samples);
         if (!final_data) {
             free(mono_data);
             return NULL;
         }
         
         if (final_data != mono_data) {
             free(mono_data);
         }
     }
     
         // Normalize audio to prevent clipping
    // NormalizeAudioData(final_data, final_samples);  // DISABLED FOR TESTING
    printf("Audio normalization disabled for testing\n");
     
     // Calculate and print statistics
     int16_t min_val = 32767, max_val = -32768;
     int64_t sum = 0;
     for (uint32_t i = 0; i < final_samples; i++) {
         if (final_data[i] < min_val) min_val = final_data[i];
         if (final_data[i] > max_val) max_val = final_data[i];
         sum += final_data[i];
     }
     
     printf("\nAudio loaded successfully:\n");
     printf("  Samples: %u\n", final_samples);
     printf("  Sample rate: %u Hz\n", kAudioSampleFrequency);
     printf("  Duration: %.2f seconds\n", 
            (float)final_samples / kAudioSampleFrequency);
     printf("  Range: [%d, %d]\n", min_val, max_val);
     printf("  Mean: %.2f\n", (float)sum / final_samples);
     
     *sample_count = final_samples;
     *sample_rate = kAudioSampleFrequency;
     
     return final_data;
 }