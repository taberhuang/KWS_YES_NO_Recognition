/* wav_reader.h - Robust WAV File Reader Header
 * Handles various WAV formats with automatic normalization and resampling
 */

 #ifndef WAV_READER_H_
 #define WAV_READER_H_
 
 #include <stdint.h>
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 // Standard WAV file header structure
 typedef struct {
     char riff[4];           // "RIFF"
     uint32_t file_size;     // File size - 8
     char wave[4];           // "WAVE"
     char fmt[4];            // "fmt "
     uint32_t fmt_size;      // Format chunk size
     uint16_t audio_format;  // Audio format (1 = PCM)
     uint16_t channels;      // Number of channels
     uint32_t sample_rate;   // Sample rate
     uint32_t byte_rate;     // Byte rate
     uint16_t block_align;   // Block alignment
     uint16_t bits_per_sample; // Bits per sample
 } WavHeader;
 
 // Extended format chunk for some WAV files
 typedef struct {
     uint16_t cb_size;       // Size of extension
     uint16_t valid_bits;    // Valid bits per sample
     uint32_t channel_mask;  // Channel mask
     uint8_t sub_format[16]; // Sub-format GUID
 } WavFormatEx;
 
 /**
  * Read WAV file and return normalized audio data
  * 
  * Features:
  * - Automatic conversion from stereo to mono
  * - Automatic resampling to target sample rate (16kHz)
  * - Automatic normalization to prevent clipping
  * - Support for various WAV chunk formats
  * - Robust error handling
  * 
  * @param filename Path to WAV file
  * @param sample_count Output: number of audio samples (after processing)
  * @param sample_rate Output: audio sample rate (always 16000 for this application)
  * @return Pointer to processed audio data (caller must free), NULL on error
  */
 int16_t* ReadWavFile(const char* filename, 
                      uint32_t* sample_count,
                      uint32_t* sample_rate);
 
 /**
  * Validate WAV file format
  * 
  * Checks:
  * - Valid RIFF/WAVE headers
  * - PCM format (uncompressed)
  * - 16-bit samples
  * - Presence of fmt and data chunks
  * 
  * @param filename Path to WAV file
  * @return 1 if valid, 0 if invalid
  */
 int ValidateWavFile(const char* filename);
 
 // Category labels for model output (defined in wav_file_reader.c)
 extern const char* kCategoryLabels[];
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif  // WAV_READER_H_