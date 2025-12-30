#!/usr/bin/env python3
"""Convert TFLite model to C array format"""
import sys

def convert_tflite_to_c(input_file, output_c, output_h, var_name):
    with open(input_file, 'rb') as f:
        data = f.read()
    
    # Generate C file
    with open(output_c, 'w') as f:
        f.write(f'#include "{output_h}"\n\n')
        f.write(f'__attribute__((aligned(8))) const int8_t {var_name}[] = {{\n')
        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write('    ')
            f.write(f'0x{byte:02x},')
            if i % 12 == 11:
                f.write('\n')
        f.write('\n};\n\n')
        f.write(f'const size_t {var_name}_len = {len(data)};\n')
    
    # Generate H file
    with open(output_h, 'w') as f:
        guard = output_h.upper().replace('.', '_').replace('/', '_')
        f.write(f'#ifndef {guard}_\n')
        f.write(f'#define {guard}_\n\n')
        f.write('#include <stddef.h>\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'extern const int8_t {var_name}[];\n')
        f.write(f'extern const size_t {var_name}_len;\n\n')
        f.write(f'#endif  // {guard}_\n')
    
    print(f'Converted {input_file} ({len(data)} bytes) to {output_c} and {output_h}')

if __name__ == '__main__':
    # 尝试 uint8 版本，可能是完全量化的
    convert_tflite_to_c(
        'digits_model_uint8.tflite',
        'micro_speech_quantized_model_data.c',
        'micro_speech_quantized_model_data.h',
        'micro_speech_quantized_tflite'
    )
