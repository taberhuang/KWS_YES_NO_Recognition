/* Minimal compat implementations to satisfy Micro kernels without pulling in
 * tensorflow/lite/kernels/kernel_util.cc (which depends on C++14-heavy
 * utilities not available in some embedded toolchains).
 */

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace tflite {

// TfLiteTensor variant used by several micro kernels (e.g. add_common).
bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

namespace {
TfLiteStatus QuantizeHelper(TfLiteContext* context, float scale,
                            int32_t zero_point, float f, int32_t& q) {
  const float tmp = TfLiteRound(f / scale);
  const bool ok = (tmp >= static_cast<float>(std::numeric_limits<int32_t>::min()) &&
                   tmp <= static_cast<float>(std::numeric_limits<int32_t>::max()));
  TF_LITE_ENSURE(context, ok);
  q = zero_point + static_cast<int32_t>(tmp);
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
  if (output->type == kTfLiteUInt8) {
    qmin = std::numeric_limits<uint8_t>::min();
    qmax = std::numeric_limits<uint8_t>::max();
  } else if (output->type == kTfLiteInt8) {
    qmin = std::numeric_limits<int8_t>::min();
    qmax = std::numeric_limits<int8_t>::max();
  } else if (output->type == kTfLiteInt16) {
    qmin = std::numeric_limits<int16_t>::min();
    qmax = std::numeric_limits<int16_t>::max();
  } else {
    TF_LITE_ENSURE(context, false);
  }

  const float scale = output->params.scale;
  const int32_t zero_point = output->params.zero_point;

  int32_t tmp_q = 0;
  if (activation == kTfLiteActRelu) {
    TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 0.0f, tmp_q));
    *act_min = std::max(qmin, tmp_q);
    *act_max = qmax;
  } else if (activation == kTfLiteActRelu6) {
    TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 0.0f, tmp_q));
    *act_min = std::max(qmin, tmp_q);
    TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 6.0f, tmp_q));
    *act_max = std::min(qmax, tmp_q);
  } else if (activation == kTfLiteActReluN1To1) {
    TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, -1.0f, tmp_q));
    *act_min = std::max(qmin, tmp_q);
    TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 1.0f, tmp_q));
    *act_max = std::min(qmax, tmp_q);
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
  return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale =
      static_cast<double>(input->params.scale * filter->params.scale);
  TF_LITE_ENSURE(context, input_product_scale >= 0);
  *multiplier = input_product_scale /
                static_cast<double>(output->params.scale);
  return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale = static_cast<double>(input->params.scale) *
                                     static_cast<double>(filter->params.scale);
  if (bias) {
    const double bias_scale = static_cast<double>(bias->params.scale);
    const double scale_diff = std::abs(input_product_scale - bias_scale);
    const double output_scale = static_cast<double>(output->params.scale);
    TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02);
  }
  return GetQuantizedConvolutionMultipler(context, input, filter, output, multiplier);
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels) {
  TF_LITE_ENSURE_EQ(context, input->quantization.type, kTfLiteAffineQuantization);
  TF_LITE_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);

  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);

  const bool is_per_channel = affine_quantization->scale->size > 1;
  if (is_per_channel) {
    TF_LITE_ENSURE(context, input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
    TF_LITE_ENSURE(context, filter->type == kTfLiteInt8 || filter->type == kTfLiteInt4);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, num_channels);
    TF_LITE_ENSURE_EQ(context, num_channels,
                      filter->dims->data[affine_quantization->quantized_dimension]);
  }

  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; ++i) {
    const float scale = is_per_channel ? filter_scales[i] : filter_scales[0];
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    per_channel_multiplier[i] = significand;
    per_channel_shift[i] = channel_shift;
  }

  if (input->type == kTfLiteUInt8) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(
        GetQuantizedConvolutionMultipler(context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, multiplier, &exponent);
    *shift = -exponent;
  }

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, activation, output, output_activation_min, output_activation_max));
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift) {
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  return PopulateConvolutionQuantizationParams(
      context, input, filter, bias, output, activation, multiplier, shift,
      output_activation_min, output_activation_max, per_channel_multiplier,
      per_channel_shift, affine_quantization->scale->size);
}

}  // namespace tflite


