/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/gemm_support.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/padding.h"

#include <iostream>

namespace tflite {
namespace ops {
namespace builtin {
namespace transpose_conv {

enum KernelType {
  kReference,
};

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multipler plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  bool hasBias = node->inputs->size == 4;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, hasBias || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[2]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  // Check dimensionality of input, filter
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  // Check input channels matching filter
  TF_LITE_ENSURE_EQ(context, input->dims->data[3], filter->dims->data[0]);

  // Check types. (We assume that UINT8 refers to quantized tensors)
  TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, filter->type, data_type);

  // Current implementation only supports equal strides in the row and column dimensions
  auto stride_width = params->stride_width;
  auto stride_height = params->stride_height;
  TF_LITE_ENSURE_EQ(context, stride_width, stride_height);

  // Check the expected input shape from the output shape
  // with the VALID padding condition.
  auto output_width = output->dims->data[2];
  auto output_height = output->dims->data[1];
  auto filter_width = filter->dims->data[2];
  auto filter_height = filter->dims->data[1];
  int width = input->dims->data[2];
  int height = input->dims->data[1];

  TfLiteTensor* bias = nullptr;
  if (hasBias) {
    bias = &context->tensors[node->inputs->data[3]];
    if (data_type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_EQ(context, bias->type, data_type);
    }
    TF_LITE_ENSURE_EQ(context, bias->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], filter->dims->data[3]);
  }

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  auto computeOutSize = [padding](int imageSize, int filterSize,
                                  int stride) -> int {
    return padding == kTfLitePaddingSame
               ? (imageSize + stride - 1) / stride
               : padding == kTfLitePaddingValid
                     ? (imageSize - filterSize + stride) / stride
                     : 0;
  };
  int expected_width = computeOutSize(output_width, filter_width, params->stride_width);
  int expected_height = computeOutSize(output_height, filter_height, params->stride_height);

  TF_LITE_ENSURE_EQ(context, input->dims->data[2], expected_width);
  TF_LITE_ENSURE_EQ(context, input->dims->data[1], expected_height);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
      context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplierSmallerThanOne(real_multiplier, &data->output_multiplier,
                                     &data->output_shift);
    CalculateActivationRangeUint8(params->activation, output,
                                  &data->output_activation_min,
                                  &data->output_activation_max);
  }

  printf("TransposeConv Prepare Finish\n");
  return kTfLiteOk;
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                 TfLiteTransposeConvParams* params, OpData* data, TfLiteTensor* input,
                 TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* output) {
  auto input_offset = input->params.zero_point;
  auto filter_offset = filter->params.zero_point;
  auto output_offset = output->params.zero_point;

  if (kernel_type == kReference) {
    reference_ops::TransposeConv(GetTensorData<uint8_t>(input), GetTensorDims(input), input_offset,
    GetTensorData<uint8_t>(filter), GetTensorDims(filter), filter_offset,
    GetTensorData<int32_t>(bias), GetTensorDims(bias), params->stride_width,
    params->stride_height, data->padding.width, data->padding.height,
    output_offset, data->output_multiplier, data->output_shift,
    data->output_activation_min, data->output_activation_max,
    GetTensorData<uint8_t>(output), GetTensorDims(output));
  } else {
    // TDDO: Optimized version.
  }
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteTransposeConvParams* params, OpData* data, TfLiteTensor* input,
               TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(params->activation, &output_activation_min,
                                &output_activation_max);
  const float* filter_data = GetTensorData<float>(filter);

  if (kernel_type == kReference) {
    reference_ops::TransposeConv(
      GetTensorData<float>(input), GetTensorDims(input), filter_data,
      GetTensorDims(filter), GetTensorData<float>(bias), GetTensorDims(bias),
      params->stride_width, params->stride_height, data->padding.width,
      data->padding.height, output_activation_min, output_activation_max,
      GetTensorData<float>(output), GetTensorDims(output));
  } else {
    // TDDO: Optimized version.
  }
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[2]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  bool hasBias = node->inputs->size == 4;
  TfLiteTensor* bias =
    hasBias ? &context->tensors[node->inputs->data[3]] : nullptr;

  switch (input->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      EvalFloat<kernel_type>(context, node, params, data, input, filter, bias, output);
      break;
    case kTfLiteUInt8:
      EvalQuantized<kernel_type>(context, node, params, data, input, filter, bias, output);
      break;
    default:
      std::cout << "Not support type = " << static_cast<int>(input->type) << std::endl;
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace transpose_conv

TfLiteRegistration* Register_TRANSPOSE_CONV_REF() {
  static TfLiteRegistration r = {transpose_conv::Init, transpose_conv::Free,
                                 transpose_conv::Prepare,
                                 transpose_conv::Eval<transpose_conv::kReference>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSE_CONV() {
    return Register_TRANSPOSE_CONV_REF();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
