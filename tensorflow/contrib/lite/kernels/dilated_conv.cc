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
#include <iostream>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace dilated_conv {

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
  auto* params = reinterpret_cast<TfLiteDilatedConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Check number of inputs/outputs
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* bias = &context->tensors[node->inputs->data[2]];
  // Check dimensionality of input, filter
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  // Check input channels matching filter
  TF_LITE_ENSURE_EQ(context, input->dims->data[3], filter->dims->data[3]);

  // Check types. (We assume that UINT8 refers to quantized tensors)
  TfLiteType data_type = input->type;
  TF_LITE_ENSURE(context,
                 data_type == kTfLiteFloat32 || data_type == kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, output->type, data_type);
  TF_LITE_ENSURE_EQ(context, filter->type, data_type);
  if (data_type == kTfLiteUInt8) {
    TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
  } else {
    TF_LITE_ENSURE_EQ(context, bias->type, data_type);
  }
  TF_LITE_ENSURE_EQ(context, bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, bias->dims->data[0], filter->dims->data[0]);

  int channels_out = filter->dims->data[0];
  int width = input->dims->data[2];
  int height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int batches = input->dims->data[0];
  int rate = params->rate;

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  auto computeOutSize = [padding](int imageSize, int filterSize, int rate) -> int {
    return padding == kTfLitePaddingSame
               ? imageSize
               : padding == kTfLitePaddingValid
                     ? imageSize - rate * (filterSize - 1)
                     : 0;
  };

  int outWidth = computeOutSize(width, filter_width, rate);
  int outHeight = computeOutSize(height, filter_height, rate);

  data->padding.height =
      ComputePadding(1, height, (filter_height - 1) * params->rate + 1, outHeight);
  data->padding.width =
      ComputePadding(1, width, (filter_width - 1) * params->rate + 1, outWidth);

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

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = outHeight;
  output_size->data[2] = outWidth;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteDilatedConvParams* params, OpData* data, TfLiteTensor* input,
                   TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* output) {
  auto input_offset = -input->params.zero_point;
  auto filter_offset = -filter->params.zero_point;
  auto output_offset = output->params.zero_point;

  switch (kernel_type) {
    case kReference:
      reference_ops::DilatedConv(
            GetTensorData<uint8_t>(input), GetTensorDims(input), input_offset,
            GetTensorData<uint8_t>(filter), GetTensorDims(filter), filter_offset,
            GetTensorData<int32_t>(bias), GetTensorDims(bias),
            data->padding.width, data->padding.height,
            params->rate, output_offset,
            data->output_multiplier, data->output_shift,
            data->output_activation_min, data->output_activation_max,
            GetTensorData<uint8_t>(output), GetTensorDims(output));
      break;
    default:
      // TODO: optimized version
      break;
  }

}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteDilatedConvParams* params, OpData* data, TfLiteTensor* input,
               TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(params->activation, &output_activation_min,
                                &output_activation_max);

  switch (kernel_type) {
    case kReference:
      reference_ops::DilatedConv(
            GetTensorData<float>(input), GetTensorDims(input),
            GetTensorData<float>(filter), GetTensorDims(filter),
            GetTensorData<float>(bias), GetTensorDims(bias),
            data->padding.width, data->padding.height, params->rate,
            output_activation_min, output_activation_max,
            GetTensorData<float>(output), GetTensorDims(output));
    break;
    default:
      // TODO: optimized version
      break;
  }
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteDilatedConvParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  bool hasBias = node->inputs->size == 3;
  TfLiteTensor* bias =
        hasBias ? &context->tensors[node->inputs->data[2]] : nullptr;

  switch (input->type) {
    case kTfLiteFloat32:
      EvalFloat<kernel_type>(context, node, params, data, input, filter, bias, output);
      break;
    case kTfLiteUInt8:
      EvalQuantized<kernel_type>(context, node, params, data, input, filter, bias, output);
      break;
    default:
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace dilatedconv

TfLiteRegistration* Register_DILATED_CONV_REF() {
  static TfLiteRegistration r = {dilated_conv::Init, dilated_conv::Free, dilated_conv::Prepare,
                                 dilated_conv::Eval<dilated_conv::kReference>};
  return &r;
}

TfLiteRegistration* Register_DILATED_CONV() {
  return Register_DILATED_CONV_REF();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
