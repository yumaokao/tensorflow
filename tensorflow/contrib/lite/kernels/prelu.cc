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
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace prelu {


TfLiteStatus PReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* input = GetInput(context, node, 0);
  //TfLiteTensor* alpha = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}


TfLiteStatus PReluEval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* alpha = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      float* in = input->data.f;
      float* in_end = in + elements;
      float* ai = alpha->data.f;
      float* out = output->data.f;
      for (; in < in_end; in++, out++){
          float pos = std::max(0.f, *in);
          float neg = *ai*(*in - fabs(*in));
          *out = pos + neg;
      }
      return kTfLiteOk;
    }
    break;
    case kTfLiteUInt8: {
      size_t elements = input->bytes / sizeof(uint8_t);
      uint8_t* in = input->data.uint8;
      uint8_t* in_end = in + elements;
      uint8_t* ai = alpha->data.uint8;
      float mul_val = *ai * alpha->params.scale;
      uint8_t* out = output->data.uint8;
      for (; in < in_end; in++, out++){
          if (*in >= output->params.zero_point){
            *out = *in;
          } else {
              float real = ((float)*in - (float)output->params.zero_point) * output->params.scale;
              real = 2.0* mul_val * real;
              int32_t neg = round(real/output->params.scale) + output->params.zero_point;
              if (neg < 0){
                neg = 0;
              }
              *out = (uint8_t) neg;
          }
          printf("%d,%d\n", *out, *in);
      }
      return kTfLiteOk;
    }
    break;
    default:
      context->ReportError(context, "Only float32 supported currently.");
      return kTfLiteError;
  }
}


}  // namespace prelu

TfLiteRegistration* Register_PRELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 prelu::PReluPrepare,
                                 prelu::PReluEval};
  return &r;
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
