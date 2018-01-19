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
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace transpose_conv {


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  //TfLiteTensor* input = GetInput(context, node, 0);
  //TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* filter = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* bias = &context->tensors[node->inputs->data[2]];

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  auto zero_point = input->params.zero_point;
  auto scale      = input->params.scale;

//  reference_ops::Dequantize(GetTensorData<uint8_t>(input), GetTensorDims(input),
//                            zero_point, scale,
//                            GetTensorData<float>(output), GetTensorDims(output));

  return kTfLiteOk;
}


}  // namespace transposeconv


TfLiteRegistration* Register_TRANSPOSE_CONV() {
  static TfLiteRegistration r = {nullptr, nullptr,
                     transpose_conv::Prepare,transpose_conv::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
