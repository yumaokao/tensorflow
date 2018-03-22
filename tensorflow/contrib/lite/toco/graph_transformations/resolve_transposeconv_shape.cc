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
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveTransposeConvShape::Run(Model* model, std::size_t op_index) {
  auto op_it = model->operators.begin() + op_index;
  auto* op = op_it->get();
  if (op->type != OperatorType::kTransposeConv) {
    return false;
  }

  CHECK_GE(op->inputs.size(), 3);

  const auto& shape_array = model->GetArray(op->inputs[0]);
  if (!shape_array.buffer) {
    // Yield until the shape is determined
    return false;
  }

  std::vector<int32> shape_data = shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  auto* transpose_conv_op = dynamic_cast<TransposeConvOperator*>(op);

  if ((transpose_conv_op->out_shape_N
        | transpose_conv_op->out_shape_H
        | transpose_conv_op->out_shape_W
        | transpose_conv_op->out_shape_C) != 0) {
    // shape already be set before
    return false;
  }

  transpose_conv_op->out_shape_N = shape_data[0];
  transpose_conv_op->out_shape_H = shape_data[1];
  transpose_conv_op->out_shape_W = shape_data[2];
  transpose_conv_op->out_shape_C = shape_data[3];

  return true;
}
}
