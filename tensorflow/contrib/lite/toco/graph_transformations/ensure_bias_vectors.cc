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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool ProcessLinearOperator(Model* model, Operator* op) {
  uint8_t input_num_with_bias = 3;
  if (op->type == OperatorType::kTransposeConv) {
    input_num_with_bias = 4;
  }

  if (op->inputs.size() >= input_num_with_bias) {
    return false;
  }
  const string& output_name = op->outputs[0];
  const string& bias_name = AvailableArrayName(*model, output_name + "_bias");
  op->inputs.push_back(bias_name);
  DCHECK_EQ(op->inputs.size(), input_num_with_bias);
  auto& bias_array = model->GetOrCreateArray(bias_name);
  bias_array.data_type = ArrayDataType::kFloat;

  return true;
}
}  // namespace

bool EnsureBiasVectors::Run(Model* model, std::size_t op_index) {
  auto* op = model->operators[op_index].get();
  if (op->type == OperatorType::kConv ||
      op->type == OperatorType::kDepthwiseConv ||
      op->type == OperatorType::kFullyConnected ||
      op->type == OperatorType::kTransposeConv) {
    if (ProcessLinearOperator(model, op)) {
      AddMessageF("Added bias vector to %s", LogName(*op));
      return true;
    }
  }
  return false;
}

}  // namespace toco
