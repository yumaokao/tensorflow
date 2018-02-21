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
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

std::vector<std::unique_ptr<Operator>>::iterator FindOperator(
    Model* model, const Operator* op) {
  auto it = model->operators.begin();
  for (; it != model->operators.end(); ++it) {
    if (it->get() == op) {
      break;
    }
  }
  return it;
}
}  // namespace



bool ResolveLeakyRelu::Run(Model* model, std::size_t op_index) {

  auto mul_it = model->operators.begin() + op_index;
  if (mul_it->get()->type != OperatorType::kMul) {
    return false;
  }
  auto* mul_op = mul_it->get();
  AddMessageF("Searching LeakyRelu Pattern...\nFind mul=%s", LogName(*mul_op));

  Operator* maximum_op = GetOpWithInput(*model, mul_op->outputs[0]);
  if (maximum_op->type != OperatorType::kTensorFlowMaximum){
    return false;
  }
  AddMessageF("Find maximum=%s", LogName(*maximum_op));

  // Create LeakyRelu Op
  auto* leakyrelu_op = new LeakyReluOperator;
  leakyrelu_op->inputs  = {mul_op->inputs[1], mul_op->inputs[0]};
  leakyrelu_op->outputs = maximum_op->outputs;

  auto leakyrelu_it = model->operators.emplace(mul_it, leakyrelu_op);

  // Erase all the other ops & arrays
  model->operators.erase(FindOperator(model, mul_op));
  model->operators.erase(FindOperator(model, maximum_op));

  return true;

}

}  // namespace toco
