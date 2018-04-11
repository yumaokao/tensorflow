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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveReshapeAttributes::Run(Model* model, std::size_t op_index) {
  const auto reshape_it = model->operators.begin() + op_index;
  auto* reshape_op = reshape_it->get();
  if (reshape_op->type != OperatorType::kTensorFlowReshape) {
    return false;
  }

  auto* op = static_cast<TensorFlowReshapeOperator*>(reshape_op);

  if (!op->shape.empty()) return false;

  if (IsConstantParameterArray(*model, reshape_op->inputs[1])) {
    const auto& constant_input_array = model->GetArray(reshape_op->inputs[1]);
    op->shape = constant_input_array.GetBuffer<ArrayDataType::kInt32>().data;
  }

  if (op->shape.empty()) return false;

  // We make the batch dim dynamic if all the other dims are fixed.
  bool make_dynamic = true;
  for (auto s : op->shape) {
    if (s == -1) {
      make_dynamic = false;
      break;
    }
  }
  if (make_dynamic) {
    op->shape[0] = -1;
    auto& shape_array = model->GetArray(reshape_op->inputs[1]);
    shape_array.GetMutableBuffer<ArrayDataType::kInt32>().data[0] = -1;
  }

  return true;
}

}  // namespace toco
