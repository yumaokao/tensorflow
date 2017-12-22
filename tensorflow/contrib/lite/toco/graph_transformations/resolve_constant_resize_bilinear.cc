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

bool ResolveConstantResizeBilinear::Run(Model* model, std::size_t op_index) {
  const auto resize_bilinear_it = model->operators.begin() + op_index;
  auto* resize_bilinear_base_op = resize_bilinear_it->get();
  if (resize_bilinear_base_op->type != OperatorType::kResizeBilinear) {
    return false;
  }

  auto* resize_bilinear_op =
      static_cast<ResizeBilinearOperator*>(resize_bilinear_base_op);

  if (resize_bilinear_op->inputs.size() == 1) {
    return false;
  }

  CHECK(resize_bilinear_op->inputs.size() == 2);

  // inputs[1] is [new_height, new_width] in int32
  if (!IsConstantParameterArray(*model, resize_bilinear_op->inputs[1])) {
    return false;
  }
  const auto& input1_array = model->GetArray(resize_bilinear_op->inputs[1]);
  CHECK(input1_array.data_type == ArrayDataType::kInt32);
  const auto& input1_buffer = input1_array.GetBuffer<ArrayDataType::kInt32>();
  CHECK(input1_buffer.data.size() == 2);

  // Check output array shape
  auto& output_array = model->GetArray(resize_bilinear_op->outputs[0]);
  CHECK(!output_array.buffer);
  if (!output_array.has_shape()) {
    return false;
  }
  CHECK_EQ(output_array.shape().dimensions_count(), 4);
  std::vector<int> output_dims(output_array.shape().dims());
  // Check new_height, new_width
  CHECK_EQ(output_dims[1], input1_buffer.data[0]);
  CHECK_EQ(output_dims[2], input1_buffer.data[1]);

  // Now could remove inputs[1]
  if (CountOpsWithInput(*model, resize_bilinear_op->inputs[1]) == 1) {
    model->arrays.erase(resize_bilinear_op->inputs[1]);
  }
  resize_bilinear_op->inputs.resize(1);

  return false;

}

}  // namespace toco
