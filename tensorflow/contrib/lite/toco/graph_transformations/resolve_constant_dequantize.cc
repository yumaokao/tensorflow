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

bool ResolveConstantDequantize::Run(Model* model, std::size_t op_index) {
  const auto dequantize_it = model->operators.begin() + op_index;
  auto* dequantize_base_op = dequantize_it->get();
  if (dequantize_base_op->type != OperatorType::kDequantize) {
    return false;
  }

  auto* dequantize_op =
      static_cast<DequantizeOperator*>(dequantize_base_op);

  if (dequantize_op->inputs.size() == 1) {
    return false;
  }

  CHECK(dequantize_op->inputs.size() == 3);
  // This transformation only applies when the input array is constant.
  if (!IsConstantParameterArray(*model, dequantize_op->inputs[1]) ||
      !IsConstantParameterArray(*model, dequantize_op->inputs[2])) {
    return false;
  }

  auto& input_array = model->GetArray(dequantize_op->inputs[0]);
  auto& output_array = model->GetArray(dequantize_op->outputs[0]);
  CHECK(input_array.data_type == ArrayDataType::kUint8);
  output_array.data_type = ArrayDataType::kFloat;
  CHECK(!output_array.buffer);

  // inputs[1] is min, inputs[2] is max
  const auto& input1_array = model->GetArray(dequantize_op->inputs[1]);
  const auto& input2_array = model->GetArray(dequantize_op->inputs[2]);
  CHECK(input1_array.data_type == ArrayDataType::kFloat);
  CHECK(input2_array.data_type == ArrayDataType::kFloat);
  const auto& input1_buffer = input1_array.GetBuffer<ArrayDataType::kFloat>();
  const auto& input2_buffer = input2_array.GetBuffer<ArrayDataType::kFloat>();
  CHECK(input1_buffer.data.size() == 1);
  CHECK(input2_buffer.data.size() == 1);

  auto& input_minmax = input_array.GetOrCreateMinMax();
  input_minmax.min = input1_buffer.data[0];
  input_minmax.max = input2_buffer.data[0];

  auto& input_qparams = input_array.GetOrCreateQuantizationParams();
  GetQuantizationParamsFromMinMax<ArrayDataType::kUint8>(input_minmax,
                                                         &input_qparams);
  // std::cout << "== get scale & zero_point: (" << input_qparams.scale << ", " << input_qparams.zero_point<< ")" << std::endl;
  // std::cout << "== input array data_type = " << static_cast<std::underlying_type<ArrayDataType>::type>(input_array.data_type) << std::endl;
  // std::cout << "== input final data_type = " << static_cast<std::underlying_type<ArrayDataType>::type>(input_array.final_data_type) << std::endl;

  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input1_buffer.data[0];
  output_minmax.max = input2_buffer.data[0];

  auto& output_qparams = output_array.GetOrCreateQuantizationParams();
  GetQuantizationParamsFromMinMax<ArrayDataType::kUint8>(output_minmax,
                                                         &output_qparams);

  for (int i = 1; i <= 2; i++) {
    if (CountOpsWithInput(*model, dequantize_op->inputs[i]) == 1) {
      model->EraseArray(dequantize_op->inputs[i]);
    }
  }
  dequantize_op->inputs.resize(1);

  return true;
}

}  // namespace toco
