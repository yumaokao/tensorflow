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



bool ResolveDilatedConv::Run(Model* model, std::size_t op_index) {

  auto space2batch_it = model->operators.begin() + op_index;
  if (space2batch_it->get()->type != OperatorType::kSpaceToBatchND) {
    return false;
  }
  auto* space2batch_op = static_cast<SpaceToBatchNDOperator*>(space2batch_it->get());
  AddMessageF("Searching Dilated Conv Pattern...\nFind SpaceToBatchND = %s", LogName(*space2batch_op));

  auto* conv_op  = static_cast<ConvOperator*>(GetOpWithInput(*model, space2batch_op->outputs[0]));
  if (conv_op == nullptr) {
    AddMessageF("Conv op Not found");
    return false;
  }
  if (conv_op->type != OperatorType::kConv) {
    return false;
  }
  if (conv_op->stride_width != 1 || conv_op->stride_height != 1) {
    return false;
  }
  AddMessageF("Find Conv=%s", LogName(*conv_op));

  auto* batch2space_op  = GetOpWithInput(*model, conv_op->outputs[0]);
  if (batch2space_op == nullptr) {
    AddMessageF("BatchToSpace op Not found");
    return false;
  }
  if (batch2space_op->type != OperatorType::kBatchToSpaceND) {
    return false;
  }
  AddMessageF("Find BatchToSpaceND = %s", LogName(*batch2space_op));

  auto& weights_array = model->GetArray(conv_op->inputs[1]);
  if (!weights_array.buffer) {
    // Yield until the weights are resolved as a constant array.
    return false;
  }
  if (weights_array.data_type != ArrayDataType::kFloat) {
    return false;
  }

  auto* dilatedConv_op = new DilatedConvOperator;
  const string dilatedConv_name = AvailableArrayName(*model, "DilatedConv");

  dilatedConv_op->inputs    = conv_op->inputs;
  dilatedConv_op->inputs[0] = space2batch_op->inputs[0];
  dilatedConv_op->outputs   = batch2space_op->outputs;

  if (conv_op->outputs.size() > 1) {
    // delete the im2col array.
    model->EraseArray(conv_op->outputs[1]);
  }

  dilatedConv_op->fused_activation_function =
      conv_op->fused_activation_function;

  if (space2batch_op->before_paddings[0] == 0) {
    dilatedConv_op->padding.type = PaddingType::kValid;
  } else {
    dilatedConv_op->padding.type = PaddingType::kSame;
  }

  dilatedConv_op->rate = space2batch_op->block_shape[0];
  if (dilatedConv_op->rate <= 0) {
    return false;
  }

  auto dilatedConv_it = model->operators.emplace(space2batch_it, dilatedConv_op);

  // Erase all the other ops & arrays
  model->operators.erase(FindOperator(model, conv_op));
  model->operators.erase(FindOperator(model, space2batch_op));
  model->operators.erase(FindOperator(model, batch2space_op));
  return true;

}

}  // namespace toco
