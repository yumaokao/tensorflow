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



bool ResolvePRelu::Run(Model* model, std::size_t op_index) {


/*

  const bool is_mulop_input_constant[2] = {
    IsConstantParameterArray(*model, add_op_input_ops[index_mul_input_for_addop]->inputs[0]),
    IsConstantParameterArray(*model, add_op_input_ops[index_mul_input_for_addop]->inputs[1]),
  };
  const int index_of_constant_mulop_input = is_mulop_input_constant[0] ? 0 : 1;
  const int index_of_variable_mulop_input = is_mulop_input_constant[0] ? 1 : 0;*/
  //CHECK(is_input_constant[index_of_constant_input]);
  //CHECK(!is_input_constant[index_of_variable_input]);


  auto abs_it = model->operators.begin() + op_index;
  if (abs_it->get()->type != OperatorType::kTensorFlowAbs) {
    return false;
  }
  auto* abs_op = abs_it->get();
  AddMessageF("Searching PRelu Pattern...\nFind abs=%s", LogName(*abs_op));

  Operator* sub_op  = GetOpWithInput(*model, abs_op->outputs[0]);
  if (sub_op->type != OperatorType::kSub){
    return false;
  }
  AddMessageF("Find sub=%s", LogName(*sub_op));

  Operator* mul_op  = GetOpWithInput(*model, sub_op->outputs[0]);
  if (mul_op->type != OperatorType::kMul){
    return false;
  }
  AddMessageF("Find mul=%s", LogName(*mul_op));

  Operator* mul1_op = GetOpWithInput(*model, mul_op->outputs[0]);
  if (mul1_op->type != OperatorType::kMul){
    if (mul1_op-> type == OperatorType::kAdd){
        // Only one mul case
        mul1_op = nullptr;
    } else {
        return false;
    }
  }

  Operator* add_op = nullptr;
  if (mul1_op == nullptr){
      add_op = GetOpWithInput(*model, mul_op->outputs[0]);
  } else {
      AddMessageF("Find 2nd mul=%s", LogName(*mul1_op));
      add_op = GetOpWithInput(*model, mul1_op->outputs[0]);
  }

  if (add_op->type != OperatorType::kAdd){
    return false;
  }
  AddMessageF("Find add=%s", LogName(*add_op));


  std::vector<Operator*> add_op_input_ops { GetOpWithOutput(*model, add_op->inputs[0]),
                                            GetOpWithOutput(*model, add_op->inputs[1]) };


  const int index_relu_input_for_addop = add_op_input_ops[0]->type == OperatorType::kRelu ? 0 : 1;
  const int index_mul_input_for_addop  = add_op_input_ops[1]->type == OperatorType::kMul  ? 1 : 0;

  Operator* relu_op = GetOpWithOutput(*model, add_op->inputs[index_relu_input_for_addop]);
  if (relu_op->type != OperatorType::kRelu){
    return false;
  }

  AddMessageF("Find relu=%s Recognize PRelu Pattern.", LogName(*relu_op));

   /*  show by single message
   if (mul1_op == nullptr){
      AddMessageF("Find PRelu Patter, abs=%s, sub=%s, mul=%s,"
                  "add=%s, relu=%s ",
                   LogName(*abs_op),  LogName(*sub_op), LogName(*mul_op),
                   LogName(*add_op), LogName(*relu_op));
   } else {
      AddMessageF("Find PRelu Patter, abs=%s, sub=%s, mul=%s,"
                  "mul1=%s, add=%s, relu=%s ",
                   LogName(*abs_op),  LogName(*sub_op), LogName(*mul_op),
                   LogName(*mul1_op), LogName(*add_op), LogName(*relu_op));
   }*/


  //auto& input_arr  = model->GetArray(abs_op->inputs[0]);

  // Create PRelu Op
  auto* prelu_op = new PReluOperator;
  const string prelu_name = AvailableArrayName(*model, "PRelu");
  const string alpha_name = AvailableArrayName(*model, "alpha");

  prelu_op->inputs  = {relu_op->inputs[0], alpha_name};
  prelu_op->outputs = {add_op->outputs[0]};

  auto& alpha_array = model->GetOrCreateArray(alpha_name);

  // neg = alphas * (_x - abs(_x)) * 0.5
  // Merge alphas*0.5 => alpha_array = mul*mul1. If needed.
  float mul, mul1;

  if (IsConstantParameterArray(*model, mul_op->inputs[0])){
    auto& mul_array  = model->GetArray(mul_op->inputs[0]);
    auto& mul_float_data = mul_array.GetBuffer<ArrayDataType::kFloat>().data;
    mul = mul_float_data[0];
  } else {
    auto& mul_array  = model->GetArray(mul_op->inputs[1]);
    auto& mul_float_data = mul_array.GetBuffer<ArrayDataType::kFloat>().data;
    mul = mul_float_data[0];
  }


  if (mul1_op != nullptr){
      if (IsConstantParameterArray(*model, mul1_op->inputs[0])){
        auto& mul_array  = model->GetArray(mul1_op->inputs[0]);
        auto& mul_float_data = mul_array.GetBuffer<ArrayDataType::kFloat>().data;
        mul1 = mul_float_data[0];
      } else {
        auto& mul_array  = model->GetArray(mul1_op->inputs[1]);
        auto& mul_float_data = mul_array.GetBuffer<ArrayDataType::kFloat>().data;
        mul1 = mul_float_data[0];
      }
  } else {
    mul1 = 1.0;
  }


  alpha_array.data_type = ArrayDataType::kFloat;
  alpha_array.copy_shape(Shape({1}));

  auto& alpha_float_data = alpha_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  alpha_float_data.resize(1, 0.0);

  alpha_float_data[0] = mul * mul1;
  //DropMinMax(model, param_name);

  auto prelu_it = model->operators.emplace(abs_it, prelu_op);

  // Erase all the other ops & arrays
  model->EraseArray(abs_op->outputs[0]);
  model->EraseArray(mul_op->inputs[0]);
  model->EraseArray(mul_op->inputs[1]);
  if (mul1_op != nullptr){
    model->EraseArray(mul1_op->inputs[0]);
    model->EraseArray(mul1_op->inputs[1]);
  }
  model->EraseArray(add_op->inputs[0]);

  model->operators.erase(FindOperator(model, sub_op));
  model->operators.erase(FindOperator(model, mul_op));
  if (mul1_op != nullptr){
    model->operators.erase(FindOperator(model, mul1_op));
  }
  model->operators.erase(FindOperator(model, add_op));
  model->operators.erase(FindOperator(model, relu_op));

  //model->operators.erase(FindOperator(model, abs_op));
  model->operators.erase(abs_it);
  return true;

}

}  // namespace toco
