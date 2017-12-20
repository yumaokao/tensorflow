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
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool RemoveUnusedOp::Run(Model* model, std::size_t op_index) {
  const auto it = model->operators.begin() + op_index;
  const auto* op = it->get();

  // Bail if any output is used, and is not an input_array of
  // the model. We allow specifying an arbitrary input_array,
  // treating the part of the graph leading up to it as unused.
  for (const auto& output : op->outputs) {
    CHECK(model->arrays.count(output));
    // If this output is provided as the model's input array,
    // then we don't need this operator to produce its contents.
    if (IsInputArray(*model, output)) {
      continue;
    }
    // If this output is provided as a RNN's state array,
    // then we don't need this operator to produce its contents.
    // So far this case has only been encountered with TensorFlow
    // Fill ops used to zero-initialize RNN states, which is
    // redundant for us as we zero-initialize RNN states anyway.
    bool found_output_as_rnn_state_array = false;
    for (const auto& rnn_state : model->flags.rnn_states()) {
      if (output == rnn_state.state_array()) {
        // TODO(YMK): second cell could not be checked
        /* CHECK(op->type == OperatorType::kTensorFlowUnsupported);
        CHECK_EQ(static_cast<const TensorFlowUnsupportedOperator*>(op)
                     ->tensorflow_op,
                 "Fill"); */
        found_output_as_rnn_state_array = true;
        break;
      }
    }
    if (found_output_as_rnn_state_array) {
      // continue;
      return false;
    }
    for (const string& output_array : model->flags.output_arrays()) {
      if (output == output_array) {
        return false;
      }
    }
    for (const auto& rnn_state : model->flags.rnn_states()) {
      if (output == rnn_state.back_edge_source_array()) {
        return false;
      }
    }
    if (CountOpsWithInput(*model, output)) {
      return false;
    }
  }

  if (op->unresolved_outputs) {
    AddMessageF("Not discarding %s because it has unresolved outputs.",
                LogName(*op));
    return false;
  }

  AddMessageF("Discarding %s because none of its outputs is used.",
              LogName(*op));

  // At that point we know that none of the outputs is used, so we will
  // definitely remove the node and all its outputs.

  // Remove any input array that is not used by anything else,
  // and that is not the output of some other operator.
  for (const auto& input : op->inputs) {
    if (IsDiscardableArray(*model, input) &&
        CountOpsWithInput(*model, input) == 1 &&
        !GetOpWithOutput(*model, input)) {
      model->arrays.erase(input);
    }
  }

  // Remove the node and its now-unused output arrays.
  for (const auto& output : op->outputs) {
    // If the output array is the model's input array, don't remove that.
    // That's the case when cropping a model at a given --input_array.
    if (!IsDiscardableArray(*model, output)) {
      continue;
    }
    // Likewise, if the output array is a RNN state array, don't remove that.
    bool found_output_as_rnn_state_array = false;
    for (const auto& rnn_state : model->flags.rnn_states()) {
      if (output == rnn_state.state_array()) {
        found_output_as_rnn_state_array = true;
        break;
      }
    }
    if (found_output_as_rnn_state_array) {
      continue;
    }
    // Generic case: do delete this output array.
    model->arrays.erase(output);
  }
  model->operators.erase(it);
  return true;
}

}  // namespace toco
