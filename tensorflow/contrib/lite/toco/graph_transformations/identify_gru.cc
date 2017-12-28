#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

namespace {

std::vector<std::unique_ptr<Operator>>::iterator FindOperator(
    Model* model, const Operator& op) {
  auto it = model->operators.begin();
  for (; it != model->operators.end(); ++it) {
    if (it->get() == &op) {
      break;
    }
  }
  return it;
}

// Returns true if the given operator has exactly 1 input, and is connected to
// the given op_type.
// We use kNone to indicate an input unattached to an operator output. Usually
// these are the static input arrays.
bool MatchOperatorInputs(const Operator& op, const Model& model,
                         OperatorType op_type, Operator** connected_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 1) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((op_type == OperatorType::kNone) && (x != nullptr)) {
    return false;
  }
  if ((op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (x->type != op_type)) {
    return false;
  }

  // Successfully matched. Optionally return matching input operators.
  if (connected_op) {
    *connected_op = x;
  }

  return true;
}

// Returns true if the given operator has exactly 2 inputs, which are connected
// to the given op_types.
// We use kNone to indicate an input unattached to an operator output. Usually
// these are the static input arrays.
bool MatchOperatorInputs(const Operator& op, const Model& model,
                         OperatorType a_op_type, Operator** a_op,
                         OperatorType b_op_type, Operator** b_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 2) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((a_op_type == OperatorType::kNone) && (x != nullptr)) {
    return false;
  }
  if ((a_op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (x->type != a_op_type)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  Operator* y = GetOpWithOutput(model, op.inputs[1]);
  if ((b_op_type == OperatorType::kNone) && (y != nullptr)) {
    return false;
  }
  if ((b_op_type != OperatorType::kNone) && (y == nullptr)) {
    return false;
  }

  // Check that second operator, if connected, is of correct type
  if ((y != nullptr) && (y->type != b_op_type)) {
    return false;
  }

  // Successfully matched. Optionally return matching input operators.
  if (a_op != nullptr) {
    *a_op = x;
  }
  if (b_op != nullptr) {
    *b_op = y;
  }
  return true;
}

// Returns true if the given operator has exactly 3 inputs, which are connected
// to the given op_types.
// We use kNone to indicate an input unattached to an operator output. Usually
// these are the static input arrays.
bool MatchOperatorInputs(const Operator& op, const Model& model,
                         OperatorType a_op_type, Operator** a_op,
                         OperatorType b_op_type, Operator** b_op,
                         OperatorType c_op_type, Operator** c_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 3) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((a_op_type == OperatorType::kNone) && (x != nullptr)) {
    return false;
  }
  if ((a_op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (x->type != a_op_type)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  Operator* y = GetOpWithOutput(model, op.inputs[1]);
  if ((b_op_type == OperatorType::kNone) && (y != nullptr)) {
    return false;
  }
  if ((b_op_type != OperatorType::kNone) && (y == nullptr)) {
    return false;
  }

  // Check that second operator, if connected, is of correct type
  if ((y != nullptr) && (y->type != b_op_type)) {
    return false;
  }

  // Check if third input is disconnected/connected to an operator
  Operator* z = GetOpWithOutput(model, op.inputs[2]);
  if ((c_op_type == OperatorType::kNone) && (z != nullptr)) {
    return false;
  }
  if ((c_op_type != OperatorType::kNone) && (z == nullptr)) {
    return false;
  }

  // Check that third operator, if connected, is of correct type
  if ((z != nullptr) && (z->type != c_op_type)) {
    return false;
  }

  // Successfully matched. Optionally return matching input operators.
  if (a_op != nullptr) {
    *a_op = x;
  }
  if (b_op != nullptr) {
    *b_op = y;
  }
  if (c_op != nullptr) {
    *c_op = z;
  }
  return true;
}

} // namespace


bool IdentifyGruCell::Run(Model* model, std::size_t op_index) {

  auto op_it = model->operators.begin() + op_index;
  Operator* final_output_add = op_it->get();
  if (final_output_add->type != OperatorType::kAdd) {
    return false;
  }
  Operator *prev_state_mul, *candidate_activation_mul;
  if (!MatchOperatorInputs(*final_output_add, *model, OperatorType::kMul,
                           &prev_state_mul, OperatorType::kMul,
                           &candidate_activation_mul)) {
    return false;
  }

  Operator *update_gate_sub, *candidate_activation;
  if (!MatchOperatorInputs(*candidate_activation_mul, *model, OperatorType::kSub,
                           &update_gate_sub, OperatorType::kTanh,
                           &candidate_activation)) {
    return false;
  }

  Operator *fc_reset_input;
  if (!MatchOperatorInputs(*candidate_activation, *model, OperatorType::kFullyConnected,
                           &fc_reset_input)) {
    return false;
  }

  Operator *concat_reset_input;
  if (!MatchOperatorInputs(*fc_reset_input, *model,
                           OperatorType::kConcatenation, &concat_reset_input,
                           OperatorType::kNone, nullptr, OperatorType::kNone,
                           nullptr)) {
    return false;
  }

  Operator *reset_state_mul;
  if (!MatchOperatorInputs(*concat_reset_input, *model, OperatorType::kNone,
                           nullptr, OperatorType::kMul,
                           &reset_state_mul)) {
    return false;
  }

  Operator *gate_output_split, *prev_state;
  if (!MatchOperatorInputs(*prev_state_mul, *model, OperatorType::kTensorFlowSplit,
                            &gate_output_split, OperatorType::kAdd, &prev_state)) { // kAdd is the output of previous gru_cell
    return false;
  }

  Operator *tmp, *tmp2;
    if (!MatchOperatorInputs(*reset_state_mul, *model, OperatorType::kTensorFlowSplit, &tmp,
                            OperatorType::kAdd, &tmp2) || // kAdd is the output of previous gru_cell
      (tmp != gate_output_split) || (tmp2 != prev_state)) {
    return false;
  }

  Operator *gate_output;
  if (!MatchOperatorInputs(*gate_output_split, *model, OperatorType::kNone,
                            nullptr, OperatorType::kLogistic, &gate_output)) {
    return false;
  }

  Operator *fc_input;
  if (!MatchOperatorInputs(*gate_output, *model, OperatorType::kFullyConnected,
                           &fc_input)) {
    return false;
  }

  Operator *concat_input;
  if (!MatchOperatorInputs(*fc_input, *model,
                           OperatorType::kConcatenation, &concat_input,
                           OperatorType::kNone, nullptr, OperatorType::kNone,
                           nullptr)) {
    return false;
  }

  assert(concat_input->inputs[1] == concat_reset_input->inputs[1]); // cur input
  assert(concat_input->inputs[0] == reset_state_mul->inputs[0]); // prev state

  printf("\n===== Found GRU cell =====\n");

  return false;
}

}
