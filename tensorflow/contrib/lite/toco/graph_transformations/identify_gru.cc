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

// Similar to MatchOperatorInputs except that op_type is considered as
// 'Don't care' if it is OperatorType::kNone.
//
// This is used for some operators that the input opreator type may change
bool MatchPartialOperatorInputs(const Operator& op, const Model& model,
                         OperatorType a_op_type, Operator** a_op,
                         OperatorType b_op_type, Operator** b_op) {
  // Check for required number of inputs
  if (op.inputs.size() != 2) {
    return false;
  }

  // Check if first input is disconnected/connected to an operator
  Operator* x = GetOpWithOutput(model, op.inputs[0]);
  if ((a_op_type != OperatorType::kNone) && (x == nullptr)) {
    return false;
  }

  // Check that first operator, if connected, is of correct type
  if ((x != nullptr) && (a_op_type != OperatorType::kNone) && (x->type != a_op_type)) {
    return false;
  }

  // Check if second input is disconnected/connected to an operator
  Operator* y = GetOpWithOutput(model, op.inputs[1]);
  if ((b_op_type != OperatorType::kNone) && (y == nullptr)) {
    return false;
  }

  // Check that second operator, if connected, is of correct type
  if ((y != nullptr) && (b_op_type != OperatorType::kNone) && (y->type != b_op_type)) {
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

  Operator *fc_activation;
  if (!MatchOperatorInputs(*candidate_activation, *model, OperatorType::kFullyConnected,
                           &fc_activation)) {
    return false;
  }

  Operator *concat_reset_input;
  if (!MatchOperatorInputs(*fc_activation, *model,
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
  // the 2nd input is the previous state
  if (!MatchPartialOperatorInputs(*prev_state_mul, *model, OperatorType::kTensorFlowSplit, &gate_output_split,
                            OperatorType::kNone, &prev_state)) {
    return false;
  }

  Operator *tmp, *tmp2;
  // the 2nd input is the previous state
  if (!MatchPartialOperatorInputs(*reset_state_mul, *model, OperatorType::kTensorFlowSplit, &tmp,
                            OperatorType::kNone, &tmp2) || // kAdd is the output of previous gru_cell
      (tmp != gate_output_split) || (tmp2 != prev_state)) {
    return false;
  }

  Operator *tmp3;
  if (!MatchOperatorInputs(*update_gate_sub, *model, OperatorType::kNone, nullptr,
                            OperatorType::kTensorFlowSplit, &tmp3) ||
      (tmp3 != gate_output_split) || (update_gate_sub->inputs[1] != prev_state_mul->inputs[0])) {
    return false;
  }

  Operator *gate_output;
  if (!MatchOperatorInputs(*gate_output_split, *model, OperatorType::kNone,
                            nullptr, OperatorType::kLogistic, &gate_output)) {
    return false;
  }

  Operator *fc_gate;
  if (!MatchOperatorInputs(*gate_output, *model, OperatorType::kFullyConnected,
                           &fc_gate)) {
    return false;
  }

  Operator *concat_input;
  if (!MatchOperatorInputs(*fc_gate, *model,
                           OperatorType::kConcatenation, &concat_input,
                           OperatorType::kNone, nullptr, OperatorType::kNone,
                           nullptr)) {
    return false;
  }

  if (concat_input->inputs[0] != concat_reset_input->inputs[0]) { // cur input
    return false;
  }
  if ((concat_input->inputs[1] != reset_state_mul->inputs[1]) || 
      (concat_input->inputs[1] != prev_state_mul->inputs[1])) { // prev state
    return false;
  }

  // Emplace a new GRU cell operator
  auto* gru_cell_op = new GruCellOperator;
  gru_cell_op->inputs.resize(GruCellOperator::NUM_INPUTS);
  gru_cell_op->inputs[GruCellOperator::DATA_INPUT] = concat_input->inputs[0];
  gru_cell_op->inputs[GruCellOperator::PREV_STATE_INPUT] = concat_input->inputs[1];
  gru_cell_op->inputs[GruCellOperator::WEIGHTS_ACTIVATION_INPUT] = fc_activation->inputs[1];
  gru_cell_op->inputs[GruCellOperator::BIASES_ACTIVATION_INPUT] = fc_activation->inputs[2];
  gru_cell_op->inputs[GruCellOperator::WEIGHTS_GATE_INPUT] = fc_gate->inputs[1];
  gru_cell_op->inputs[GruCellOperator::BIASES_GATE_INPUT] = fc_gate->inputs[2];

  gru_cell_op->outputs.resize(GruCellOperator::NUM_OUTPUTS);
  gru_cell_op->outputs[GruCellOperator::STATE_OUTPUT] = final_output_add->outputs[0];
  model->operators.emplace(op_it, gru_cell_op);
  AddMessageF("Creating %s replacing equivalent subgraph",
              LogName(*gru_cell_op));

  // Delete arrays and operators replaced by the GRU cell operator. Order is
  // important - DeleteArrayIfUnused() only succeeds if dependent operators
  // have been removed first. Start at the output and work towards the input.
  model->operators.erase(FindOperator(model, *final_output_add));
  DeleteArrayIfUnused(candidate_activation_mul->outputs[0], model);
  model->operators.erase(FindOperator(model, *candidate_activation_mul));
  DeleteArrayIfUnused(candidate_activation->outputs[0], model);
  model->operators.erase(FindOperator(model, *candidate_activation));
  DeleteArrayIfUnused(fc_activation->outputs[0], model);
  model->operators.erase(FindOperator(model, *fc_activation));
  DeleteArrayIfUnused(concat_reset_input->outputs[0], model);
  model->operators.erase(FindOperator(model, *concat_reset_input));
  DeleteArrayIfUnused(prev_state_mul->outputs[0], model);
  model->operators.erase(FindOperator(model, *prev_state_mul));
  DeleteArrayIfUnused(update_gate_sub->outputs[0], model);
  model->operators.erase(FindOperator(model, *update_gate_sub));
  DeleteArrayIfUnused(reset_state_mul->outputs[0], model);
  model->operators.erase(FindOperator(model, *reset_state_mul));
  DeleteArrayIfUnused(gate_output_split->outputs[0], model);
  DeleteArrayIfUnused(gate_output_split->outputs[1], model);
  string dims_array = gate_output_split->inputs[0];
  model->operators.erase(FindOperator(model, *gate_output_split));
  DeleteArrayIfUnused(dims_array, model);
  DeleteArrayIfUnused(gate_output->outputs[0], model);
  model->operators.erase(FindOperator(model, *gate_output));
  DeleteArrayIfUnused(fc_gate->outputs[0], model);
  model->operators.erase(FindOperator(model, *fc_gate));
  DeleteArrayIfUnused(concat_input->outputs[0], model);
  model->operators.erase(FindOperator(model, *concat_input));

  return true;
}

}
