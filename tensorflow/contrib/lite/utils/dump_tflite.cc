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
// NOTE: this is an example driver that converts a tflite model to TensorFlow.
// This is an example that will be integrated more tightly into tflite in
// the future.
#include <cstdarg>
#include <cstdio>
#include "tensorflow/contrib/lite/model.h"

// TODO(aselle): FATAL leaves resources hanging.
void FATAL(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fflush(stderr);
  exit(1);
}

void Dump(const char* filename) {
  auto model = tflite::FlatBufferModel::BuildFromFile(filename);
  if (!model || !model->CheckModelIdentifier()) {
    FATAL("Cannot read file %s\n", filename);
  }

  auto model_ = model->GetModel();
  auto* subgraphs = model_->subgraphs();
  printf("number of subgraphs: %d\n", subgraphs->size());

  auto opcodes = model_->operator_codes();
  printf("number of opcodes: %d\n", opcodes->size());
  for (unsigned int i = 0; i < opcodes->Length(); ++i) {
    const auto* opcode = opcodes->Get(i);
    auto op = opcode->builtin_code();
    printf("  %2d: buildin_code: %2d %s\n", i, op, EnumNameBuiltinOperator(op));
  }

  auto* buffers = model_->buffers();
  printf("number of buffers: %d\n", buffers->size());
  for (unsigned int i = 0; i < buffers->Length(); ++i) {
    const auto* buffer = buffers->Get(i);
    if (const auto* array = buffer->data()) {
      size_t size = array->size();
      printf("  %2d: size %zu\n", i, size);
      /* float weights[16];
      uint8_t * wptr = (uint8_t*) weights;
      memcpy(wptr, array->data(), 64);
      printf("  [0]: %e\n", weights[0]);
      printf("  [1]: %e\n", weights[1]); */
    } else {
      printf("  %2d: size 0\n", i);
    }
  }

  const tflite::SubGraph* subgraph = (*subgraphs)[0];
  auto tensors = subgraph->tensors();
  printf("number of tensors: %d\n", tensors->size());
  for (unsigned int i = 0; i < tensors->Length(); ++i) {
    const auto* tensor = tensors->Get(i);
    printf("  %2d: name %s type %s buffer %d",
           i, tensor->name()->c_str(),
           EnumNameTensorType(tensor->type()),
           tensor->buffer());

    const auto* buffer = buffers->Get(tensor->buffer());
    size_t size = 0;
    if (const auto* array = buffer->data())
      size = array->size();
    printf(" -> size %zu", size);

    const auto* shape = tensor->shape();
    printf(" shape [");
    for (auto s : *shape)
        printf(" %d", s);
    printf(" ]");

    const auto* quant_info = tensor->quantization();
    if (quant_info != nullptr) {
      if ((quant_info->min() != nullptr) && (quant_info->max() != nullptr)
          && (quant_info->scale() != nullptr) && (quant_info->zero_point() != nullptr)) {
        
        printf(" minmax (%f %f) quantization (%f %ld)\n",
            quant_info->min()->Get(0), quant_info->max()->Get(0),
            quant_info->scale()->Get(0), quant_info->zero_point()->Get(0));
      }
      else {
        printf("\n");
      }
    }
    else {
      printf("\n");
    }
  }

  auto operators = subgraph->operators();
  printf("number of operators: %d\n", operators->Length());
  for (unsigned int i = 0; i < operators->Length(); ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    const auto* opcode = opcodes->Get(index);
    auto bop = opcode->builtin_code();
    auto botype = op->builtin_options_type();
    printf("  %2d: index %2d -> %2d %s builtin_options_type %s\n",
           i, index,
           bop, EnumNameBuiltinOperator(bop),
           EnumNameBuiltinOptions(botype));
    const auto* inputs = op->inputs();
    printf("      inputs: [");
    for (auto i : *inputs) {
        const auto* tensor = tensors->Get(i);
        printf(" %s", tensor->name()->c_str());
    }
    const auto* outputs = op->outputs();
    printf(" ] -> outputs: [");
    for (auto o : *outputs) {
        const auto* tensor = tensors->Get(o);
        printf(" %s", tensor->name()->c_str());
    }
    printf(" ]\n");

  }

  const auto inputs = subgraph->inputs();
  printf("number of input tensors: %d\n", inputs->size());
  for (unsigned int i = 0; i < inputs->Length(); ++i) {
    const auto index = inputs->Get(i);
    const auto* tensor = tensors->Get(index);
    printf("  %2d: index %d -> name %s\n",
	   i, index, tensor->name()->c_str());
  }

  const auto outputs = subgraph->outputs();
  printf("number of output tensors: %d\n", outputs->size());
  for (unsigned int i = 0; i < outputs->Length(); ++i) {
    const auto index = outputs->Get(i);
    const auto* tensor = tensors->Get(index);
    printf("  %2d: index %d -> name %s\n",
	   i, index, tensor->name()->c_str());
  }
}

int main(int argc, char* argv[]) {
  bool use_nnapi = true;
  if (argc == 3) {
    use_nnapi = strcmp(argv[2], "1") == 0 ? true : false;
  }
  if (argc < 2) {
    fprintf(stderr,
            "Compiled " __DATE__ __TIME__
            "\n"
            "Usage!!!: %s <tflite model>"
            "{ use nn api i.e. 0,1}\n",
            argv[0]);
    return 1;
  }
  Dump(argv[1]);
  return 0;
}
