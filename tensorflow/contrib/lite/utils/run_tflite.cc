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
#include <iostream>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

#include "tensorflow/core/util/command_line_flags.h"
#define LOG(x) std::cerr

using tensorflow::Flag;
using tensorflow::string;


// TODO(aselle): FATAL leaves resources hanging.
void FATAL(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fflush(stderr);
  exit(1);
}

#define CHECK_TFLITE_SUCCESS(x)                       \
  if (x != kTfLiteOk) {                               \
    FATAL("Aborting since tflite returned failure."); \
  }

static TfLiteStatus PrepareInputs(tflite::Interpreter* interpreter) {
  TfLiteTensor* tensor = interpreter->tensor(interpreter->inputs()[0]);
  float* dst_data = interpreter->typed_tensor<float>(interpreter->inputs()[0]);
  if (!dst_data)
    return kTfLiteError;

  size_t num = tensor->bytes / sizeof(float);
  float* p = dst_data;
  for (p = dst_data; p < dst_data + num; p++) {
    *p = 0.0f;
  }
  return kTfLiteOk;
}

static TfLiteStatus ClearOutputs(tflite::Interpreter* interpreter) {
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  float* data = interpreter->typed_tensor<float>(interpreter->outputs()[0]);
  if (!data)
    return kTfLiteError;
  if (data) {
    size_t num = tensor->bytes / sizeof(float);
    for (float* p = data; p < data + num; p++) {
      *p = 0;
    }
  }
  return kTfLiteOk;
}

static TfLiteStatus PrintOutputs(tflite::Interpreter* interpreter) {
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  float* out_data = interpreter->typed_tensor<float>(interpreter->outputs()[0]);
  if (!out_data)
    return kTfLiteError;

  TfLiteStatus result = kTfLiteOk;
  size_t num = tensor->bytes / sizeof(float);
  for (size_t idx = 0; idx < num; idx++) {
    printf("out_data[%zu]: %f\n", idx, out_data[idx]);
  }
  return result;
}

TfLiteStatus Run(const char* filename, bool use_nnapi) {
  // Read tflite
  auto model = tflite::FlatBufferModel::BuildFromFile(filename);
  if (!model) FATAL("Cannot read file %s\n", filename);

  // Build interpreter
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver builtins;
  CHECK_TFLITE_SUCCESS(
      tflite::InterpreterBuilder(*model, builtins)(&interpreter));

  // Allocate tensors
  printf("Use nnapi is set to: %d\n", use_nnapi);
  interpreter->UseNNAPI(use_nnapi);
  interpreter->AllocateTensors();

  // Prepare inputs[0]
  PrepareInputs(interpreter.get());

  // Clear outputs[0]
  ClearOutputs(interpreter.get());

  // Invoke = Run
  interpreter->Invoke();

  // Compare outputs
  TfLiteStatus result = PrintOutputs(interpreter.get());
  printf("Running: %s\n", filename);
  printf("  Result: %s\n", (result == kTfLiteOk) ? "OK" : "FAILED");

  return result;
}

int main(int argc, char* argv[]) {
  string tflite_file = "";
  bool use_nnapi = true;
  std::vector<Flag> flag_list = {
	Flag("tflite_file", &tflite_file, "tflite filename to be invoked (Must)"),
    Flag("use_nnapi", &use_nnapi, "use nn api i.e. 0,1"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (tflite_file == "") {
    LOG(ERROR) << usage;
    return -1;
  }

  Run(tflite_file.c_str(), use_nnapi);

  return 0;
}
