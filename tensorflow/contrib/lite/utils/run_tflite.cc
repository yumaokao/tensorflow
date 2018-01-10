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
#include "cnpy.h"
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

#if 0
static TfLiteStatus ReshapeInputs(tflite::Interpreter* interpreter,
                                  const char* batch_xs) {
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
  printf(" ReshapeInputs word_size=%zu\n", arr.word_size);
  float* src_data = arr.data<float>();
  if (!src_data)
    return kTfLiteError;

  std::vector<int> shape;
  for (size_t s : arr.shape) {
    shape.push_back(static_cast<int>(s));
  }

  int input = interpreter->inputs()[0];
  interpreter->ResizeInputTensor(input, shape);

  return kTfLiteOk;
}

static TfLiteStatus PrepareInputs(tflite::Interpreter* interpreter,
                                  const char* batch_xs) {
  TfLiteTensor* tensor = interpreter->tensor(interpreter->inputs()[0]);
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
  float* dst_data = interpreter->typed_tensor<float>(interpreter->inputs()[0]);
  float* src_data = arr.data<float>();
  if (!dst_data || !src_data)
    return kTfLiteError;

  size_t num = tensor->bytes / sizeof(float);
  float* p = dst_data;
  float* q = src_data;
  for (p = dst_data, q = src_data; p < dst_data + num; p++, q++) {
    *p = *q;
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

static TfLiteStatus SaveOutputs(tflite::Interpreter* interpreter,
                                   const char* batch_ys) {
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  float* out_data = interpreter->typed_tensor<float>(interpreter->outputs()[0]);
  if (!out_data)
    return kTfLiteError;

  TfLiteStatus result = kTfLiteOk;
  // get output shape
  std::vector<size_t> npyshape;
  // printf(" shape len=%d\n", tensor->dims->size);
  for (int i = 0; i < tensor->dims->size; i++) {
    npyshape.push_back(tensor->dims->data[i]);
    // printf(" %d\n", tensor->dims->data[i]);
  }

  std::vector<float> npydata;
  size_t num = tensor->bytes / sizeof(float);
  // printf(" num %zu\n", num);
  for (size_t idx = 0; idx < num; idx++) {
    npydata.push_back(out_data[idx]);
  }

  cnpy::npy_save(batch_ys, &npydata[0], npyshape, "w");
  return result;
}

TfLiteStatus Run(const char* filename, bool use_nnapi,
         const char* batch_xs, const char* batch_ys) {
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

  // Reshape with batch
  ReshapeInputs(interpreter.get(), batch_xs);

  // Allocate Tensors
  interpreter->AllocateTensors();

  // Clear outputs[0]
  ClearOutputs(interpreter.get());

  // Prepare inputs[0]
  PrepareInputs(interpreter.get(), batch_xs);

  // Invoke = Run
  interpreter->Invoke();

  // Compare outputs
  TfLiteStatus result = SaveOutputs(interpreter.get(), batch_ys);
  printf("Running: %s\n", filename);
  printf("  Result: %s\n", (result == kTfLiteOk) ? "OK" : "FAILED");

  return result;
}
#endif

class TFLiteRunner {
  public:
    TFLiteRunner(const string tflite_file, const bool use_nnapi)
      : m_tflite_file(tflite_file), m_use_nnapi(use_nnapi) {}
    TfLiteStatus Run(const string batch_xs, const string batch_ys);

  private:
    const string m_tflite_file;
    const bool m_use_nnapi;
    std::unique_ptr<tflite::Interpreter> m_interpreter;

    TfLiteStatus ReshapeInputs(const char* batch_xs);
    TfLiteStatus ClearOutputs();
    TfLiteStatus PrepareInputs(const char* batch_xs);
    TfLiteStatus SaveOutputs(const char* batch_ys);
};

TfLiteStatus TFLiteRunner::Run(const string batch_xs, const string batch_ys) {
  auto model = tflite::FlatBufferModel::BuildFromFile(m_tflite_file.c_str());
  tflite::ops::builtin::BuiltinOpResolver builtins;
  CHECK_TFLITE_SUCCESS(
      tflite::InterpreterBuilder(*model, builtins)(&m_interpreter));
  m_interpreter->UseNNAPI(m_use_nnapi);

  // Reshape with batch
  TF_LITE_ENSURE_STATUS(ReshapeInputs(batch_xs.c_str()));
  // Allocate Tensors
  TF_LITE_ENSURE_STATUS(m_interpreter->AllocateTensors());
  // Clear outputs[0]
  TF_LITE_ENSURE_STATUS(ClearOutputs());
  // Prepare inputs[0]
  TF_LITE_ENSURE_STATUS(PrepareInputs(batch_xs.c_str()));
  // Invoke = Run
  TF_LITE_ENSURE_STATUS(m_interpreter->Invoke());
  // Save outputs
  TF_LITE_ENSURE_STATUS(SaveOutputs(batch_ys.c_str()));
  return kTfLiteOk;
}

TfLiteStatus TFLiteRunner::ReshapeInputs(const char* batch_xs) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
  printf(" ReshapeInputs word_size=%zu\n", arr.word_size);
  float* src_data = arr.data<float>();
  if (!src_data)
    return kTfLiteError;

  std::vector<int> shape;
  for (size_t s : arr.shape) {
    shape.push_back(static_cast<int>(s));
  }
  int input = interpreter->inputs()[0];
  interpreter->ResizeInputTensor(input, shape);
  return kTfLiteOk;
}

TfLiteStatus TFLiteRunner::ClearOutputs() {
  tflite::Interpreter* interpreter = m_interpreter.get();
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

TfLiteStatus TFLiteRunner::PrepareInputs(const char* batch_xs) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteTensor* tensor = interpreter->tensor(interpreter->inputs()[0]);
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
  float* dst_data = interpreter->typed_tensor<float>(interpreter->inputs()[0]);
  float* src_data = arr.data<float>();
  if (!dst_data || !src_data)
    return kTfLiteError;

  size_t num = tensor->bytes / sizeof(float);
  float* p = dst_data;
  float* q = src_data;
  for (p = dst_data, q = src_data; p < dst_data + num; p++, q++) {
    *p = *q;
  }
  return kTfLiteOk;
}

TfLiteStatus TFLiteRunner::SaveOutputs(const char* batch_ys) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  float* out_data = interpreter->typed_tensor<float>(interpreter->outputs()[0]);
  if (!out_data)
    return kTfLiteError;

  TfLiteStatus result = kTfLiteOk;
  // get output shape
  std::vector<size_t> npyshape;
  // printf(" shape len=%d\n", tensor->dims->size);
  for (int i = 0; i < tensor->dims->size; i++) {
    npyshape.push_back(tensor->dims->data[i]);
    // printf(" %d\n", tensor->dims->data[i]);
  }

  std::vector<float> npydata;
  size_t num = tensor->bytes / sizeof(float);
  // printf(" num %zu\n", num);
  for (size_t idx = 0; idx < num; idx++) {
    npydata.push_back(out_data[idx]);
  }

  cnpy::npy_save(batch_ys, &npydata[0], npyshape, "w");
  return result;
}

int main(int argc, char* argv[]) {
  string tflite_file = "";
  string batch_xs = "";
  string batch_ys = "";
  bool use_nnapi = true;
  std::vector<Flag> flag_list = {
	Flag("tflite_file", &tflite_file, "tflite filename to be invoked (Must)"),
	Flag("batch_xs", &batch_xs, "batch_xs npy file to be set as inputs (Must)"),
	Flag("batch_ys", &batch_ys, "batch_xy npy file to be saved as outputs (Must)"),
    Flag("use_nnapi", &use_nnapi, "use nn api i.e. 0,1"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (tflite_file == "" || batch_xs == "") {
    LOG(ERROR) << usage;
    return -1;
  }

  TFLiteRunner runner(tflite_file, use_nnapi);
  TfLiteStatus result = runner.Run(batch_xs, batch_ys);

  // Run(tflite_file.c_str(), use_nnapi, batch_xs.c_str(), batch_ys.c_str());

  return 0;
}
