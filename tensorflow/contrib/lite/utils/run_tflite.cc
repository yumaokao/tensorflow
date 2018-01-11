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
// #define NDEBUG
#include <cassert>

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

template <typename T>
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

template <typename T>
TfLiteStatus TFLiteRunner<T>::Run(const string batch_xs, const string batch_ys) {
  auto model = tflite::FlatBufferModel::BuildFromFile(m_tflite_file.c_str());
  tflite::ops::builtin::BuiltinOpResolver builtins;
  CHECK_TFLITE_SUCCESS(
      tflite::InterpreterBuilder(*model, builtins)(&m_interpreter));
  m_interpreter->UseNNAPI(m_use_nnapi);

  // Reshape with batch
  TF_LITE_ENSURE_STATUS(ReshapeInputs(batch_xs.c_str()));
  // printf("ReshapeInputs\n");
  // Allocate Tensors
  TF_LITE_ENSURE_STATUS(m_interpreter->AllocateTensors());
  // printf("AllocateTensors\n");
  // Clear outputs[0]
  TF_LITE_ENSURE_STATUS(ClearOutputs());
  // printf("ClearOutputs\n");
  // Prepare inputs[0]
  TF_LITE_ENSURE_STATUS(PrepareInputs(batch_xs.c_str()));
  // printf("PrepareInputs\n");
  // Invoke = Run
  TF_LITE_ENSURE_STATUS(m_interpreter->Invoke());
  // printf("Invoke\n");
  // Save outputs
  TF_LITE_ENSURE_STATUS(SaveOutputs(batch_ys.c_str()));
  // printf("SaveOutputs\n");
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::ReshapeInputs(const char* batch_xs) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
  // TODO(yumaokao): assert doesn't work
  assert(arr.word_size == sizeof(T));
  if (arr.word_size != sizeof(T)) {
    LOG(ERROR) << "Input npy work_size " << arr.word_size << "!= "
               << " sizeof(T) " << sizeof(T) << std::endl;
    return kTfLiteError;
  }
  T* src_data = arr.data<T>();
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

template <typename T>
TfLiteStatus TFLiteRunner<T>::ClearOutputs() {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  T* data = interpreter->typed_tensor<T>(interpreter->outputs()[0]);
  if (!data)
    return kTfLiteError;
  if (data) {
    size_t num = tensor->bytes / sizeof(T);
    for (T* p = data; p < data + num; p++) {
      *p = 0;
    }
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::PrepareInputs(const char* batch_xs) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteTensor* tensor = interpreter->tensor(interpreter->inputs()[0]);
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
  T* dst_data = interpreter->typed_tensor<T>(interpreter->inputs()[0]);
  T* src_data = arr.data<T>();
  if (!dst_data || !src_data)
    return kTfLiteError;

  size_t num = tensor->bytes / sizeof(T);
  T* p = dst_data;
  T* q = src_data;
  for (p = dst_data, q = src_data; p < dst_data + num; p++, q++) {
    *p = *q;
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::SaveOutputs(const char* batch_ys) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  T* out_data = interpreter->typed_tensor<T>(interpreter->outputs()[0]);
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

  std::vector<T> npydata;
  size_t num = tensor->bytes / sizeof(T);
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
  string inference_type = "float";
  std::vector<Flag> flag_list = {
	Flag("tflite_file", &tflite_file, "tflite filename to be invoked (Must)"),
	Flag("batch_xs", &batch_xs, "batch_xs npy file to be set as inputs (Must)"),
	Flag("batch_ys", &batch_ys, "batch_xy npy file to be saved as outputs (Must)"),
    Flag("use_nnapi", &use_nnapi, "use nn api i.e. 0,1"),
    Flag("inference_type", &inference_type, "inference type: float, uint8"),
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

  // LOG(INFO) << inference_type << std::endl;
  // TODO(yumaokao); base class
  TfLiteStatus result = kTfLiteError;
  if (inference_type == "float") {
    TFLiteRunner<float> runner(tflite_file, use_nnapi);
    result = runner.Run(batch_xs, batch_ys);
  } else if (inference_type == "uint8") {
    TFLiteRunner<uint8_t> runner(tflite_file, use_nnapi);
    result = runner.Run(batch_xs, batch_ys);
  }

  return result;
}
