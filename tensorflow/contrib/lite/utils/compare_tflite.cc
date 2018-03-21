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

template<typename T>
static TfLiteStatus ReshapeInputs(tflite::Interpreter* interpreter,
                                  const char* batch_xs, T type) {
  cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
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

template<typename T>
static TfLiteStatus PrepareInputs(tflite::Interpreter* interpreter,
                                  const char* batch_xs, T type) {
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

template<typename T>
static TfLiteStatus ClearOutputs(tflite::Interpreter* interpreter, T type) {
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

static TfLiteStatus CompareOutputs_UINT8(tflite::Interpreter* interpreter,
                                   const char* batch_ys, bool ignore) {
  constexpr int kAbsoluteThreshold = 2; // 1e-4f;
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  cnpy::NpyArray arr = cnpy::npy_load(batch_ys);
  uint8_t* out_data = interpreter->typed_tensor<uint8_t>(interpreter->outputs()[0]);
  uint8_t* ref_data = arr.data<uint8_t>();
  if (!out_data) printf("no out_data\n");
  if (!ref_data) printf("no ref_data\n");
  if (!out_data || !ref_data)
    return kTfLiteError;

  TfLiteStatus result = kTfLiteOk;
  size_t num = tensor->bytes / sizeof(uint8_t);
  int max_diff = 0;
  int err_cnt = 0;
  int minus_cnt = 0;
  for (size_t idx = 0; idx < num; idx++) {
    uint8_t computed = out_data[idx];
    uint8_t reference = ref_data[idx];
    int diff = std::abs((int)computed - (int)reference);
    bool error_is_large = false;
    max_diff = std::max(diff, max_diff);

    error_is_large = (diff >= kAbsoluteThreshold);
    if (error_is_large) {
      fprintf(stdout, "output[%d][%zu] did not match %hhu vs reference %hhu\n",
              0, idx, computed, reference);
      result = kTfLiteError;
      err_cnt += 1;
      if (ignore == false)
        break;
    } else if (diff == 1){
      minus_cnt += 1;
    }
  }
  printf("max diff: %d, err_cnt: (%d/%d), diff_1_cnt: (%d/%d)\n", max_diff, err_cnt,(int)num, minus_cnt, (int)num);
  return result;
}

static TfLiteStatus CompareOutputs_FLOAT(tflite::Interpreter* interpreter,
                                   const char* batch_ys, bool ignore) {
  constexpr double kRelativeThreshold = 1e-2f;
  constexpr double kAbsoluteThreshold = 1e-4f;
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  cnpy::NpyArray arr = cnpy::npy_load(batch_ys);
  float* out_data = interpreter->typed_tensor<float>(interpreter->outputs()[0]);
  float* ref_data = arr.data<float>();
  if (!out_data || !ref_data)
    return kTfLiteError;

  TfLiteStatus result = kTfLiteOk;
  size_t num = tensor->bytes / sizeof(float);
  for (size_t idx = 0; idx < num; idx++) {
    float computed = out_data[idx];
    float reference = ref_data[idx];
    float diff = std::abs(computed - reference);
    bool error_is_large = false;
    if (std::abs(reference) < kRelativeThreshold) {
      error_is_large = (diff > kAbsoluteThreshold);
    } else {
      error_is_large = (diff > kRelativeThreshold * std::abs(reference));
    }
    if (error_is_large) {
      fprintf(stdout, "output[%d][%zu] did not match %f vs reference %f\n",
              0, idx, computed, reference);
      result = kTfLiteError;
      if (ignore == false)
        break;
    }
  }
  return result;
}

TfLiteStatus Compare(const char* filename, bool use_nnapi,
         const char* batch_xs, const char* batch_ys, bool ignore, string infer_type, string input_type) {
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
  if (input_type == "UINT8") {
      ReshapeInputs(interpreter.get(), batch_xs, (uint8_t)1);
  } else {
      ReshapeInputs(interpreter.get(), batch_xs, (float)1.0);
  }
  // Allocate Tensors
  interpreter->AllocateTensors();

  // Clear outputs[0]
  if (infer_type == "UINT8") {
    ClearOutputs(interpreter.get(), (uint8_t)1);
  } else {
    ClearOutputs(interpreter.get(), (float)1.0);
  }

  // Prepare inputs[0]
  if (input_type == "UINT8") {
    PrepareInputs(interpreter.get(), batch_xs, (uint8_t)1);
  } else {
    PrepareInputs(interpreter.get(), batch_xs, (float)1.0);
  }
  // Invoke = Run
  interpreter->Invoke();
  // std::cout << "=== infer type: " << infer_type << ", input type: " << input_type << ", ys :" << batch_ys << std::endl;
  // Compare outputs
  TfLiteStatus result;
  if (infer_type == "UINT8") {
    result = CompareOutputs_UINT8(interpreter.get(), batch_ys, ignore);
  } else {
    result = CompareOutputs_FLOAT(interpreter.get(), batch_ys, ignore);
  }
  printf("Running: %s\n", filename);
  printf("  Result: %s\n", (result == kTfLiteOk) ? "OK" : "FAILED");

  return result;
}

int main(int argc, char* argv[]) {
  string tflite_file = "";
  string batch_xs = "";
  string batch_ys = "";
  string inference_type = "";
  string input_type = "";
  bool use_nnapi = true;
  bool ignore = false;
  std::vector<Flag> flag_list = {
	Flag("tflite_file", &tflite_file, "tflite filename to be invoked (Must)"),
	Flag("batch_xs", &batch_xs, "batch_xs npy file to be set as inputs (Must)"),
	Flag("batch_ys", &batch_ys, "batch_xy npy file to be compared with outputs (Must)"),
    Flag("use_nnapi", &use_nnapi, "use nn api i.e. 0,1"),
    Flag("inference_type", &inference_type, "use FLOAT or UINT8 comparison as inference type"),
    Flag("input_type", &input_type, "use FLOAT or UINT8 comparison as input type"),
    Flag("ignore", &ignore, "ignore error to continue compare all, 0,1"),
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
  Compare(tflite_file.c_str(), use_nnapi, batch_xs.c_str(), batch_ys.c_str(), ignore, inference_type, input_type);
  return 0;
}
