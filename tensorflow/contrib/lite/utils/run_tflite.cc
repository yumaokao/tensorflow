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
    TfLiteStatus Run(const string batch_xs,
                     const string batch_ys,
                     const int output_tensor_idx,
                     const bool use_npz);

  private:
    const string m_tflite_file;
    const bool m_use_nnapi;
    std::unique_ptr<tflite::Interpreter> m_interpreter;

    TfLiteStatus ReshapeInputs(const char* batch_xs, const bool use_npz);
    TfLiteStatus ReshapeInput(const int tensor_id, cnpy::NpyArray array);
    TfLiteStatus ClearOutputs();
    TfLiteStatus PrepareInputs(const char* batch_xs, const bool use_npz);
    TfLiteStatus PrepareInput(const int tensor_id, cnpy::NpyArray array);
    TfLiteStatus SaveOutputs(const char* batch_ys, const bool use_npz);
};

template <typename T>
TfLiteStatus TFLiteRunner<T>::Run(const string batch_xs,
                                  const string batch_ys,
                                  const int output_tensor_idx,
                                  const bool use_npz) {
  auto model = tflite::FlatBufferModel::BuildFromFile(m_tflite_file.c_str());
  tflite::ops::builtin::BuiltinOpResolver builtins;
  CHECK_TFLITE_SUCCESS(
      tflite::InterpreterBuilder(*model, builtins)(&m_interpreter));
  m_interpreter->UseNNAPI(m_use_nnapi);
  if (output_tensor_idx != -1) {
    m_interpreter->SetOutputs({output_tensor_idx});
  }

  // Reshape with batch
  TF_LITE_ENSURE_STATUS(ReshapeInputs(batch_xs.c_str(), use_npz));
  // printf("ReshapeInputs\n");
  // Allocate Tensors
  TF_LITE_ENSURE_STATUS(m_interpreter->AllocateTensors());
  // printf("AllocateTensors\n");
  // Clear outputs[0]
  TF_LITE_ENSURE_STATUS(ClearOutputs());
  // printf("ClearOutputs\n");
  // Prepare inputs[0]
  TF_LITE_ENSURE_STATUS(PrepareInputs(batch_xs.c_str(), use_npz));
  // printf("PrepareInputs\n");
  // Invoke = Run
  m_interpreter->SetNumThreads(4);
  TF_LITE_ENSURE_STATUS(m_interpreter->Invoke());
  // printf("Invoke\n");
  // Save outputs
  TF_LITE_ENSURE_STATUS(SaveOutputs(batch_ys.c_str(), use_npz));
  // printf("SaveOutputs\n");
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::ReshapeInput(const int tensor_id,
                                           cnpy::NpyArray array) {
    tflite::Interpreter* interpreter = m_interpreter.get();
    // TODO(yumaokao): assert doesn't work
    assert(array.word_size == sizeof(T));
    if (array.word_size != sizeof(T)) {
      LOG(ERROR) << "Input array work_size " << array.word_size << "!= "
                 << " sizeof(T) " << sizeof(T) << std::endl;
      return kTfLiteError;
    }
    T* src_data = array.data<T>();
    if (!src_data)
      return kTfLiteError;

    std::vector<int> shape;
    for (size_t s : array.shape) {
      shape.push_back(static_cast<int>(s));
    }
    interpreter->ResizeInputTensor(tensor_id, shape);
    return kTfLiteOk;
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::ReshapeInputs(const char* batch_xs,
                                            const bool use_npz) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteStatus result = kTfLiteError;
  if (use_npz) {
    // check inputs size
    const std::vector<int> inputs = interpreter->inputs();
    cnpy::npz_t arrs = cnpy::npz_load(batch_xs);
    if (arrs.size() != inputs.size()) {
      LOG(ERROR) << "Input npz arrays size " << arrs.size() << "!= "
                 << " network inputs size " << inputs.size() << std::endl;
      return kTfLiteError;
    }

    for (int i = 0; i < inputs.size(); i++) {
      int tensor_id = inputs[i];
      const char *tensor_name = interpreter->GetInputName(i);
      if (arrs.find(tensor_name) == arrs.end()) {
        LOG(ERROR) << "Could not find input array name " << tensor_name
                   << " in npz arrays " << std::endl;
        return kTfLiteError;
      }
      result = ReshapeInput(tensor_id, arrs[tensor_name]);
      if (result != kTfLiteOk)
        return result;
    }
    return result;
  } else { // use npy, so default inputs.size() == 1
    cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
    int tensor_id = interpreter->inputs()[0];
    return ReshapeInput(tensor_id, arr);
  }
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::ClearOutputs() {
  tflite::Interpreter* interpreter = m_interpreter.get();
  const std::vector<int> outputs = interpreter->outputs();
  for (int i = 0; i < outputs.size(); i++) {
    TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[i]);
    T* data = interpreter->typed_tensor<T>(interpreter->outputs()[i]);
    if (!data)
      return kTfLiteError;
    if (data) {
      size_t num = tensor->bytes / sizeof(T);
      for (T* p = data; p < data + num; p++) {
        *p = 0;
      }
    }
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::PrepareInput(const int tensor_id,
                                           cnpy::NpyArray array) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteTensor* tensor = interpreter->tensor(tensor_id);
  T* dst_data = interpreter->typed_tensor<T>(tensor_id);
  T* src_data = array.data<T>();
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
TfLiteStatus TFLiteRunner<T>::PrepareInputs(const char* batch_xs, const bool use_npz) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteStatus result = kTfLiteError;
  if (use_npz) {
    // check inputs size
    const std::vector<int> inputs = interpreter->inputs();
    cnpy::npz_t arrs = cnpy::npz_load(batch_xs);
    if (arrs.size() != inputs.size()) {
      LOG(ERROR) << "Input npz arrays size " << arrs.size() << "!= "
                 << " network inputs size " << inputs.size() << std::endl;
      return kTfLiteError;
    }

    for (int i = 0; i < inputs.size(); i++) {
      int tensor_id = inputs[i];
      const char *tensor_name = interpreter->GetInputName(i);
      if (arrs.find(tensor_name) == arrs.end()) {
        LOG(ERROR) << "Could not find input array name " << tensor_name
                   << " in npz arrays " << std::endl;
        return kTfLiteError;
      }
      result = PrepareInput(tensor_id, arrs[tensor_name]);
      if (result != kTfLiteOk)
        return result;
    }
    return result;
  } else {
    cnpy::NpyArray arr = cnpy::npy_load(batch_xs);
    int tensor_id = interpreter->inputs()[0];
    return PrepareInput(tensor_id, arr);
  }
}

template <typename T>
TfLiteStatus TFLiteRunner<T>::SaveOutputs(const char* batch_ys, const bool use_npz) {
  tflite::Interpreter* interpreter = m_interpreter.get();
  TfLiteStatus result = kTfLiteError;
  if (use_npz) {
    const std::vector<int> outputs = interpreter->outputs();
    bool append = false;
    for (int o = 0; o < outputs.size(); o++) {
      int tensor_id = outputs[o];
      const char *tensor_name = interpreter->GetOutputName(o);
      // printf("YMK in SaveOutputs output %d: %s\n", tensor_id, tensor_name);

      TfLiteTensor* tensor = interpreter->tensor(tensor_id);
      T* out_data = interpreter->typed_tensor<T>(tensor_id);
      if (!out_data)
        return kTfLiteError;

      // get shape
      std::vector<size_t> npyshape;
      for (int i = 0; i < tensor->dims->size; i++) {
        npyshape.push_back(tensor->dims->data[i]);
      }

      // get data
      std::vector<T> npydata;
      size_t num = tensor->bytes / sizeof(T);
      // printf(" num %zu\n", num);
      for (size_t idx = 0; idx < num; idx++) {
        npydata.push_back(out_data[idx]);
      }

      // save npz
      cnpy::npz_save(batch_ys, tensor_name, &npydata[0], npyshape,
                     (append) ? "a" : "w");
      append = true;
      result = kTfLiteOk;
    }
    return result;
  } else {
    TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
    T* out_data = interpreter->typed_tensor<T>(interpreter->outputs()[0]);
    if (!out_data)
      return kTfLiteError;

    result = kTfLiteOk;
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
}

int main(int argc, char* argv[]) {
  string tflite_file = "";
  string batch_xs = "";
  string batch_ys = "";
  int output_tensor_idx = -1;
  bool use_nnapi = false;
  bool use_npz = false;
  string inference_type = "float";
  std::vector<Flag> flag_list = {
    Flag("tflite_file", &tflite_file, "tflite filename to be invoked (Must)"),
    Flag("batch_xs", &batch_xs, "batch_xs npy or npz file to be set as inputs (Must)"),
    Flag("batch_ys", &batch_ys, "batch_xy npy or npz file to be saved as outputs (Must)"),
    Flag("use_nnapi", &use_nnapi, "use nn api i.e. true/false"),
    Flag("use_npz", &use_npz, "use npz for inputs and outputs i.e. true/false"),
    Flag("output_tensor_idx", &output_tensor_idx, "index of the output tensor defined in the tflite model"),
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
    result = runner.Run(batch_xs, batch_ys, output_tensor_idx, use_npz);
  } else if (inference_type == "uint8") {
    TFLiteRunner<uint8_t> runner(tflite_file, use_nnapi);
    result = runner.Run(batch_xs, batch_ys, output_tensor_idx, use_npz);
  }

  return result;
}
