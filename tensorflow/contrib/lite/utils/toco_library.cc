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
#include <cstdio>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/toco_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/toco_tooling.h"
#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"

#ifndef CHECK_OK
#define CHECK_OK(val) CHECK_EQ((val).ok(), true)
#define QCHECK_OK(val) QCHECK_EQ((val).ok(), true)
#endif

#if 0
namespace toco {
namespace {

#define QCHECK_REQUIRE_TOCO_FLAG(arg) \
  QCHECK(parsed_toco_flags.arg.specified()) << "Missing required flag: " #arg;

void CheckFilePermissions(const ParsedTocoFlags& parsed_toco_flags,
                          const ParsedModelFlags& parsed_model_flags,
                          const TocoFlags& toco_flags) {
  port::CheckInitGoogleIsDone("InitGoogle is not done yet");

  QCHECK_REQUIRE_TOCO_FLAG(input_file)
  QCHECK_OK(port::file::Exists(parsed_toco_flags.input_file.value(),
                               port::file::Defaults()))
      << "Specified input_file does not exist: "
      << parsed_toco_flags.input_file.value();
  QCHECK_OK(port::file::Readable(parsed_toco_flags.input_file.value(),
                                 port::file::Defaults()))
      << "Specified input_file exists, but is not readable: "
      << parsed_toco_flags.input_file.value();

  QCHECK_REQUIRE_TOCO_FLAG(output_file);
  QCHECK_OK(port::file::Writable(parsed_toco_flags.output_file.value()))
      << "parsed_toco_flags.input_file.value() output_file is not writable: "
      << parsed_toco_flags.output_file.value();
}

void ToolMain(const ParsedTocoFlags& parsed_toco_flags,
              const ParsedModelFlags& parsed_model_flags) {
  ModelFlags model_flags;
  ReadModelFlagsFromCommandLineFlags(parsed_model_flags, &model_flags);

  TocoFlags toco_flags;
  ReadTocoFlagsFromCommandLineFlags(parsed_toco_flags, &toco_flags);

  CheckFilePermissions(parsed_toco_flags, parsed_model_flags, toco_flags);

  string input_file_contents;
  CHECK_OK(port::file::GetContents(parsed_toco_flags.input_file.value(),
                                   &input_file_contents,
                                   port::file::Defaults()));
  std::unique_ptr<Model> model =
      Import(toco_flags, model_flags, input_file_contents);
  Transform(toco_flags, model.get());
  string output_file_contents;
  Export(toco_flags, *model, toco_flags.allow_custom_ops(),
         &output_file_contents);
  CHECK_OK(port::file::SetContents(parsed_toco_flags.output_file.value(),
                                   output_file_contents,
                                   port::file::Defaults()));
}

}  // namespace
}  // namespace toco
#endif

namespace toco {
  void MakeGeneralGraphTransformationsSet(
    toco::GraphTransformationsSet* transformations) {
  CHECK(transformations->empty());
  transformations->Add(new ConvertExpandDimsToReshape);
  transformations->Add(new ConvertTrivialAddNToAdd);
  transformations->Add(new ConvertTrivialTransposeToReshape);
  transformations->Add(new ConvertReorderAxes);
  transformations->Add(new ResolveReshapeAttributes);
  transformations->Add(new PropagateArrayDataTypes);
  transformations->Add(new PropagateFixedSizes);
  transformations->Add(new RemoveTensorFlowAssert);
  transformations->Add(new RemoveTensorFlowIdentity);
  transformations->Add(new RemoveTrivialConcatenation);
  transformations->Add(new RemoveTrivialConcatenationInput);
  transformations->Add(new RemoveUnusedOp);
  transformations->Add(new EnsureBiasVectors);
  transformations->Add(new ResolveReorderAxes);
  transformations->Add(new ResolveTensorFlowMatMul);
  transformations->Add(new FuseBinaryIntoPrecedingAffine);
  transformations->Add(new FuseBinaryIntoFollowingAffine);
  transformations->Add(new ReorderActivationFunctions);
  transformations->Add(new ResolveBatchNormalization);
  transformations->Add(new ResolveConstantBinaryOperator);
  transformations->Add(new ResolveConstantFill);
  transformations->Add(new ResolveConstantRange);
  transformations->Add(new ResolveConstantStack);
  transformations->Add(new ResolveConstantStridedSlice);
  transformations->Add(new ResolveConstantUnaryOperator);
  transformations->Add(new ResolveTensorFlowMerge);
  transformations->Add(new ResolveSqueezeAttributes);
  transformations->Add(new ResolveTensorFlowSwitch);
  transformations->Add(new ResolveTensorFlowTile);
  transformations->Add(new ResolveTensorFlowConcat);
  transformations->Add(new IdentifyL2Normalization);
  transformations->Add(new IdentifyL2Pool);
  transformations->Add(new IdentifyRelu1);
  transformations->Add(new RemoveTrivialBinaryOperator);
  transformations->Add(new ReadFakeQuantMinMax);
  transformations->Add(new ResolveSpaceToBatchNDAttributes);
  transformations->Add(new ResolveBatchToSpaceNDAttributes);
  transformations->Add(new ResolvePadAttributes);
  transformations->Add(new ResolveStridedSliceAttributes);
  transformations->Add(new ResolveSliceAttributes);
  transformations->Add(new ResolveMeanAttributes);
  transformations->Add(new ResolveTransposeAttributes);
  transformations->Add(new ResolveConstantShapeOrRank);
  transformations->Add(new MakeInitialDequantizeOperator);
  transformations->Add(new ResolvePRelu);
  }
}  // namespace toco

int version() {
  std::cout << "in version" << std::endl;
}

int testGraphTransformationsSet() {
  std::cout << "in testGraphTransformationsSet" << std::endl;
  toco::GraphTransformationsSet transformations;
  toco::MakeGeneralGraphTransformationsSet(&transformations);
  for (const auto& transformation : transformations) {
    const std::string& name = transformation->Name();
    std::cout << "    " << name << std::endl;
  }
  std::cout << "in testGraphTransformationsSet done" << std::endl;
}
