package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "cnpy",
    srcs = ["cnpy.cc"],
    hdrs = ["cnpy.h"],
)

cc_binary(
    name = "dump_tflite",
    srcs = ["dump_tflite.cc"],
    deps = [
        "//tensorflow/contrib/lite:framework",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "compare_tflite",
    srcs = ["compare_tflite.cc"],
    deps = [
        ":cnpy",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/nnapi:nnapi_lib",
	"//tensorflow/core:tensorflow",
    ],
    linkopts = [
        "-lm",
    ],
)

tf_cc_binary(
    name = "run_tflite",
    srcs = ["run_tflite.cc"],
    deps = [
        ":cnpy",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/nnapi:nnapi_lib",
	"//tensorflow/core:tensorflow",
    ],
    linkopts = [
        "-lm",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
