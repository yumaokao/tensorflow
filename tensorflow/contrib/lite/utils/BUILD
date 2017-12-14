package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "dump_tflite",
    srcs = ["dump_tflite.cc"],
    deps = [
        "//tensorflow/contrib/lite:framework",
    ],
)

cc_binary(
    name = "run_tflite",
    srcs = ["run_tflite.cc"],
    deps = [
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/nnapi:nnapi_lib",
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
