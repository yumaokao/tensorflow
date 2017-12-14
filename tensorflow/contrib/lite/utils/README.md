# Tensorflow Lite Utilities

## Build dump_tflite
```sh
$ bazel build //tensorflow/contrib/lite/utils:dump_tflite
```

## Build and run dump_tflite
```sh
$ bazel run //tensorflow/contrib/lite/utils:dump_tflite /home/tflite/sandbox/mnist/fc/export/mnist.lite
```

## compare_tflite
```sh
# generate mnist.lite, batch_xs.npy, ys.npy
$ cd /home/tflite/sandbox/mnist
$ make fc

# build and run compare_tflite
$ bazel build //tensorflow/contrib/lite/utils:compare_tflite \
  && bazel-bin/tensorflow/contrib/lite/utils/compare_tflite \
  --tflite_file=/home/tflite/sandbox/mnist/fc/export/mnist.lite \
  --batch_xs=/home/tflite/sandbox/mnist/fc/export/batch_xs.npy \
  --batch_ys=/home/tflite/sandbox/mnist/fc/export/ys.npy
```
