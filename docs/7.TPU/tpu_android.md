# TPU Android

## TOCO


```shell
$ export CONFIG_FILE=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
$ export CHECKPOINT_PATH=gs://${YOUR_GCS_BUCKET}/train/model.ckpt-2000
$ export OUTPUT_DIR=/tmp/tflite
```

```shell
python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```


```
/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
2018-08-14 22:23:32.219149: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Converted 387 variables to const ops.
2018-08-14 22:23:40.674300: I tensorflow/tools/graph_transforms/transform_graph.cc:264] Applying strip_unused_nodes
```

/tmp/tfliteに下記ファイルがダウンロードされる。

```shell
tflite_graph.pb  tflite_graph.pbtxt
```


## TensorFlow

> $ git clone https://github.com/tensorflow/tensorflow


## Copy

tensorflow/examples/android/のプロジェクトを使う。tensorflow/contrib/lite/examples/android/のDetectoreActivity.javaとTFLiteObjectDetectionAPIModel.javaを使用する。

tensorflowのディレクトリに移動

```shelll
$ cp tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java
$ cp tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/TFLiteObjectDetectionAPIModel.java tensorflow/examples/android/src/org/tensorflow/demo/TFLiteObjectDetectionAPIModel.java
```


## TOCO

DockerにBazelをインストール

```shell
$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
$ wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh
$ chmod +x bazel-0.16.0-installer-linux-x86_64.sh
$ ./bazel-<version>-installer-linux-x86_64.sh --user
```

```shall
$ export PATH="$PATH:$HOME/bin"
```


```shell
bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

```
INFO: Analysed target //tensorflow/contrib/lite/toco:toco (0 packages loaded).
INFO: Found 1 target...
Target //tensorflow/contrib/lite/toco:toco up-to-date:
  bazel-bin/tensorflow/contrib/lite/toco/toco
INFO: Elapsed time: 0.254s, Critical Path: 0.00s
INFO: 0 processes.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/tensorflow/contrib/lite/toco/toco '--input_file=/tmp/tflite/tflite_graph.pb' '--output_file=/tmp/tflite/detect.tflite' '--input_shapes=1,300,300,3' '--input_arrays=normalized_input_image_tensor' '--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' '--inference_type=QUANTIZED_UINT8' '--mean_values=128' '--std_values=128' '--change_concat_input_ranges=falINFO: Build completed successfully, 1 total action
2018-08-14 22:26:06.150620: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1055] Converting unsupported operation: TFLite_Detection_PostProcess
2018-08-14 22:26:06.177519: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before Removing unused ops: 900 operators, 1355 arrays (0 quantized)
2018-08-14 22:26:06.207086: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before general graph transformations: 900 operators, 1355 arrays (0 quantized)
2018-08-14 22:26:06.557619: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After general graph transformations pass 1: 112 operators, 224 arrays (1 quantized)
2018-08-14 22:26:06.560343: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before pre-quantization graph transformations: 112 operators, 224 arrays (1 quantized)
2018-08-14 22:26:06.561888: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After pre-quantization graph transformations pass 1: 65 operators, 177 arrays (1 quantized)
2018-08-14 22:26:06.563376: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before quantization graph transformations: 65 operators, 177 arrays (1 quantized)
2018-08-14 22:26:06.606566: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After quantization graph transformations pass 1: 71 operators, 183 arrays (151 quantized)
2018-08-14 22:26:06.610353: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After quantization graph transformations pass 2: 71 operators, 183 arrays (155 quantized)
2018-08-14 22:26:06.613796: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After quantization graph transformations pass 3: 66 operators, 178 arrays (157 quantized)
2018-08-14 22:26:06.617289: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After quantization graph transformations pass 4: 66 operators, 178 arrays (158 quantized)
2018-08-14 22:26:06.620144: W tensorflow/contrib/lite/toco/graph_transformations/quantize.cc:99] Constant array anchors lacks MinMax information. To make up for that, we will now compute the MinMax from actual array elements. That will result in quantization parameters that probably do not match whichever arithmetic was used during training, and thus will probably be a cause of poor inference accuracy.
2018-08-14 22:26:06.620393: W tensorflow/contrib/lite/toco/graph_transformations/quantize.cc:599] (Unsupported TensorFlow op: TFLite_Detection_PostProcess) is a quantized opbut it has a model flag that sets the output arrays to float.
2018-08-14 22:26:06.620444: W tensorflow/contrib/lite/toco/graph_transformations/quantize.cc:599] (Unsupported TensorFlow op: TFLite_Detection_PostProcess) is a quantized opbut it has a model flag that sets the output arrays to float.
2018-08-14 22:26:06.621551: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After quantization graph transformations pass 5: 64 operators, 176 arrays (159 quantized)
2018-08-14 22:26:06.622068: W tensorflow/contrib/lite/toco/graph_transformations/quantize.cc:599] (Unsupported TensorFlow op: TFLite_Detection_PostProcess) is a quantized opbut it has a model flag that sets the output arrays to float.
2018-08-14 22:26:06.625193: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before shuffling of FC weights: 64 operators, 176 arrays (159 quantized)
2018-08-14 22:26:06.626365: I tensorflow/contrib/lite/toco/allocate_transient_arrays.cc:329] Total transient array allocated size: 2160064 bytes, theoretical optimal value: 1620032 bytes.
2018-08-14 22:26:06.626757: I tensorflow/contrib/lite/toco/toco_tooling.cc:392] Estimated count of arithmetic ops: 1.36705 billion (note that a multiply-add is counted as 2 ops).
2018-08-14 22:26:06.627287: W tensorflow/contrib/lite/toco/tflite/operator.cc:1163] Ignoring unsupported type in list attribute with key '_output_types'
```


detect.tfline, pet_labels_list.txt を tensorflow/examples/android/assets にコピー。


## Android Studio

tensroflow/examples/android をAndroid Studioで開く

## Build.gradle

```
def nativeBuildSystem = 'none'
```

```
android {
	aaptOptions {
        noCompress 'tflite'
    }
}
```

```
dependencies {
	compile 'org.tensorflow:tensorflow-android:+'
    compile 'org.tensorflow:tensorflow-lite:+'
}
```
