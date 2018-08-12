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


/tmp/tfliteに下記ファイルがダウンロードされる。

```shell
tflite_graph.pb  tflite_graph.pbtxt
```


/notebooks以下にコピーして、Jupyterを経由してローカルにダウンロードする。

```shell
$ cp /tmp/tflite/* /notebooks
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
