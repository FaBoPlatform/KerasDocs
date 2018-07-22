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

## TensorFlow

> $ git clone https://github.com/tensorflow/tensorflow


## Copy

> $ cp tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java
> $ cp tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/TFLiteObjectDetectionAPIModel.java tensorflow/examples/android/src/org/tensorflow/demo/TFLiteObjectDetectionAPIModel.java


## Android Studio

tensroflow/example/android

を開く

## Build.gradle

```
def nativeBuildSystem = 'none'
```

```
dependencies {
    compile 'org.tensorflow:tensorflow-android:+'
    compile 'org.tensorflow:tensorflow-lite:+'
}
```
