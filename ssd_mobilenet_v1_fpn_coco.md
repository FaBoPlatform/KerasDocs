
# Git clone直後の場合　

COCO APIのインストール

```shell
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ make
$ cp -r pycocotools ../../models/research/
```

```shell
$ cd models/reasearch
$ protoc object_detection/protos/*.proto --python_out=.
docker内で実行
```

```shell
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

# モデル

```shell
$ cd /tmp
$ curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
$ tar xzf ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
$ gsutil cp /tmp/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/```

# TPUで学習

```shell
$ vim object_detection/samples/configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config
$ gsutil cp object_detection/samples/configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```

```shell
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` --job-dir=gs://${YOUR_GCS_BUCKET}/train_fpn --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz --module-name object_detection.model_tpu_main --runtime-version 1.8 --scale-tier BASIC_TPU --region us-central1 -- --model_dir=gs://${YOUR_GCS_BUCKET}/train_fpn --tpu_zone us-central1 --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```

# 取得

```shell
$ export CONFIG_FILE=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
$ export CHECKPOINT_PATH=gs://${YOUR_GCS_BUCKET}/train_fpn/model.ckpt-2000
$ export OUTPUT_DIR=/tmp/tflite
```


```shell
python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=$CONFIG_FILE --trained_checkpoint_prefix=$CHECKPOINT_PATH --output_directory=$OUTPUT_DIR --add_postprocessing_op=true
```

# toco

```
 bazel run -c opt tensorflow/contrib/lite/toco:toco -- --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops --default_ranges_min=0 --default_ranges_max=6
```

# 後処理

```shell
$ git reset --hard
```
