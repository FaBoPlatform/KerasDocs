
# モデル

```shell
$ cd /tmp
$ curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
$ tar xzf ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
$ gsutil cp /tmp/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/```

# TPUで学習

```shell
$ export CHECKPOINT_PATH=gs://${YOUR_GCS_BUCKET}/train1/model.ckpt-5000
```


```shell
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` --job-dir=gs://${YOUR_GCS_BUCKET}/train1 --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz --module-name object_detection.model_tpu_main --runtime-version 1.8 --scale-tier BASIC_TPU --region us-central1 -- --model_dir=gs://${YOUR_GCS_BUCKET}/train1 --tpu_zone us-central1 --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```

# 修正

```shell
python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=$CONFIG_FILE --trained_checkpoint_prefix=$CHECKPOINT_PATH --output_directory=$OUTPUT_DIR --add_postprocessing_op=true
```

で

```shell
File "/notebooks/models/research/object_detection/builders/model_builder.py", line 171, in _build_ssd_feature_extractor
    if feature_extractor_config.HasField('fpn'):
ValueError: Unknown field fpn.
```

fpnというValueがUnknownとエラーがでる。171行目以下をコメントアウト。

object_detection/builders/model_builder.py
```shell
#if feature_extractor_config.HasField('fpn'):
  #  kwargs.update({
  #      'fpn_min_level': feature_extractor_config.fpn.min_level,
  #      'fpn_max_level': feature_extractor_config.fpn.max_level,
  #  })
```



```shell
File "/notebooks/models/research/object_detection/builders/model_builder.py", line 226, in _build_ssd_model
    weight_regression_loss_by_score = (ssd_config.weight_regression_loss_by_score)
AttributeError: 'Ssd' object has no attribute 'weight_regression_loss_by_score'
```

次に、weight_regression_loss_by_scoreという属性がないとでるので、227行目をNoneにする。

```shell
#weight_regression_loss_by_score = (ssd_config.weight_regression_loss_by_score)
  weight_regression_loss_by_score = None
```

```shell
File "/notebooks/models/research/object_detection/builders/model_builder.py", line 236, in _build_ssd_model
    if ssd_config.use_expected_classification_loss_under_sampling:
AttributeError: 'Ssd' object has no attribute 'use_expected_classification_loss_under_sampling'
```

次に、use_expected_classification_loss_under_samplingという属性がないとでるので、236行目以下をコメントアウトする。

```shell
#if ssd_config.use_expected_classification_loss_under_sampling:
  #  expected_classification_loss_under_sampling = functools.partial(
  #      ops.expected_classification_loss_under_sampling,
  #      minimum_negative_sampling=ssd_config.minimum_negative_sampling,
  #      desired_negative_sampling_ratio=ssd_config.
  #      desired_negative_sampling_ratio)
```

# toco

```
 bazel run -c opt tensorflow/contrib/lite/toco:toco -- --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops --default_ranges_min=0 --default_ranges_max=6
```

# 後処理

```shell
$ git reset --hard
```
