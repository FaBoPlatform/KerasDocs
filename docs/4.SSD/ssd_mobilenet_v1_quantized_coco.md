 Git clone直後の場合　

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
$ curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz
$ tar xzf ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz
$ gsutil cp /tmp/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/
```

# TPUで学習

```shell
$ vim object_detection/samples/configs/ssd_mobilenet_v1_quantized_300x300_coco14_sync.config
$ gsutil cp object_detection/samples/configs/ssd_mobilenet_v1_quantized_300x300_coco14_sync.config gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```


```shell
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` --job-dir=gs://${YOUR_GCS_BUCKET}/train_quantized  --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz --module-name object_detection.model_tpu_main --runtime-version 1.8 --scale-tier BASIC_TPU --region us-central1 -- --model_dir=gs://${YOUR_GCS_BUCKET}/train_quantized --tpu_zone us-central1 --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```

# エラー

```shell
master-replica-0
Traceback (most recent call last): File "/usr/lib/python2.7/runpy.py", line 174, in _run_module_as_main "__main__", fname, loader, pkg_name) File "/usr/lib/python2.7/runpy.py", line 72, in _run_code exec code in run_globals File "/root/.local/lib/python2.7/site-packages/object_detection/model_tpu_main.py", line 134, in <module> tf.app.run() File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 126, in run _sys.exit(main(argv)) File "/root/.local/lib/python2.7/site-packages/object_detection/model_tpu_main.py", line 119, in main estimator.train(input_fn=train_input_fn, max_steps=train_steps) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 363, in train loss = self._train_model(input_fn, hooks, saving_listeners) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 843, in _train_model return self._train_model_default(input_fn, hooks, saving_listeners) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 856, in _train_model_default features, labels, model_fn_lib.ModeKeys.TRAIN, self.config) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 831, in _call_model_fn model_fn_results = self._model_fn(features=features, **kwargs) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2036, in _model_fn _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn)) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2244, in _train_on_tpu_system device_assignment=ctx.device_assignment) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu.py", line 690, in shard name=name) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu.py", line 516, in replicate outputs = computation(*computation_inputs) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 2237, in multi_tpu_train_steps_on_single_shard single_tpu_train_step, [_INITIAL_LOSS]) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/training_loop.py", line 207, in repeat cond, body_wrapper, inputs=inputs, infeed_queue=infeed_queue, name=name) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/training_loop.py", line 169, in while_loop name="") File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/control_flow_ops.py", line 3224, in while_loop result = loop_context.BuildLoop(cond, body, loop_vars, shape_invariants) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/control_flow_ops.py", line 2956, in BuildLoop pred, body, original_loop_vars, loop_vars, shape_invariants) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/control_flow_ops.py", line 2893, in _BuildLoop body_result = body(*packed_vars_for_body) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/training_loop.py", line 120, in body_wrapper outputs = body(*(inputs + dequeue_ops)) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/training_loop.py", line 203, in body_wrapper return [i + 1] + _convert_to_list(body(*args)) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 1156, in train_step self._call_model_fn(features, labels)) File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/tpu/python/tpu/tpu_estimator.py", line 1317, in _call_model_fn estimator_spec = self._model_fn(features=features, **kwargs) File "/root/.local/lib/python2.7/site-packages/object_detection/model_lib.py", line 252, in model_fn preprocessed_images, features[fields.InputDataFields.true_image_shape]) File "/root/.local/lib/python2.7/site-packages/object_detection/meta_architectures/ssd_meta_arch.py", line 514, in predict preprocessed_inputs) File "/root/.local/lib/python2.7/site-packages/object_detection/models/ssd_mobilenet_v1_fpn_feature_extractor.py", line 146, in extract_features depth=depth_fn(256)) File "/root/.local/lib/python2.7/site-packages/object_detection/models/feature_map_generators.py", line 218, in fpn_top_down_feature_maps top_down += residual File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/math_ops.py", line 979, in binary_op_wrapper return func(x, y, name=name) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_math_ops.py", line 297, in add "Add", x=x, y=y, name=name) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper op_def=op_def) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 3392, in create_op op_def=op_def) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1734, in __init__ control_input_ops) File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1570, in _create_c_op raise ValueError(str(e)) ValueError: Dimensions must be equal, but are 20 and 19 for 'FeatureExtractor/MobilenetV1/fpn/top_down/add' (op: 'Add') with input shapes: [8,20,20,256], [8,19,19,256].
```

# 取得

```shell
$ export CONFIG_FILE=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
$ export CHECKPOINT_PATH=gs://${YOUR_GCS_BUCKET}/train_resnet/model.ckpt-2000
$ export OUTPUT_DIR=/tmp/tflite
```

```shell
$ python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=$CONFIG_FILE --trained_checkpoint_prefix=$CHECKPOINT_PATH --output_directory=$OUTPUT_DIR --add_postprocessing_op=true
```


# toco

```
 bazel run -c opt tensorflow/contrib/lite/toco:toco -- --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops --default_ranges_min=0 --default_ranges_max=6
```

# 後処理

```shell
$ git reset --hard
```
