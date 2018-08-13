
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

