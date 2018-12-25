# TPU

## TF Slim

docker内で実行
```shell
$ bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
$ python setup.py sdist
$ (cd slim && python setup.py sdist)
```

## トレーニングJob

--job-dirは、学習毎に違うフォルダを指定する(そうしないとエラー)。

docker内で実行
```shell
$ gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
--job-dir=gs://${YOUR_GCS_BUCKET}/train \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_tpu_main \
--runtime-version 1.8 \
--scale-tier BASIC_TPU \
--region us-central1 \
-- \
--model_dir=gs://${YOUR_GCS_BUCKET}/train \
--tpu_zone us-central1 \
--pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config
```

## 評価Job

docker内で実行
```shell
gcloud ml-engine jobs submit training `whoami`_object_detection_eval_validation_`date +%s` \
--job-dir=gs://${YOUR_GCS_BUCKET}/train \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_main \
--runtime-version 1.8 \
--scale-tier BASIC_GPU \
--region us-central1 \
-- \
--model_dir=gs://${YOUR_GCS_BUCKET}/train \
--pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/pipeline.config \
--checkpoint_dir=gs://${YOUR_GCS_BUCKET}/train
```

## TensorBoardで結果表示

docker内で実行
```shell
$ tensorboard --logdir=gs://${YOUR_GCS_BUCKET}/train
```




