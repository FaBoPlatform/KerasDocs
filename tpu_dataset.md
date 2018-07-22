# Dataset

## データ・セットの用意

docker内で実行
```shell
$ mkdir /tmp/pet_faces_tfrecord/
$ cd /tmp/pet_faces_tfrecord/
$ curl "http://download.tensorflow.org/models/object_detection/pet_faces_tfrecord.tar.gz" | tar xzf -
```

## DatasetをGCSにアップ

docker内で実行
```shell
$ gsutil -m cp -r /tmp/pet_faces_tfrecord/pet_faces* gs://${YOUR_GCS_BUCKET}/data/
```

## pet_label_map.pbtxtをGCSにアップ

docker内で実行
```
$ cd /notebooks
$ cd models/research
$ gsutil cp object_detection/data/pet_label_map.pbtxt gs://${YOUR_GCS_BUCKET}/data/pet_label_map.pbtxt
```

## SSD MobileNet checkpointをGCSにアップ

docker内で実行
```
$ cd /tmp
$ curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
$ tar xzf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
$ gsutil cp /tmp/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/
```

