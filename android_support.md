
# 動作確認

|機種|MCU|OS|モデル|動作|
|:--|:--|:--|:--|:--|
|Nexus 5X|Snapdragon 808|8.1.0|ssd_mobilenet_v1_0.75_depth_quantized_coco|問題なく動作|
|Nexus 5X|Snapdragon 808|8.1.0|ssd_mobilenet_v1_ppn_coco|エラー1|

## エラー1

```shell
Cannot allocate memory for the interpreter: tensorflow/contrib/lite/kernels/conv.cc:260 real_multiplier < 1.0 was not true.Node 28 failed to prepare.
```