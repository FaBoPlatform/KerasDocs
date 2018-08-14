# TPUをサポートしているモデル

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md


|Model名|Speed (ms)|	COCO mAP[^1]|config|
|:--|:--|:--|:--|:--|
|ssd_mobilenet_v1_0.75_depth_coco | 26 |	18	|[config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync.config)|
|ssd_mobilenet_v1_quantized_coco  | 29 |	18	|[config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_quantized_300x300_coco14_sync.config)|
|ssd_mobilenet_v1_0.75_depth_quantized_coco  | 29 | 16 |[config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config)|
|ssd_mobilenet_v1_ppn_coco | 26	|20	| [config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync.config) |
|ssd_mobilenet_v1_fpn_coco |	56	|32	| [config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync.config)
|ssd_resnet_50_fpn_coco | 76 | 35 | [config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config) |


