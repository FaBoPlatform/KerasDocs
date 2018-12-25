# SSD

## Githubのプロジェクト
Keras 2.0で動作するように改良されたプロジェクト

https://github.com/SnowMasaya/ssd_keras

## Dataset
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

## weights_SSD300.hdf5
https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA

## 自作のデータ・セット
下記ツールでメタ画像のデータを作成

https://github.com/tzutalin/ImageNet_Utils

下記Scriptで、pkl形式のファイルを作成する

https://github.com/rykov8/ssd_keras/blob/master/PASCAL_VOC/get_data_from_XML.py

## SSD_training

下記を修正

> from ssd import SSD300  ->  from ssd_v2 import SSD300v2

> model = SSD300(input_shape, num_classes=NUM_CLASSES) -> model = SSD300v2(input_shape, num_classes=NUM_CLASSES)

