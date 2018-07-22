# TensorFlowの設定

## Gitのインストール

docker内で実行
```shell
$ apt-get install git
```

## TensorFlowとModelのプロジェクトをClone

docker内で実行
```shell
$ git clone https://github.com/tensorflow/tensorflow
```

docker内で実行
```shell
$ git clone https://github.com/tensorflow/model
```

## Object Detectionで必要なパッケージの設定

docker内で実行
```shell
$ pip install --user Cython
$ pip install --user contextlib2
$ pip install --user pillow
$ pip install --user lxml
``` 

## COCO APIのインストール

docker内で実行
```shell
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ make
$ cp -r pycocotools ../../models/research/
```

## Protobufのインストール

docker内で実行
```shell
$ curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
$ unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
$ mv protoc3/bin/* /usr/bin/
$ mv protoc3/include/* /usr/include/
```

## Protobufの編集

docker内で実行
```shell
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ make
$ cp -r pycocotools ../../models/research/
```

docker内で実行
```shell
$ cd models/reasearch
$ protoc object_detection/protos/*.proto --python_out=.
``` 

docker内で実行
```shell
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## model_builder_test.pyの実行

docker内で実行
```shell
$ python object_detection/builders/model_builder_test.py
```

```
/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters

..................
----------------------------------------------------------------------
Ran 18 tests in 0.084s

OK
```
