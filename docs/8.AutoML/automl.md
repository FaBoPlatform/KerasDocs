# AutoML

AutoMLは、簡単な処理で独自のモデルが作成できます。現在、[Vision](https://beta-dot-custom-vision.appspot.com/vision/overview), [Translate](https://beta-dot-custom-vision.appspot.com/translation/overview), [Natural Language](https://beta-dot-custom-vision.appspot.com/text/overview)の3つがBeta公開されています。

# AutoML Visionの準備

AutoML Visionを使用するには、料金支払の設定と、APIの有効化が必要です。

[AutoML用プロジェクトの作成](https://console.cloud.google.com/cloud-resource-manager?hl=ja&_ga=2.165375740.-2107046452.1532174868)

[プロジェクトの支払いの有効化](https://console.cloud.google.com/billing)

[AutoML関連 APIの有効化](https://console.cloud.google.com/flows/enableapi?apiid=storage-component.googleapis.com,automl.googleapis.com,storage-api.googleapis.com&hl=ja&_ga=2.165375740.-2107046452.1532174868)

# Datasetの新規作成

![](./img/automl001.png)

![](./img/automl002.png)

![](./img/automl003.png)

# Imageのアップロード

[Aizu Dataset](https://github.com/FaBoPlatform/KerasDocs/raw/master/dataset/aizu_dataset.zip) をダウンロードします。中に、赤べこと起き上がり小法師の画像がはいっています。

![](./img/automl004.png)

# Label作成

![](./img/automl005.png)

![](./img/automl006.png)

![](./img/automl007.png)

![](./img/automl008.png)

# ImageへのLabelづけ

![](./img/automl009.png)

![](./img/automl010.png)

![](./img/automl011.png)

# 学習

![](./img/automl012.png)

![](./img/automl013.png)

# 評価

![](./img/automl014.png)

![](./img/automl015.png)





