# hand_sign_recognition
MediaPipeを用いてハンドサインを検出するプログラムです.   

![sample](https://github.com/kou1046/hand_sign_recognition/assets/84436650/af7d28d0-fa3a-45e6-bb25-edca6a4b6d70)  

自分でカスタマイズしたハンドサインを認識し，自分のプロジェクトに組み込むことが出来ます．

## Requirements
* mediapipe ^0.10.2
* pytorch ^2.0.1
* ndjson ^0.3.1
* dacite ^1.8.1
* opencv-python ^4.8.0.74
* scikit-learn ^1.3.0

## Install
プロジェクト直下まで移動し，以下のどちらかを実行してください．
### pip
```bash
pip install -r requirements.txt
```
### poetry
```bash
pip install poetry
poetry install
```
こちらはプロジェクト内に仮想環境が作成されます．



## Quick start
デモはexamplesディレクトリに配置してあります．
### examples/app.py
ビデオカメラを用いた，推論用のサンプルスクリプトです．デフォルトでは FIST（✊）, LIKE（👍）のハンドサインを学習してあります．
### examples/train_classifier.py
学習させるときに用いるスクリプトです．起動すると学習データを集める画面が表示されます．  
0 ~ 9のキーを押すと，データ記録モードとなり，現在移っている手の座標と，押下したキーの数字をラベルとして，指定されたファイル(デフォルトではmodels/labeled_hand_sign_dataset.csv)に追記されていきます．
dキーを押すとディスプレイモードとなり，データの追記を中断できます．
必要に応じて，既存データを消去したり，更にデータを追記してみてください．また，0から順番にハンドサインのラベル付けをする必要があることに注意してください．

ある程度データが集ったら，ESCキーを押下してください．取得したデータを用いて学習が始まります．学習させた重みは指定した場所（デフォルトでは models/hand_sign_classifier_weights.pth）に保存されます．

## Application Example
出来る限りモジュール化しているので，自分のプロジェクトに組み込むことが出来ます．

#### HandDetector (lib/detectors/hand_detector.py)
mediapipeの手検出機能を，より簡潔に記述できるようにしたラップオブジェクトです．
detectメソッドにRGB画像を渡すと，検出した手の座標 (HandPointsオブジェクト)を返します．
```python
import cv2
from lib import HandDetector, HandSignClassifier

cap = cv2.VideoCapture(0)
detector = HandDetector()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    handpoints: HandPoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(handpoints)
```

#### HandSignClassifier (lib/classifiers/hand_sign_classifier.py)
学習させたハンドサインの推論をするオブジェクトです．引数に出力ラベルの個数が必要になることに注意してください．
また，推論時は引数に学習済みの重みファイルのパスを与える必要があります．
predictメソッドで, HandPointsオブジェクトからハンドサインのラベルを取得できます．
```python
from lib import HandDetector, HandSignClassifier

detector = HandDetector()
classifier = HandSignClassifier(output_size=2, pretrained_model_path = "your/model/path")

handpoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
   if handpoints is not None:
       sign_label: int = classifier.predict(handpoints)
```
  
以下は応用例です.  

https://github.com/kou1046/open-campus-2023  
ハンドサインでドローンを動かすプログラムです．オープンキャンパスの際に使用しました．