# セットアップと再現手順

## 1. 前提

- Raspberry Pi Zero 2 W
- 64-bit Raspberry Pi OS
- Picamera2で認識できるCSIカメラ
- SG90 2個と外部5V電源
- Piと同じネットワークに接続したPC

## 2. 基本パッケージ

基本機能では次を使用します。

- Python 3
- Picamera2 / libcamera
- gpiozero
- pigpio / pigpiod
- Pillow

OpenCV DNN・KCF実験ではOpenCV、FOMO実験ではEdge Impulse Linux Python SDKが別途必要です。使用中のRaspberry Pi OSに適した公式またはディストリビューションの手順で導入してください。

## 3. 段階的な確認

1. `src/01A_camera_still_test.py`で静止画取得
2. `src/01B_camera_dual_stream_test.py`でカメラ速度確認
3. `src/02A_mjpeg_stream_test.py`でMJPEG配信
4. `src/03_servo_test.py`で狭い角度の2軸動作確認
5. `src/05_manual_gimbal_web_control.py`で手動操作
6. 必要に応じて`experiments/`以下の方式を個別評価

## 4. 手動操作

```bash
python3 src/05_manual_gimbal_web_control.py
```

PCブラウザーで次を開きます。

```text
http://raspberrypi.local:8000/
```

名前解決できない場合は、`raspberrypi.local`をPiのIPアドレスへ置き換えます。Webサーバーをインターネットへ転送しないでください。

## 5. KCF実験

KCF実験のソースとWeb画面は`experiments/kcf-tracking/`にあります。映像上で対象領域を手動選択する方式で、自動人物検出とは別です。人物や動く対象を使う前に、静止物と狭いサーボ角度で確認してください。

## 6. モデルについて

この公開準備版にはEIM、ONNX、PyTorch、Caffeモデルを同梱していません。各実験コードの引数で、利用者が正規の配布元から取得または生成したモデルを指定してください。
