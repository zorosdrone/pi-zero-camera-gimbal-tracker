# Raspberry Pi Camera SG90 Gimbal Tracker

Raspberry Pi Zero 2 W、OV5647カメラ、SG90サーボ2個を使ったパン・チルトジンバルの実験プロジェクトです。カメラ配信、手動操作、物体検知、KCF追跡を段階的に検証しています。

![カメラ搭載ジンバル](assets/04_カメラ搭載ジンバルとPiZero2W接続.jpg)

> このプロジェクトは実験段階です。人物を自動検知して安定して追い続ける完成品ではありません。

## KCF追跡の構成とデモ

![手動ROI選択とKCF追跡の構成](assets/kcf-tracking-system-overview.jpg)

PCブラウザーで対象領域（ROI）を選び、Pi Zero 2 W上のOpenCV KCFで追跡します。YOLOによる自動検出は使わず、追跡中心と映像中心の差からSG90を制御する構成です。

パン・チルト機構には、MakerWorldの[SG90 Servo 2 Axis Gimbal](https://makerworld.com/ja/models/511916-sg90-servo-2-axis-gimbal)を3Dプリントして使用しています。モデルの作者名、ライセンス、利用条件は公開時にMakerWorld掲載ページで再確認してください。

- [ブラウザーでROIを選ぶデモ動画](https://github.com/zorosdrone/pi-zero-camera-gimbal-tracker/releases/download/v0.1.0-kcf-demo/gimbal-kcf-roi-browser-demo-v0.1.0.mp4)
- [ジンバルが追従するデモ動画](https://github.com/zorosdrone/pi-zero-camera-gimbal-tracker/releases/download/v0.1.0-kcf-demo/gimbal-kcf-servo-demo-v0.1.0.mp4)
- [Release v0.1.0-kcf-demo](https://github.com/zorosdrone/pi-zero-camera-gimbal-tracker/releases/tag/v0.1.0-kcf-demo)

動画はGit履歴へ含めず、GitHub Releaseの添付ファイルとして配布します。公開前は上のリンクが未作成です。

## 現在の状態

| 機能 | 状態 |
|---|---|
| OV5647静止画・デュアルストリーム取得 | 実機確認済み |
| MJPEG・H.264配信 | 実機確認済み |
| SG90 2軸の安全範囲試験 | 実機確認済み |
| ブラウザーからの手動ジンバル操作 | 実機確認済み |
| PCブラウザーでROIを選ぶKCF追跡 | 基本動作確認済み、実環境評価は継続 |
| YOLO/OpenCV DNN | Pi Zero 2 Wでは速度不足 |
| MobileNet-SSD/OpenCV DNN | Pi Zero 2 Wでは速度不足 |
| Edge Impulse FOMO | 速度は実用候補、背景誤検出が多い |
| 完全自動人物追尾 | 未完成 |

## 最初に試す構成

現在もっとも再現しやすい入口は、カメラ映像をブラウザーへ配信し、手動でジンバルを動かす構成です。

```bash
python3 src/05_manual_gimbal_web_control.py
```

同じネットワークのPCから次を開きます。

```text
http://raspberrypi.local:8000/
```

配線前に[安全上の注意](SAFETY.md)を読み、[部品表](BOM.md)と[セットアップ手順](docs/setup.md)を確認してください。

## 実験フォルダ

- `experiments/edge-impulse-fomo/`: FOMOのカメラ・静止画推論
- `experiments/opencv-dnn/`: YOLO、MobileNet-SSDのCPU性能比較
- `experiments/kcf-tracking/`: ブラウザーROI選択、KCF追跡、サーボ制御

測定条件と結果は[ベンチマーク結果](docs/benchmarks.md)へまとめています。

## 公開対象外

初回版には学習・推論モデル、人物サンプル画像、商品ページ画像、室内メモが写った試写、既存動画を含めません。モデルは利用者自身が正規の配布元から取得・生成する前提です。

新しいデモ動画は後から`media/`またはGitHub Releasesへ追加します。

## 公開準備状態

コードと公開用文書の初回整理まで完了しています。ライセンス決定、最終監査、独立Git履歴の作成、GitHub公開はまだ行っていません。
