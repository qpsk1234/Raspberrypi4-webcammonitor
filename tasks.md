# 実装タスク：Raspberry Pi 4 人間検知監視カメラアプリ

## 1. 準備
- [ ] `requirements.txt` の作成と依存ライブラリの定義
- [ ] TensorFlow Lite モデル (MobileNet SSD) のダウンロード

## 2. コアモジュール
- [ ] `camera.py`: カメラキャプチャクラスの実装
    - [ ] OpenCVによるフレーム取得
    - [ ] スレッドセーフなバッファリング
- [ ] `detector.py`: 検知エンジングクラスの実装
    - [ ] TFLite Interpreter の初期化
    - [ ] 推論処理とバウンディングボックスの計算
- [ ] `web_stream.py`: Flaskによるストリーミングサーバーの構築
    - [ ] MJPEG 配信エンドポイント

## 3. アプリケーション統合
- [ ] `main.py`: マルチスレッド制御
    - [ ] スレッド間のデータ（フレーム）共有
    - [ ] UI表示（cv2.imshow）
- [ ] `notifier.py`: Telegram通知の実装
    - [ ] 検知時のスナップショット送信

## 4. 仕上げ
- [ ] 録画ロジックの追加
- [ ] 設定ファイル（config.json）による閾値やトークンの管理
