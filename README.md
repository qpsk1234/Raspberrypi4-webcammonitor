# AI-Surveillance-Camera (Raspberry Pi 4)

Raspberry Pi 4 と TensorFlow Lite を利用した、高機能な人間検知監視カメラシステムです。
リアルタイムでの物体検知、録画、静止画キャプチャ、Telegram通知、およびブラウザベースの管理ダッシュボードを提供します。

- Google Antigravityで開発しました。
- Tensorflow Lite用のモデルファイル（tflite）の準備は別途必要です
（COCO-SSDモデルの利用を想定しています）

## 🌟 主な機能

- **高度なリアルタイム検知**: TensorFlow Lite モデルを使用し、人間、車、ペットなどの複数オブジェクトをリアルタイムで検知・識別。
- **インテリジェント通知**: ターゲット（例：人間）を検知した際のみ Telegram への即時通知。静止画のみ、動画のみ、あるいは両方を送信するか Web 画面から柔軟に設定可能。
- **高品質・高安定録画システム**:
    - **FFmpeg エンジン**: ブラウザ互換性の高い H.264/MP4 形式で確実に保存。
    - **プリ録画機能**: 検知の数秒前から遡って録画可能なバッファリング機能を搭載。
    - **タイムスタンプ同期**: 負荷時でも正確な再生速度を維持するフレーム補正ロジック。
- **Webダッシュボード / メディアブラウザ**: 
    - リアルタイムストリーミング映像、動作ステータス、検知ログ履歴の閲覧。
    - **メディアブラウザ**: 保存された動画や写真を一覧表示・再生・管理できる専用インターフェース。
    - ログイン認証（デフォルト ID: admin / PASS: admin）によるセキュリティ確保。
    - 動的な検知設定（閾値、クラス選択、解像度、通知設定、プリ録画枚数等）の変更。
- **モデルテストツール (`Tools/model_test.py`)**: 実導入前にモデルの精度や性能を確認・評価できる専用ツール。

## 📋 セットアップ

### 1. 必要要件
- Raspberry Pi 4 (または同等の性能を持つPC/SBC)
- USBカメラ または Piカメラ
- Python 3.8+
- **FFmpeg**: 動画のエンコード・保存に必要 (`sudo apt install ffmpeg` 等)

### 2. ライブラリのインストール
```bash
pip install -r requirements.txt
```

### 3. モデルの準備
デフォルトの COCO-SSD モデルをセットアップします：
```bash
python setup_model.py
```

## 🚀 使い方

### 1. システムの起動
```bash
python main.py
```

### 2. Web管理画面へのアクセス
起動後、ブラウザで以下のURLにアクセスしてください：
- URL: `http://<RaspberryPiのIP>:5000`
- デフォルトID: `admin` / パスワード: `admin` （`config.json`で画面上から変更可能）

### 3. Telegram 通知の設定
「✈️ Telegram」タブから、Bot Token と Chat ID を入力。通知モード（静止画/動画/両方/なし）を選択して保存し、「テスト送信」をクリックして確認してください。

### 4. モデルテストツールの使用
```bash
python Tools/model_test.py --model model.tflite --input input.mp4 --output result.mp4
```
任意の画像・動画・モデルを使用して、検知精度や推論速度を事前に確認できます。

## ⚙️ 主な設定項目 (`config.json`)

Web UI からほぼすべての設定を変更可能です：
- `detection_threshold`: 検知の感度（レベル指定）
- `target_classes`: 検知対象とするクラスの個別選択
- `telegram_notify_mode`: 通知メディアの選択 (`photo`, `video`, `both`, `none`)
- `recorder_pre_frames`: プリ録画バッファ（遡り秒数に相当）
- `snapshot_mode`: 静止画保存の枚数設定

## 📂 ディレクトリ構造

- `main.py`: エントリーポイント（メインループ）
- `web_stream.py`: Flask Web サーバーと管理画面UI
- `detector.py`: TFLite による物体検知エンジン
- `recorder.py`: 動画録画モジュール
- `notifier.py`: Telegram 通知モジュール
- `SPEC/`: 要件定義・詳細設計ドキュメント
- `records/`: 録画・スナップショット保存先

## 📄 ライセンス

MIT License
