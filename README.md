# AI-Surveillance-Camera (Raspberry Pi 4)

Raspberry Pi 4 と TensorFlow Lite を利用した、高機能な人間検知監視カメラシステムです。
リアルタイムでの物体検知、録画、静止画キャプチャ、Telegram通知、およびブラウザベースの管理ダッシュボードを提供します。

- Google Antigravityで開発しました。
- Tensorflow Lite用のモデルファイル（tflite）の準備は別途必要です
（COCO-SSDモデルの利用を想定しています）

## 🌟 主な機能

- **高度なリアルタイム検知**: TensorFlow Lite モデルを使用し、人間、車、ペットなどの複数オブジェクトをリアルタイムで検知・識別。
- **インテリジェント通知**: ターゲット（例：人間）を検知した際のみ Telegram への即時通知（写真付き）。検知セッションごとの通知抑制機能を搭載。
- **高品質・高安定録画システム**:
    - **FFmpeg エンジン**: 内部エンジンを FFmpeg へ刷新。ブラウザ互換性の高い H.264/MP4 形式で確実に保存。
    - **プリ録画（遡り録画）**: 検知した瞬間の数秒前（フレーム指定）から録画を開始。決定的な瞬間を逃しません。
    - **実時間ベースのFPS同期（早送り防止）**: キャプチャ時刻に基づいた精密なフレーム補完ロジックにより、負荷時でも実時間通りの正確な再生速度を維持。
    - **非同期プロセス制御**: FFmpeg の起動と書き込みを完全に非同期化し、Raspberry Pi 上でのメインループ（検知処理）の安定性を最大化。
- **Webダッシュボード / メディアブラウザ**: 
    - リアルタイムストリーミング映像の閲覧。
    - **メディアブラウザ**: 保存された動画や写真を一覧表示・再生・管理できる専用インターフェース。
    - 動的な検知設定（閾値、クラス選択、解像度、通知設定、プリ録画枚数等）の変更。
- **モデルテストツール**: 実導入前にモデルの精度を確認できる専用ツール (`Tools/model_test.py`) を同梱。

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
- デフォルトID: `admin` / パスワード: `admin` （`config.json`で変更可能）

### 3. Telegram 通知の設定
「✈️ Telegram」タブから、Bot Token と Chat ID を入力して保存し、「通知テスト」をクリックして動作を確認してください。

## ⚙️ 設定項目 (`config.json`)

Web UI からほぼすべての設定を変更可能です：
- `detection_threshold`: 検知の感度
- `target_classes`: 検知対象とするCOCOクラスID
- `recorder_post_seconds`: 検知終了後の録画継続時間
- `snapshot_mode`: 「開始時のみ」または「開始と終了時」の写真保存モード

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
