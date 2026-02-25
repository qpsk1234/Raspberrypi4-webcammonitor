# AI-Surveillance-Camera (Raspberry Pi 4)

Raspberry Pi 4 と TensorFlow Lite を利用した、高機能な人間検知監視カメラシステムです。
リアルタイムでの物体検知、録画、静止画キャプチャ、Telegram通知、およびブラウザベースの管理ダッシュボードを提供します。

- Google Antigravityで開発しました。
- Tensorflow Lite用のモデルファイル（tflite）の準備は別途必要です
（COCO-SSDモデルの利用を想定しています）

## 🌟 主な機能

- **高度なリアルタイム検知**: TensorFlow Lite モデルを使用し、人間、車、ペットなどの複数オブジェクトをリアルタイムで検知・識別。
- **インテリジェント通知**: ターゲット（例：人間）を検知した際のみ Telegram への即時通知（写真付き）。
- **自動録画・保存**: 検知開始から終了までをMP4形式で自動録画。ブラウザで再生可能なH.264互換（avc1）コーデックを採用。
- **Webダッシュボード**: 
    - リアルタイムストリーミング映像の閲覧
    - 検知ログ履歴の確認とスナップショット表示・動画再生
    - 動的な検知設定（閾値、クラス選択、解像度、通知設定等）の変更
- **柔軟なカスタマイズ**:
    - 検知対象クラスの動的選択（「人」だけ、「車」だけなど）
    - ラベル名の自由な変更
    - ポスト録画秒数やスナップショット解像度の詳細設定

## 📋 セットアップ

### 1. 必要要件
- Raspberry Pi 4 (または同等の性能を持つPC/SBC)
- USBカメラ または Piカメラ
- Python 3.8+

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
