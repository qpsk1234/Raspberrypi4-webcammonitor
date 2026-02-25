# 実装タスク：Raspberry Pi 4 人間検知監視カメラアプリ

## 1. 環境・準備
- [x] `requirements.txt` の作成と依存ライブラリの定義
- [x] `numpy<2` によるNumPy 2.0互換問題の対処
- [x] `setup_model.py` によるTFLiteモデルの自動ダウンロード
- [x] `records/` 保存ディレクトリの自動作成

## 2. コアモジュール
- [x] `camera.py`: スレッドセーフなカメラキャプチャクラス
- [x] `detector.py`: TFLiteを使用した人間検知エンジン（雛形）
    - [ ] TFLite Interpreter の初期化（実環境への対応）
    - [ ] 推論処理とバウンディングボックスの計算（実装）
- [x] `notifier.py`: Telegram Bot による通知（画像送信）
- [x] `web_stream.py`: Flask Webサーバー
    - [x] MJPEG ストリーミング (`/video_feed`)
    - [x] Web管理ダッシュボード (`/`)
    - [x] ステータスAPI (`/api/status`)
    - [x] 設定保存API (`/api/config`)

## 3. メインアプリケーション (`main.py`)
- [x] `config.json` からの設定読み込み
- [x] マルチスレッド制御（Webサーバー用スレッド）
- [x] 検知ループの実装
- [x] `system_status` へのリアルタイムステータス書き込み
- [x] Telegram通知の間隔制御（`notify_interval`）
- [x] ヘッドレス環境対応（`use_gui` フラグ + try-except 自動無効化）

## 4. Web管理画面
- [x] ダークテーマUIの実装
- [x] ライブ映像の表示
- [x] 人間検知アラートバナー
- [x] リアルタイムステータスパネル（2秒ポーリング）
- [x] 検知閾値スライダー
- [x] 通知間隔・解像度設定フォーム
- [x] Telegram Bot Token / Chat ID フォーム
- [x] 設定保存（`/api/config` → `config.json`）

## 5. 完了した追加実装
- [x] `detector.py` に実際のTFLite推論ロジックを実装（前処理・推論・後処理）
- [x] 録画機能の実装（`recorder.py` — 後バッファ付き `cv2.VideoWriter` 録画）
- [x] ログ機能の追加（`detection_logger.py` — CSVへの検知履歴保存）
- [x] 認証機能の追加（Web管理画面へのBasic認証）
- [x] 検知ログのダッシュボード表示（日時・人数・確信度・スナップ）
- [x] 設定ホットリロード（ループごとに `config.json` を再読み込み）
- [x] **複数オブジェクト検知への対応**: `detector.py` に COCO クラスマップを追加し、車や動物なども描画するように強化
- [x] **通知フィルタリングの実装**: `main.py` にて人間が検知された場合のみ通知および録画を開始するように制御
