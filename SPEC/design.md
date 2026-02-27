# 詳細設計：Raspberry Pi 4 人間検知監視カメラアプリ

## 1. システムアーキテクチャ
システムはマルチスレッド型アプリケーションとして構成する。

| スレッド/モジュール | 役割 |
|---|---|
| カメラキャプチャ・スレッド | OpenCVで映像取得、スレッドセーフなバッファに書き込み |
| 検知エンジン（メインループ） | バッファからフレームを取得しTFLiteで人間検知。常時プリ録画バッファ（タイムスタンプ付）を更新 |
| Web配信・スレッド | FlaskでMJPEGストリーミング、Web管理画面、メディアブラウザを提供 |
| 録画エンジン (`recorder.py`) | FFmpegの非同期起動、タイムスタンプベースの精密FPS同期、非同期連鎖書き出し |
| 通知モジュール (`notifier.py`) | 検知イベント発生時にTelegram APIへ送信（セッション抑制機能付） |

## 2. テクノロジースタック

| カテゴリ | 採用技術 | 理由 |
|---|---|---|
| 言語 | Python 3.9+ | |
| 画像処理 | OpenCV (cv2) | |
| AIエンジン | TensorFlow Lite (`tflite-runtime`) | Raspberry Pi 4 での高速動作 |
| AIモデル | SSD MobileNet V2 (COCOデータセット) | 人間検知に特化、軽量 |
| Webフレームワーク | Flask | 軽量なストリーミング・UI配信 |
| 通知 | Telegram Bot API (`requests`) | |
| 依存バージョン | `numpy<2` | NumPy 2.0 との互換性エラー回避 |

## 3. プロジェクト構成

```
project/
├── main.py           # エントリーポイント。スレッド制御・検知ループ
├── camera.py         # スレッドセーフなカメラキャプチャクラス
├── detector.py       # TFLiteを使用した人間検知エンジン
├── recorder.py       # FFmpegを使用した高度な録画モジュール（非同期・プリ録画対応）
├── notifier.py       # Telegram通知モジュール（セッション抑制対応）
├── detection_logger.py # 履歴ログ管理と特定日付の抽出
├── web_stream.py     # 管理画面・メディアブラウザ・ストリーミング・API
├── config.json       # 全設定を管理する外部設定ファイル
└── records/          # 録画（MP4）およびスナップショット（JPG）の保存先
```

## 4. 主要機能の設計

### 4.1 カメラキャプチャ (`camera.py`)
- `Camera` クラスが専用スレッドで常時フレームを取得。
- `threading.Lock` によりスレッドセーフなバッファリングを実現。

### 4.2 人間検知 (`detector.py`)
- `HumanDetector` クラスが TFLite Interpreter を保持。
- `detect(frame)` が前処理・推論・後処理を行い、バウンディングボックスリストを返す。
- `draw_detections(frame, detections)` で検知枠を描画。

### 4.3 Web管理画面 (`web_stream.py`)

| URL | メソッド | 説明 |
|---|---|---|
| `/` | GET | リアルタイム監視ダッシュボード |
| `/media` | GET | 保存済み動画・写真のメディアブラウザ |
| `/video_feed` | GET | MJPEG ライブストリーミング |
| `/api/config` | POST | 閾値・解像度・プリ録画・通知等の設定更新 |
| `/api/media_list` | GET | 保存済みファイルの一覧取得 |

**`/api/status` レスポンス例:**
```json
{
  "running": true,
  "detections_total": 5,
  "last_detected": "2026-02-25 14:53:00",
  "fps": 15.2,
  "human_count": 1,
  "stream_width": 640,
  "stream_height": 480
}
```

### 4.4 ヘッドレス対応
- `config.json` の `use_gui: false` でモニター出力を無効化。
- `cv2.imshow` / `cv2.waitKey` は try-except で保護し、エラー時は自動無効化。

### 4.5 Telegram通知 (`notifier.py`)
- 検知の確信度が閾値を超えた場合、`notify_interval` 秒に1回スナップショットを送信。
- 画像を一時ファイルとして保存して送信後に削除。

## 5. 設定スキーマ (`config.json`)

| キー | 型 | 説明 |
|---|---|---|
| `telegram_token` | string | Telegram Bot のトークン |
| `telegram_chat_id` | string | 通知先のチャットID |
| `detection_threshold` | float | 検知閾値 (0.1〜1.0) |
| `notify_interval` | int | Telegram通知の最低間隔（秒） |
| `model_path` | string | TFLiteモデルファイルのパス |
| `video_source` | int | カメラデバイス番号 |
| `save_directory` | string | 画像保存ディレクトリ |
| `stream_width` | int | Webストリーミング幅（px） |
| `stream_height` | int | Webストリーミング高さ（px） |
| `use_gui` | bool | モニター出力の有効／無効 |

## 6. データフロー

```mermaid
graph TD
    A[Camera] -->|OpenCV| B[main.py - 検知ループ]
    B -->|毎フレーム| C[recorder.update_buffer]
    C -->|Timestamp + Frame| D[プリ録画バッファ]
    B --> E{ターゲット検知?}
    E -->|Yes| F[recorder.start_recording]
    F -->|Thread| G[Async FFmpeg Startup]
    G -->|実時間同期投入| H[FFmpeg Process]
    B -->|毎フレーム| I[recorder.write]
    I -->|Queue| J[Async Worker Thread]
    J -->|pipe| H
    H -->|libx264| K[records/ .mp4]
```
