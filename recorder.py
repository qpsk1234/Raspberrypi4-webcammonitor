import cv2
import os
import datetime
import threading

class Recorder:
    """
    人間検知中に映像を録画するモジュール。
    検知が途切れた後も `post_seconds` 秒間だけ録画を継続する（後バッファ）。
    """
    def __init__(self, save_directory='records', fps=20.0, resolution=(640, 480), post_seconds=5):
        self.save_directory = save_directory
        self.fps = fps
        self.resolution = resolution
        self.post_seconds = post_seconds

        self._writer = None
        self._lock = threading.Lock()
        self._stop_timer = None
        self.is_recording = False
        self.current_video_path = None # 現在または直前の録画パス

        os.makedirs(save_directory, exist_ok=True)

    def _new_filepath(self):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.save_directory, f"detected_{ts}.mp4")

    def start_recording(self, frame):
        """録画を開始する。既に録画中の場合はタイマーをリセットするだけ。"""
        with self._lock:
            # すでに録画終了待ちならタイマーキャンセルして継続
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None
                return

            if self._writer is None:
                # 候補となる (コーデック, 拡張子) のリスト
                # ユーザー環境のテスト結果: MJPG が動作したため、MJPG を手厚く。
                # avc1/mp4v はブラウザ互換性が高いが、OpenCV環境によって失敗しやすい。
                candidates = [
                    ('avc1', '.mp4'), # H.264 (ブラウザ最適)
                    ('mp4v', '.mp4'), # MP4 (高い互換性)
                    ('MJPG', '.mp4'), # MJPG in MP4 (ユーザー環境で期待)
                    ('MJPG', '.avi'), # MJPG in AVI (確実に動作する可能性大)
                    ('XVID', '.avi'), # 汎用AVI
                ]
                
                base_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                
                for codec, ext in candidates:
                    filepath = os.path.join(self.save_directory, f"detected_{base_ts}{ext}")
                    
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        self._writer = cv2.VideoWriter(
                            filepath, fourcc, self.fps, self.resolution)
                        
                        if self._writer is not None and self._writer.isOpened():
                            print(f"[Recorder] 録画開始成功: {codec} ({filepath})")
                            self.current_video_path = filepath
                            self.is_recording = True
                            return
                        else:
                            if self._writer: self._writer.release()
                            self._writer = None
                    except Exception as e:
                        # 既知の「VIDEOIO(CV_IMAGES)」警告等は無視し、詳細ログは出さない
                        if self._writer: self._writer.release()
                        self._writer = None
                
                print(f"[CRITICAL] Recorder: 全てのコーデック・拡張子の組み合わせで失敗しました。")
                self.is_recording = False

    def write(self, frame):
        """フレームを書き込む。"""
        if not self.is_recording:
            return
            
        with self._lock:
            if self._writer is not None:
                try:
                    # 解像度が一致している必要がある
                    resized = cv2.resize(frame, self.resolution)
                    self._writer.write(resized)
                except Exception as e:
                    print(f"[ERROR] Recorder.write で例外発生: {e}")

    def schedule_stop(self, override_post_seconds=None):
        """検知が途切れた際に `post_seconds` 後に録画終了をスケジュールする。"""
        with self._lock:
            if not self.is_recording:
                return
            
            # すでに停止タイマーが動いている場合は何もしない（二重設定防止）
            if self._stop_timer is not None:
                return
            
            wait_time = override_post_seconds if override_post_seconds is not None else self.post_seconds
            print(f"[Recorder] 録画終了をスケジュール ({wait_time}秒後)")
            self._stop_timer = threading.Timer(wait_time, self._stop_)
            self._stop_timer.start()

    def _stop_(self):
        """Timerから呼ばれるラッパー。"""
        self._stop()
        with self._lock:
            self._stop_timer = None

    def _stop(self):
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception as e:
                    print(f"[ERROR] Recorder release でエラー: {e}")
                
                self._writer = None
                self.is_recording = False
                print(f"[Recorder] 録画終了・保存完了: {self.current_video_path}")

    def release(self):
        """アプリ終了時に呼ぶ。"""
        print("[Recorder] システム終了に伴うリリース処理")
        if self._stop_timer:
            self._stop_timer.cancel()
            self._stop_timer = None
        self._stop()
