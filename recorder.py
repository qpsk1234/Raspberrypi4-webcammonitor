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
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None

            if self._writer is None:
                filepath = self._new_filepath()
                # ブラウザ互換性の高い H.264 (avc1) を優先、失敗したら mp4v
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    self._writer = cv2.VideoWriter(
                        filepath, fourcc, self.fps, self.resolution)
                    if not self._writer.isOpened():
                        raise Exception("avc1 codec failed")
                except Exception:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._writer = cv2.VideoWriter(
                        filepath, fourcc, self.fps, self.resolution)
                
                if not self._writer.isOpened():
                    print(f"[ERROR] Recorder: 録画ファイルを開けませんでした: {filepath}")
                    self._writer = None
                    self.is_recording = False
                    return

                self.current_video_path = filepath
                self.is_recording = True
                print(f"[Recorder] 録画開始: {filepath}")

    def write(self, frame):
        """フレームを書き込む。"""
        with self._lock:
            if self._writer is not None:
                resized = cv2.resize(frame, self.resolution)
                self._writer.write(resized)

    def schedule_stop(self, override_post_seconds=None):
        """検知が途切れた際に `post_seconds` 後に録画終了をスケジュールする。"""
        with self._lock:
            # すでに停止タイマーが動いている場合は何もしない（これにより毎フレームのリセットを防ぐ）
            if self._stop_timer is not None:
                return
            
            wait_time = override_post_seconds if override_post_seconds is not None else self.post_seconds
            self._stop_timer = threading.Timer(wait_time, self._stop_)
            self._stop_timer.start()

    def _stop_(self):
        """Timerから呼ばれるラッパー。ロック制御のために _stop を呼ぶ。"""
        self._stop()
        with self._lock:
            self._stop_timer = None

    def _stop(self):
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
                self.is_recording = False
                print("[Recorder] 録画終了")

    def release(self):
        """アプリ終了時に呼ぶ。"""
        if self._stop_timer:
            self._stop_timer.cancel()
        self._stop()
