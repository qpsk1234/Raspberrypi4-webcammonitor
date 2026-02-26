import cv2
import os
import datetime
import threading
import time
import queue
from collections import deque

class Recorder:
    """
    非同期書き込み、フレーム補完、およびプリ録画（検知前録画）に対応した録画モジュール。
    """
    def __init__(self, save_directory='records', fps=20.0, resolution=(1280, 720), post_seconds=5, pre_frames=60):
        self.save_directory = save_directory
        self.fps = fps
        self.resolution = resolution
        self.post_seconds = post_seconds
        self.frame_duration = 1.0 / fps
        self.pre_frames = pre_frames

        self._writer = None
        self._lock = threading.Lock()
        self._stop_timer = None
        self.is_recording = False
        self.current_video_path = None
        
        # 非同期処理用
        self._queue = queue.Queue(maxsize=500) # プリ録画バッファ分を考慮して余裕を持たせる
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
        # プリ録画バッファ (リングバッファ)
        self._pre_buffer = deque(maxlen=pre_frames)
        
        # フレーム補完用
        self._last_frame_time = 0
        self._last_frame = None

        os.makedirs(save_directory, exist_ok=True)

    def _worker(self):
        """バックグラウンドでQueueからフレームを取り出して書き込むスレッド"""
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None: break
                
                writer, frame = item
                if writer is not None:
                    try:
                        writer.write(frame)
                    except Exception as e:
                        print(f"[Recorder Worker] Write error: {e}")
                
                self._queue.task_done()
            except queue.Empty:
                continue

    def update_buffer(self, frame):
        """常時呼び出し。リングバッファを更新する。"""
        if frame is None: return
        # リサイズして保存（メモリ節約と書き込み負荷軽減のため）
        resized = cv2.resize(frame, self.resolution)
        with self._lock:
            self._pre_buffer.append(resized)

    def start_recording(self, frame):
        """録画を開始する。"""
        with self._lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None
                return

            if self._writer is None:
                candidates = [
                    ('avc1', '.mp4'),
                    ('mp4v', '.mp4'),
                    ('MJPG', '.mp4'),
                    ('MJPG', '.avi'),
                ]
                
                base_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                
                for codec, ext in candidates:
                    filepath = os.path.join(self.save_directory, f"detected_{base_ts}{ext}")
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        w = cv2.VideoWriter(filepath, fourcc, self.fps, self.resolution)
                        
                        if w.isOpened():
                            print(f"[Recorder] Start (Async+Pre): {codec} ({filepath}) PreFrames: {len(self._pre_buffer)}")
                            self._writer = w
                            self.current_video_path = filepath
                            self.is_recording = True
                            
                            # プリ録画バッファの中身をQueueに一気に投入
                            for f in self._pre_buffer:
                                self._push_to_queue(f)
                            
                            self._last_frame_time = time.time()
                            self._last_frame = None
                            return
                        else:
                            w.release()
                    except:
                        pass
                
                print("[CRITICAL] Recorder: Failed to open any VideoWriter.")
                self.is_recording = False

    def write(self, frame):
        """フレーム補完を行いながらQueueに投入。"""
        if not self.is_recording:
            return

        now = time.time()
        
        with self._lock:
            if self._writer is None:
                return
            
            # 初回フレーム
            if self._last_frame is None:
                self._last_frame_time = now
                self._last_frame = cv2.resize(frame, self.resolution)
                self._push_to_queue(self._last_frame)
                return

            elapsed = now - self._last_frame_time
            num_frames = int(elapsed / self.frame_duration)

            if num_frames > 0:
                resized = cv2.resize(frame, self.resolution)
                for i in range(num_frames - 1):
                    self._push_to_queue(self._last_frame)
                
                self._push_to_queue(resized)
                self._last_frame = resized
                self._last_frame_time += num_frames * self.frame_duration

    def _push_to_queue(self, frame):
        try:
            self._queue.put_nowait((self._writer, frame))
        except queue.Full:
            pass

    def schedule_stop(self, override_post_seconds=None):
        with self._lock:
            if not self.is_recording or self._stop_timer is not None:
                return
            
            wait_time = override_post_seconds if override_post_seconds is not None else self.post_seconds
            self._stop_timer = threading.Timer(wait_time, self._stop_callback)
            self._stop_timer.start()

    def _stop_callback(self):
        self._stop()
        with self._lock:
            self._stop_timer = None

    def _stop(self):
        with self._lock:
            if self._writer is not None:
                self._queue.join()
                self._writer.release()
                self._writer = None
                self.is_recording = False
                self._last_frame = None
                print(f"[Recorder] Saved (with Pre): {self.current_video_path}")

    def release(self):
        if self._stop_timer:
            self._stop_timer.cancel()
        self._stop()
        self._running = False
        self._queue.put(None)
        self._thread.join()
