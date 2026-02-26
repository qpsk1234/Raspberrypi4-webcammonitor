import cv2
import os
import datetime
import threading
import time
import queue
import subprocess
from collections import deque

class Recorder:
    """
    FFmpegパイプ、非同期書き込み、フレーム補完、およびプリ録画に対応した録画モジュール。
    """
    def __init__(self, save_directory='records', fps=20.0, resolution=(1280, 720), post_seconds=5, pre_frames=60):
        self.save_directory = save_directory
        self.fps = fps
        self.resolution = resolution
        self.post_seconds = post_seconds
        self.frame_duration = 1.0 / fps
        self.pre_frames = pre_frames

        self._process = None # FFmpeg subprocess
        self._lock = threading.Lock()
        self._stop_timer = None
        self.is_recording = False
        self.current_video_path = None
        
        # 非同期処理用
        self._queue = queue.Queue(maxsize=500)
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
        """バックグラウンドでQueueからフレームを取り出してFFmpegのstdinへ流し込むスレッド"""
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None: break
                
                process, frame = item
                if process is not None and process.poll() is None:
                    try:
                        # OpenCV(BGR) -> FFmpegへ生データを書き出す
                        process.stdin.write(frame.tobytes())
                    except Exception as e:
                        print(f"[Recorder Worker] FFmpeg Pipe error: {e}")
                
                self._queue.task_done()
            except queue.Empty:
                continue

    def update_buffer(self, frame):
        """常時呼び出し。リングバッファを更新する。"""
        if frame is None: return
        # リサイズして保存
        resized = cv2.resize(frame, self.resolution)
        with self._lock:
            self._pre_buffer.append(resized)

    def start_recording(self, frame):
        """録画を開始する（FFmpegプロセスの起動）。"""
        with self._lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None
                return

            if self._process is None:
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = os.path.join(self.save_directory, f"detected_{ts}.mp4")
                self.current_video_path = filepath
                
                # FFmpeg コマンド構築
                # bgr24 (OpenCV) -> yuv420p (H.264標準)
                cmd = [
                    'ffmpeg',
                    '-y', # Overwrite
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f"{self.resolution[0]}x{self.resolution[1]}",
                    '-pix_fmt', 'bgr24',
                    '-r', str(self.fps),
                    '-i', '-', # Stdin
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', # ブラウザ互換性
                    '-preset', 'ultrafast', # 低負荷
                    '-f', 'mp4',
                    filepath
                ]
                
                try:
                    self._process = subprocess.Popen(
                        cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    
                    if self._process:
                        print(f"[Recorder] FFmpeg Started: {filepath}")
                        self.is_recording = True
                        
                        # プリ録画バッファをQueueへ
                        for f in self._pre_buffer:
                            self._push_to_queue(f)
                        
                        self._last_frame_time = time.time()
                        self._last_frame = None
                except Exception as e:
                    print(f"[ERROR] Failed to launch FFmpeg: {e}")
                    self.is_recording = False

    def write(self, frame):
        """フレーム補完を行いながらQueueに投入。"""
        if not self.is_recording:
            return

        now = time.time()
        
        with self._lock:
            if self._process is None:
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
            self._queue.put_nowait((self._process, frame))
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
            if self._process is not None:
                # Queueが完全に消化されるのを待つ
                self._queue.join()
                
                # パイプを閉じてFFmpegを終了させる
                try:
                    if self._process.stdin:
                        self._process.stdin.close()
                    self._process.wait(timeout=5)
                except Exception as e:
                    print(f"[Recorder] FFmpeg termination error: {e}")
                
                self._process = None
                self.is_recording = False
                self._last_frame = None
                print(f"[Recorder] Saved (FFmpeg MP4): {self.current_video_path}")

    def release(self):
        if self._stop_timer:
            self._stop_timer.cancel()
        self._stop()
        self._running = False
        self._queue.put(None)
        self._thread.join()
