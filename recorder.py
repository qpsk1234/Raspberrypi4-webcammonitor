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
    FFmpegパイプ、非同期書き込み、精密フレーム補完(FPS同期)、およびプリ録画に対応した録画モジュール。
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
        self._starting = False # 起動処理中フラグ
        self.current_video_path = None
        
        # 非同期処理用
        self._queue = queue.Queue(maxsize=2000) # 補完フレーム増を考慮して拡張
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
        # プリ録画バッファ (リングバッファ) : (timestamp, frame)
        self._pre_buffer = deque(maxlen=pre_frames)
        
        # 精密同期用
        self._start_session_time = 0
        self._total_frames_pushed = 0
        self._last_frame = None

        os.makedirs(save_directory, exist_ok=True)

    def _worker(self):
        """バックグラウンドでQueueからフレームを取り出してFFmpegのstdinへ流し込むスレッド"""
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    self._queue.task_done()
                    break
                
                proc = self._process
                frame = item[1]

                if proc is not None:
                    try:
                        if proc.poll() is None:
                            proc.stdin.write(frame.tobytes())
                    except Exception as e:
                        print(f"[Recorder Worker] FFmpeg Pipe error: {e}")
                
                self._queue.task_done()
            except queue.Empty:
                continue

    def update_buffer(self, frame):
        """常時呼び出し。タイムスタンプと共にリングバッファを更新する。"""
        if frame is None: return
        now = time.time()
        resized = cv2.resize(frame, self.resolution)
        with self._lock:
            self._pre_buffer.append((now, resized))

    def start_recording(self, frame):
        """録画を開始する（非同期プロセス起動）。"""
        with self._lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None
                return

            if self.is_recording or self._starting:
                return

            self._starting = True
            self.is_recording = True
            self._total_frames_pushed = 0
            
            # 非同期でFFmpegを起動
            threading.Thread(target=self._async_start_ffmpeg, daemon=True).start()

    def _async_start_ffmpeg(self):
        """FFmpegを別スレッドで起動し、バッファを同期的に流し込む。"""
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.save_directory, f"detected_{ts}.mp4")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f"{self.resolution[0]}x{self.resolution[1]}",
            '-pix_fmt', 'bgr24', '-r', str(self.fps),
            '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-f', 'mp4', filepath
        ]
        
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            with self._lock:
                self._process = proc
                self.current_video_path = filepath
                self._starting = False
                
                # プリ録画バッファのFPS同期投入
                if len(self._pre_buffer) > 0:
                    # 最初のフレームを基準にする
                    buffer_copy = list(self._pre_buffer)
                    start_t = buffer_copy[0][0]
                    self._start_session_time = start_t
                    
                    for t, f in buffer_copy:
                        self._sync_write(t, f)
                else:
                    self._start_session_time = time.time()
                    
                print(f"[Recorder] Synced Recording Started: {filepath}")

        except Exception as e:
            print(f"[ERROR] Failed to launch FFmpeg: {e}")
            with self._lock:
                self._starting = False
                self.is_recording = False

    def write(self, frame):
        """実時間に基づいた精密補完を行いながらQueueに投入。"""
        if not self.is_recording:
            return
        
        now = time.time()
        with self._lock:
            # プロセス起動中も同期処理を実行してQueueに積む
            self._sync_write(now, frame)

    def _sync_write(self, timestamp, frame):
        """
        指定されたタイムスタンプと現在のセッション開始時刻の差分から、
        出力すべきフレーム数を計算してQueueに投入する（FPS平準化）。
        """
        if self._last_frame is None:
            # セッション一番最初のフレーム
            self._last_frame = cv2.resize(frame, self.resolution)
            self._push_to_queue(self._last_frame)
            self._total_frames_pushed = 1
            return

        # セッション開始からの理想的な累積フレーム数
        elapsed = timestamp - self._start_session_time
        target_total_frames = int(elapsed * self.fps) + 1
        
        # 不足しているフレーム数（ドロップ補完）
        needed = target_total_frames - self._total_frames_pushed
        
        if needed > 0:
            resized = cv2.resize(frame, self.resolution)
            # 不足分は「直前のフレーム」で埋める（最後の1枚だけ「今のフレーム」にする）
            for _ in range(needed - 1):
                self._push_to_queue(self._last_frame)
            
            self._push_to_queue(resized)
            self._last_frame = resized
            self._total_frames_pushed = target_total_frames

    def _push_to_queue(self, frame):
        try:
            self._queue.put_nowait((None, frame))
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
        # 起動中の場合は完了を待つ
        retries = 20
        while self._starting and retries > 0:
            time.sleep(0.5)
            retries -= 1

        with self._lock:
            if self._process is not None:
                self._queue.join()
                try:
                    if self._process.stdin:
                        self._process.stdin.close()
                    self._process.wait(timeout=5)
                except Exception as e:
                    print(f"[Recorder] FFmpeg termination error: {e}")
                
                self._process = None
                self.is_recording = False
                self._last_frame = None
                self._total_frames_pushed = 0
                print(f"[Recorder] Saved (Synced): {self.current_video_path}")

    def release(self):
        if self._stop_timer:
            self._stop_timer.cancel()
        self._stop()
        self._running = False
        self._queue.put(None)
        try:
            self._thread.join(timeout=2)
        except:
            pass
