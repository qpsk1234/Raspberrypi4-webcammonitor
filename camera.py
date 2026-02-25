import cv2
import threading
import time

MAX_RETRY = 5        # デバイスビジー時の最大リトライ回数
RETRY_INTERVAL = 2.0 # リトライ間隔（秒）

class Camera:
    def __init__(self, source=0):
        self.source = source
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.cap = None
        self._open()

    def _open(self):
        """カメラデバイスをオープンする。失敗した場合はリトライする。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        for attempt in range(1, MAX_RETRY + 1):
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                print(f"[Camera] デバイス {self.source} をオープンしました (試行 {attempt}回目)")
                return
            print(f"[Camera] デバイスビジー: device busy (試行 {attempt}/{MAX_RETRY}). "
                  f"{RETRY_INTERVAL}秒後にリトライします...")
            self.cap.release()
            self.cap = None
            time.sleep(RETRY_INTERVAL)

        raise RuntimeError(
            f"[Camera] カメラデバイス {self.source} を開けませんでした。"
            " 他のプロセスがカメラを使用中の可能性があります。\n"
            " → `sudo fuser /dev/video0` で確認後、プロセスを終了してください。")

    def _reset(self):
        """フレーム取得エラー時にデバイスをリセットする。"""
        print("[Camera] デバイスをリセット中...")
        with self.lock:
            self.frame = None
        self._open()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        consecutive_failures = 0
        MAX_FAILURES = 10  # 連続失敗許容回数

        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                try:
                    self._reset()
                    consecutive_failures = 0
                except RuntimeError as e:
                    print(e)
                    break

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"[Camera] フレーム取得失敗 ({consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    print("[Camera] 連続失敗のためデバイスをリセットします。")
                    try:
                        self._reset()
                        consecutive_failures = 0
                    except RuntimeError as e:
                        print(e)
                        break
                time.sleep(0.5)
                continue

            consecutive_failures = 0
            with self.lock:
                self.frame = frame
            time.sleep(0.01)  # CPU負荷軽減

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
