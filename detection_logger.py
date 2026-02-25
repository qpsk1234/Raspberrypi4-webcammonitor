import csv
import os
import datetime
import threading

LOG_FILE = 'detection_log.csv'
FIELDNAMES = ['timestamp', 'human_count', 'confidence_max', 'snapshot_path']

class DetectionLogger:
    """検知イベントを CSV ファイルへ記録するモジュール。"""

    def __init__(self, log_path=LOG_FILE):
        self.log_path = log_path
        self._lock = threading.Lock()
        # ヘッダーが存在しなければ初期化
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def log(self, human_count, confidence_max=0.0, snapshot_path=''):
        """1件の検知イベントを記録する。"""
        row = {
            'timestamp':      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'human_count':    human_count,
            'confidence_max': f"{confidence_max:.3f}",
            'snapshot_path':  snapshot_path,
        }
        with self._lock:
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)

    def read_recent(self, n=50):
        """直近 n 件のログを新しい順に返す。"""
        with self._lock:
            if not os.path.exists(self.log_path):
                return []
            with open(self.log_path, 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        return list(reversed(rows[-n:]))
