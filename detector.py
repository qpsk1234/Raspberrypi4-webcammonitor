import cv2
import numpy as np
import json
import os

# TFLite ランタイムを動的にインポート（tflite_runtime または tensorflow.lite を使用）
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

# COCO データセットのクラスマップ (デフォルト値)
DEFAULT_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

PERSON_CLASS_ID = 1

class HumanDetector:
    def __init__(self, model_path='model.tflite', threshold=0.5):
        self._model_path = model_path
        self.threshold = float(threshold)
        self.interpreter = None
        self.classes = {} # Initialize as empty, will be loaded by refresh_classes
        self.refresh_classes() # Load classes from JSON

        if tflite is None:
            print("[WARNING] tflite_runtime / tensorflow が見つかりません。モック検知を使用します。")
            return

        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_height   = self.input_details[0]['shape'][1]
            self.input_width    = self.input_details[0]['shape'][2]

            # 出力テンソルのインデックスを形状とデータ型から自動判別
            self.idx_boxes   = -1
            self.idx_classes = -1
            self.idx_scores  = -1
            self.idx_count   = -1

            num_outputs = len(self.output_details)
            print(f"Model has {num_outputs} outputs.")

            for i, detail in enumerate(self.output_details):
                shape = detail['shape']
                dtype = detail['dtype']
                print(f"  Output {i}: shape={shape}, dtype={dtype}")
                
                if len(shape) == 3 and shape[2] == 4: # [1, N, 4]
                    self.idx_boxes = i
                elif len(shape) == 2: # [1, N]
                    # dtype が整数ならクラスID、浮動小数点ならスコアと判断
                    if np.issubdtype(dtype, np.integer):
                        self.idx_classes = i
                    elif np.issubdtype(dtype, np.floating):
                        # スコアが複数ある場合は最初を優先するか、後で上書き
                        if self.idx_scores == -1: self.idx_scores = i
                        else:
                            # もしクラスIDがまだ見つかっていなければ、これまでのスコア候補をクラスにする等の調整
                            if self.idx_classes == -1:
                                self.idx_classes = self.idx_scores
                                self.idx_scores = i
                elif len(shape) == 1: # [1]
                    self.idx_count = i

            # 判別できなかった場合の安全なデフォルト（配列の範囲内に収める）
            if self.idx_boxes == -1 and num_outputs > 0: self.idx_boxes = 0
            if self.idx_classes == -1 and num_outputs > 1: self.idx_classes = 1
            if self.idx_scores == -1 and num_outputs > 2: self.idx_scores = 2
            if self.idx_count == -1 and num_outputs > 3: self.idx_count = 3

            print(f"[OK] Detector initialized with model: {model_path}")
            print(f"Indices determined: boxes={self.idx_boxes}, classes={self.idx_classes}, scores={self.idx_scores}, count={self.idx_count}")
        except Exception as e:
            print(f"[WARNING] 検出器の初期化中にエラーが発生しました: {e}\nモック検知を使用します。")
            self.interpreter = None

    def _preprocess(self, frame):
        """フレームをモデル入力サイズにリサイズ＆正規化する。"""
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        # uint8 モデルの場合はそのまま、float32 モデルは 0-1 正規化
        if self.input_details[0]['dtype'] == np.float32:
            resized = (np.float32(resized) - 127.5) / 127.5
        return np.expand_dims(resized, axis=0)

    def detect(self, frame):
        """
        フレームを解析してオブジェクトを検知する。
        戻り値: list of (x, y, w, h, score, class_id) — フレーム内の絶対座標
        """
        if self.interpreter is None:
            return []

        h, w = frame.shape[:2]
        input_data = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        try:
            # 範囲チェック付きでテンソル取得
            num_ops = len(self.output_details)
            boxes = classes = scores = None
            
            if 0 <= self.idx_boxes < num_ops:
                boxes = self.interpreter.get_tensor(self.output_details[self.idx_boxes]['index'])[0]
            if 0 <= self.idx_classes < num_ops:
                classes = self.interpreter.get_tensor(self.output_details[self.idx_classes]['index'])[0]
            if 0 <= self.idx_scores < num_ops:
                scores = self.interpreter.get_tensor(self.output_details[self.idx_scores]['index'])[0]

            if boxes is None or classes is None or scores is None:
                return []

            detections = []
            count = 0
            if 0 <= self.idx_count < num_ops:
                count_tensor = self.interpreter.get_tensor(self.output_details[self.idx_count]['index'])
                count = int(count_tensor[0]) if count_tensor.size > 0 else 0
            else:
                count = len(scores)

            # 最大検知数制限（安全のため）
            count = min(count, len(boxes), len(classes), len(scores))

            for i in range(count):
                if scores[i] >= self.threshold:
                    ymin, xmin, ymax, xmax = boxes[i]
                    x = int(xmin * w)
                    y = int(ymin * h)
                    bw = int((xmax - xmin) * w)
                    bh = int((ymax - ymin) * h)
                    # (x, y, w, h, score, class_id)
                    detections.append((x, y, bw, bh, float(scores[i]), int(classes[i])))

            return detections
        except Exception as e:
            print(f"[ERROR] 推論処理中にエラーが発生しました: {e}")
            return []

    def get_model_info(self):
        """Web UI 向けにモデルの構成情報を返す。"""
        if self.interpreter is None:
            return {"status": "Failed / Not loaded", "path": "Unknown"}

        def _fmt_type(t):
            return str(t).split("'")[1] if "'" in str(t) else str(t)

        info = {
            "status": "Loaded",
            "path": getattr(self, '_model_path', 'model.tflite'),
            "classes": self.classes, # クラスマップも含める
            "input": [
                {
                    "name": d['name'],
                    "shape": d['shape'].tolist(),
                    "dtype": _fmt_type(d['dtype'])
                } for d in self.input_details
            ],
            "output": [
                {
                    "name": d['name'],
                    "shape": d['shape'].tolist(),
                    "dtype": _fmt_type(d['dtype'])
                } for d in self.output_details
            ],
            "indices": {
                "boxes": self.idx_boxes,
                "classes": self.idx_classes,
                "scores": self.idx_scores,
                "count": self.idx_count
            }
        }
        return info

    def refresh_classes(self):
        """外部 JSON ファイルからクラスマップを再読み込みする。"""
        json_path = os.path.join(os.path.dirname(__file__), 'coco_classes.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # キーを数値に変換
                    self.classes = {int(k): v for k, v in data.items()}
                print(f"[OK] Loaded {len(self.classes)} classes from {json_path}")
            except Exception as e:
                print(f"[ERROR] クラスマップのロードに失敗しました: {e}")
        else:
            print(f"[WARNING] coco_classes.json が見つかりません。デフォルトのクラスマップを使用します。")
            # Fallback to a hardcoded default if the file doesn't exist
            self.classes = {
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
                21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
                27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
                39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
                43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
                54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
                59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
                64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
                72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
            }

    def draw_detections(self, frame, detections):
        for item in detections:
            x, y, bw, bh = item[:4]
            score = item[4]
            class_id = item[5]
            
            class_name = self.classes.get(class_id, f"ID:{class_id}")
            color = (0, 220, 50) if class_id == PERSON_CLASS_ID else (220, 150, 0)
            
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label_text = f"{class_name} {int(score * 100)}%"
            cv2.putText(frame, label_text, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
