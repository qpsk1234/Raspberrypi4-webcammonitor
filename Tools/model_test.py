import cv2
import numpy as np
import sys
import os
import argparse
import time

# プロジェクトルートをパスに追加して detector をロードできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import HumanDetector

def test_image(detector, image_path, output_path):
    print(f"\n[Image Test] Processing: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    start_t = time.time()
    detections = detector.detect(frame)
    end_t = time.time()

    print(f"Inference Time: {(end_t - start_t)*1000:.2f} ms")
    print(f"Detections: {len(detections)}")
    for i, d in enumerate(detections):
        x, y, w, h, score, cid = d
        class_name = detector.classes.get(cid, f"ID:{cid}")
        print(f"  {i+1}: {class_name} ({score:.2f}) at [{x}, {y}, {w}, {h}]")

    # 結果の描画
    result_frame = detector.draw_detections(frame.copy(), detections)
    cv2.imwrite(output_path, result_frame)
    print(f"Result saved to: {output_path}")

def test_video(detector, video_path, output_path):
    print(f"\n[Video Test] Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力ビデオ設定 (.mp4 / avc1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    total_inf_time = 0
    class_counts = {}

    print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_t = time.time()
        detections = detector.detect(frame)
        total_inf_time += (time.time() - start_t)

        for d in detections:
            cid = d[5]
            name = detector.classes.get(cid, f"ID:{cid}")
            class_counts[name] = class_counts.get(name, 0) + 1

        result_frame = detector.draw_detections(frame, detections)
        out.write(result_frame)

        count += 1
        if count % 30 == 0:
            sys.stdout.write(f"\rProgress: {count}/{total_frames} frames...")
            sys.stdout.flush()

    cap.release()
    out.release()
    print(f"\nDone! Result saved to: {output_path}")
    print(f"Average Inference Time: {(total_inf_time/max(1,count))*1000:.2f} ms")
    print("Class Statistics (Total occurrences across all frames):")
    for name, c in class_counts.items():
        print(f"  {name}: {c}")

def main():
    parser = argparse.ArgumentParser(description="TFLite Model Testing Tool")
    parser.add_argument("--model", type=str, default="model.tflite", help="Path to .tflite model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--output", type=str, help="Path to output result (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold (0.0 - 1.0)")
    args = parser.parse_args()

    # パスの正規化
    model_path = os.path.abspath(args.model)
    input_path = os.path.abspath(args.input)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(input_path):
        print(f"Error: Input not found at {input_path}")
        return

    # 出力パスの決定
    if not args.output:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_result{ext}"
    else:
        output_path = args.output

    # 検出器の初期化
    print(f"Initializing Detector with {model_path} (Threshold: {args.threshold})")
    detector = HumanDetector(model_path=model_path, threshold=args.threshold)
    
    # 入力ファイル形式の判定
    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']

    if ext in image_exts:
        test_image(detector, input_path, output_path)
    elif ext in video_exts:
        test_video(detector, input_path, output_path)
    else:
        print(f"Error: Unsupported file format {ext}")

if __name__ == "__main__":
    main()
