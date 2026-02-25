import cv2
import time
import threading
import datetime
import json
import os
from camera import Camera
from detector import HumanDetector
from notifier import TelegramNotifier
from recorder import Recorder
from detection_logger import DetectionLogger
import web_stream
from web_stream import run_server, system_status

def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("Starting Monitoring Camera System...")
    config = load_config()

    # 保存ディレクトリの作成
    os.makedirs(config['save_directory'], exist_ok=True)

    # モジュールの初期化
    cam      = Camera(source=config['video_source'])
    detector = HumanDetector(
        model_path=config['model_path'],
        threshold=config['detection_threshold'])
    notifier = TelegramNotifier(
        config['telegram_token'],
        config['telegram_chat_id'])
    recorder = Recorder(
        save_directory=config['save_directory'],
        resolution=(config.get('stream_width', 640), config.get('stream_height', 480)))
    logger   = DetectionLogger()

    # Webサーバーを別スレッドで起動（logger, detectorも渡す）
    web_thread = threading.Thread(
        target=run_server, args=(cam, logger, detector), daemon=True)
    web_thread.start()

    cam.start()
    print("System is running. Press 'q' to quit.")
    print("Web UI: http://0.0.0.0:5000")

    last_notify_time = 0

    try:
        while True:
            # config をループごとに再読み込み（Web管理画面の変更を即反映）
            config = load_config()
            detector.threshold = float(config.get('detection_threshold', 0.5))
            notify_interval = float(config.get('notify_interval', 30))

            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # オブジェクト検知
            all_detections = detector.detect(frame)
            
            # 設定されたターゲットクラスIDに含まれるもののみを抽出 (d[5] は class_id)
            target_classes = config.get('target_classes', [1])
            target_detections = [d for d in all_detections if d[5] in target_classes]

            # 描画処理
            draw_list = all_detections if config.get('show_all_detections', True) else target_detections
            if draw_list:
                frame = detector.draw_detections(frame, draw_list)

            if target_detections:
                # 最大スコア算出
                max_score = max((d[4] for d in target_detections), default=0.0)

                # ステータス更新（Webダッシュボード向け：ターゲット数をカウント）
                system_status['human_count']      = len(target_detections) # UI上はhuman_countのまま共有
                system_status['detections_total'] += 1
                system_status['last_detected']    = \
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 録画（ターゲットがいる時のみ）
                recorder.start_recording(frame)
                recorder.write(frame)

                # Telegram 通知
                current_time = time.time()
                if current_time - last_notify_time > notify_interval:
                    label_names = [detector.classes.get(d[5], f"ID:{d[5]}") for d in target_detections]
                    target_summary = ", ".join(list(set(label_names)))
                    print(f"Target detected ({target_summary})! Sending notification...")
                    
                    # スナップショット保存
                    snap_path = os.path.join(
                        config['save_directory'],
                        f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snap_path, frame)
                    
                    notifier.send_photo(frame, caption=f"⚠️ 検知対象を捕捉しました！\n対象: {target_summary}\n数: {len(target_detections)}")
                    
                    # ログ記録
                    logger.log(
                        human_count=len(target_detections),
                        confidence_max=max_score,
                        snapshot_path=snap_path)
                    last_notify_time = current_time
            else:
                system_status['human_count'] = 0
                if recorder.is_recording:
                    recorder.schedule_stop()

            # 最終的なフレームを Web UI ストリーム用に共有
            web_stream.latest_processed_frame = frame

            # モニター表示 (GUI)
            if config.get('use_gui', False):
                try:
                    cv2.imshow("Surveillance Camera", frame)
                except cv2.error:
                    print("Warning: GUI not available. Disabling.")
                    config['use_gui'] = False

            if config.get('use_gui', False):
                try:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    config['use_gui'] = False
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping System...")
        recorder.release()
        cam.stop()
        if config.get('use_gui', False):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

if __name__ == "__main__":
    main()
