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
        resolution=(config.get('recorder_width', 1280), config.get('recorder_height', 720)),
        pre_frames=config.get('recorder_pre_frames', 60))
    logger   = DetectionLogger()

    # Webサーバーを別スレッドで起動
    web_thread = threading.Thread(
        target=run_server, args=(cam, logger, detector, notifier), daemon=True)
    web_thread.start()

    cam.start()
    print("System is running. Press 'q' to quit.")
    print("Web UI: http://0.0.0.0:5000")

    detection_session_start = None
    last_target_time = 0
    session_notified = False 
    
    # 遅延通知用バッファ
    pending_notification = {
        "frame": None,
        "summary": "",
        "max_score": 0.0,
        "human_count": 0,
        "video_path": None
    }

    def process_deferred_notification(notif_data, current_config):
        """録画終了後に別スレッドで実行される通知処理"""
        try:
            mode = current_config.get('telegram_notify_mode', 'photo')
            save_dir = current_config['save_directory']
            
            # 1. 静止画の保存
            snap_w = current_config.get('snapshot_width', 1280)
            snap_h = current_config.get('snapshot_height', 720)
            snap_frame = cv2.resize(notif_data["frame"], (snap_w, snap_h))
            
            snap_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            snap_path = os.path.join(save_dir, f"snap_{snap_ts}.jpg")
            cv2.imwrite(snap_path, snap_frame)
            
            # 2. Telegram送信
            if mode != "none":
                caption = f"⚠️ 検知通知\n対象: {notif_data['summary']}\n数: {notif_data['human_count']}\n時刻: {datetime.datetime.now().strftime('%H:%M:%S')}"
                
                if mode in ["photo", "both"]:
                    notifier.send_photo(snap_frame, caption=caption)
                
                if mode in ["video", "both"]:
                    # 録画ファイルが確定するまで少し待機（FFmpegの書き出し完了待ち）
                    time.sleep(1.0) 
                    if notif_data["video_path"] and os.path.exists(notif_data["video_path"]):
                        notifier.send_video(notif_data["video_path"], caption=f"📹 録画ファイル: {notif_data['summary']}")
            
            # 3. ログ記録 (動画パスを含める)
            logger.log(
                human_count=notif_data["human_count"],
                confidence_max=notif_data["max_score"],
                snapshot_path=snap_path,
                video_path=notif_data["video_path"])
                
            print(f"[Main] Deferred notification processed successfully (Mode: {mode})")
        except Exception as e:
            print(f"[Error] process_deferred_notification: {e}")

    try:
        while True:
            current_config = load_config()
            detector.threshold = float(current_config.get('detection_threshold', 0.5))
            post_seconds = float(current_config.get('recorder_post_seconds', 5))
            
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            all_detections = detector.detect(frame)
            target_classes = current_config.get('target_classes', [1])
            target_detections = [d for d in all_detections if d[5] in target_classes]

            if target_detections:
                frame = detector.draw_detections(frame, all_detections if current_config.get('show_all_detections', True) else target_detections)

            recorder.update_buffer(frame)
            recorder.write(frame)

            if target_detections:
                last_target_time = time.time()
                max_score = max((d[4] for d in target_detections), default=0.0)

                system_status['human_count'] = len(target_detections) 
                if len(target_detections) > system_status.get('human_count_max', 0):
                    system_status['human_count_max'] = len(target_detections)
                
                system_status['detections_total'] += 1
                system_status['last_detected'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if detection_session_start is None:
                    detection_session_start = time.time()
                
                elapsed_ms = (time.time() - detection_session_start) * 1000
                delay_ms = current_config.get('recorder_start_delay_ms', 0)
                
                if elapsed_ms >= delay_ms:
                    recorder.start_recording(frame)

                # 通知データの保持（セッション内で一度だけ、最良の瞬間のフレームを確保）
                if not session_notified and elapsed_ms >= delay_ms:
                    label_names = [detector.classes.get(d[5], f"ID:{d[5]}") for d in target_detections]
                    target_summary = ", ".join(list(set(label_names)))
                    
                    # メモリにバッファリング
                    pending_notification["frame"] = frame.copy() # コピーして保持
                    pending_notification["summary"] = target_summary
                    pending_notification["max_score"] = max_score
                    pending_notification["human_count"] = len(target_detections)
                    pending_notification["video_path"] = recorder.current_video_path # パスを保持
                    
                    session_notified = True
                    print(f"[Main] Detection Buffered. Will notify after recording ends.")
            else:
                system_status['human_count'] = 0
                if recorder.is_recording:
                    recorder.schedule_stop(post_seconds)
                
                # セッション終了（ポスト録画分が経過）
                if session_notified and (time.time() - last_target_time > post_seconds):
                    # 録画が終了し、かつ通知待ちデータがある場合
                    # バックグラウンドスレッドで通知処理を実行
                    notif_thread = threading.Thread(
                        target=process_deferred_notification, 
                        args=(pending_notification.copy(), current_config),
                        daemon=True
                    )
                    notif_thread.start()
                    
                    # フラグとバッファをリセット
                    detection_session_start = None
                    session_notified = False
                    pending_notification["frame"] = None

            web_stream.latest_processed_frame = frame

            if current_config.get('use_gui', False):
                try:
                    cv2.imshow("Surveillance Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                except cv2.error:
                    current_config['use_gui'] = False
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        recorder.release()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
