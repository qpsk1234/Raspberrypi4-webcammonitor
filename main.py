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

    # Webサーバーを別スレッドで起動（logger, detector, notifierも渡す）
    web_thread = threading.Thread(
        target=run_server, args=(cam, logger, detector, notifier), daemon=True)
    web_thread.start()

    cam.start()
    print("System is running. Press 'q' to quit.")
    print("Web UI: http://0.0.0.0:5000")

    last_notify_time = 0
    detection_session_start = None
    last_target_time = 0 # 最後にターゲットを検知した時刻
    session_notified = False  # セッション内で一度だけ通知するためのフラグ

    try:
        while True:
            # config をループごとに再読み込み
            current_config = load_config()
            detector.threshold = float(current_config.get('detection_threshold', 0.5))
            post_seconds = float(current_config.get('recorder_post_seconds', 5))
            
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # オブジェクト検知
            all_detections = detector.detect(frame)
            
            target_classes = current_config.get('target_classes', [1])
            target_detections = [d for d in all_detections if d[5] in target_classes]

            # 描画処理
            draw_list = all_detections if current_config.get('show_all_detections', True) else target_detections
            if draw_list:
                frame = detector.draw_detections(frame, draw_list)

            # プリ録画バッファの更新（常時）
            recorder.update_buffer(frame)

            # 録画の書き込み（録画中であれば毎フレーム実行）
            recorder.write(frame)

            if target_detections:
                last_target_time = time.time()
                # 最大スコア算出
                max_score = max((d[4] for d in target_detections), default=0.0)

                # ステータス更新
                system_status['human_count'] = len(target_detections) 
                if len(target_detections) > system_status.get('human_count_max', 0):
                    system_status['human_count_max'] = len(target_detections)
                
                system_status['detections_total'] += 1
                system_status['last_detected'] = \
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 録画開始遅延の判定
                if detection_session_start is None:
                    detection_session_start = time.time()
                
                elapsed_ms = (time.time() - detection_session_start) * 1000
                delay_ms = current_config.get('recorder_start_delay_ms', 0)
                
                # 録画開始
                if elapsed_ms >= delay_ms:
                    recorder.start_recording(frame)

                # 通知とログ記録（セッション内で一度だけ実行）
                if not session_notified and elapsed_ms >= delay_ms:
                    label_names = [detector.classes.get(d[5], f"ID:{d[5]}") for d in target_detections]
                    target_summary = ", ".join(list(set(label_names)))
                    print(f"Target detected ({target_summary})! Sending session notification...")
                    
                    # スナップショット保存
                    snap_w = current_config.get('snapshot_width', 1280)
                    snap_h = current_config.get('snapshot_height', 720)
                    snap_frame = cv2.resize(frame, (snap_w, snap_h))
                    
                    snap_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    snap_path = os.path.join(current_config['save_directory'], f"snap_{snap_ts}_start.jpg")
                    cv2.imwrite(snap_path, snap_frame)
                    
                    # 通知
                    notifier.send_photo(snap_frame, caption=f"⚠️ 検知（開始）: {target_summary}\n数: {len(target_detections)}")
                    
                    # ログ記録
                    logger.log(
                        human_count=len(target_detections),
                        confidence_max=max_score,
                        snapshot_path=snap_path,
                        video_path=recorder.current_video_path)
                    
                    session_notified = True
                    last_notify_time = time.time()
            else:
                system_status['human_count'] = 0
                
                # 検知が途切れたときの判定
                if recorder.is_recording:
                    # 既に録画中の場合、停止をスケジュール
                    recorder.schedule_stop(post_seconds)
                
                # セッションのリセット判定：最後に検知してから post_seconds 以上経過した場合のみ
                # これにより、一瞬の検知漏れで通知が複数回飛ぶのを防ぐ
                if time.time() - last_target_time > post_seconds:
                    if session_notified and current_config.get('snapshot_mode') == 'both':
                        # 検知終了時（ポスト録画終了直前）のスナップショットと通知
                        snap_w = current_config.get('snapshot_width', 1280)
                        snap_h = current_config.get('snapshot_height', 720)
                        snap_frame = cv2.resize(frame, (snap_w, snap_h))
                        snap_path = os.path.join(
                            current_config['save_directory'],
                            f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_end.jpg")
                        cv2.imwrite(snap_path, snap_frame)
                        notifier.send_photo(snap_frame, caption=f"ℹ️ 検知終了\n最終確認時刻: {datetime.datetime.now().strftime('%H:%M:%S')}")

                    detection_session_start = None
                    session_notified = False

            # 最終的なフレームを Web UI ストリーム用に共有
            web_stream.latest_processed_frame = frame

            # モニター表示 (GUI)
            if current_config.get('use_gui', False):
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
