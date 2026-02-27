import os
import time
import cv2
import numpy as np
import json
from flask import Blueprint, render_template_string, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from detector import HumanDetector

model_test_bp = Blueprint('model_test', __name__)

# --- UI Template ---
MODEL_TEST_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI モデルテストツール</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0d11; --surface: #161922; --border: #2a2d3a;
            --accent: #4f8ef7; --accent2: #34d399; --danger: #f87171;
            --text: #e2e8f0; --muted: #8892a4;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
        
        header {
            background: var(--surface); border-bottom: 1px solid var(--border);
            padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;
            position: sticky; top: 0; z-index: 100;
        }
        .btn-back {
            text-decoration: none; color: var(--text); background: var(--border);
            padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
            transition: all 0.2s;
        }
        .btn-back:hover { background: #3a3f50; }

        .container { max-width: 1100px; margin: 30px auto; padding: 0 20px; }
        
        .grid { display: grid; grid-template-columns: 350px 1fr; gap: 24px; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; height: fit-content; }
        .card-header { padding: 16px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 0.9rem; color: var(--muted); }
        .card-body { padding: 20px; }

        /* Upload Area */
        .upload-zone {
            border: 2px dashed var(--border); border-radius: 10px; padding: 30px 20px;
            text-align: center; cursor: pointer; transition: all 0.2s;
            background: rgba(255,255,255,0.02); margin-top: 10px;
        }
        .upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: rgba(79, 142, 247, 0.05); }
        .upload-zone p { font-size: 0.85rem; color: var(--muted); margin-bottom: 10px; }
        .upload-zone .icon { font-size: 2rem; margin-bottom: 10px; display: block; }

        .file-input { display: none; }
        .status-badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-top: 10px; }
        .status-ready { background: rgba(52, 211, 153, 0.15); color: var(--accent2); }
        .status-empty { background: rgba(136, 146, 164, 0.15); color: var(--muted); }

        /* Controls */
        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; }
        input[type=number], input[type=range] {
            width: 100%; background: var(--bg); border: 1px solid var(--border);
            border-radius: 8px; color: var(--text); padding: 8px 12px; outline: none;
        }

        .btn-run {
            width: 100%; padding: 12px; border: none; border-radius: 8px;
            background: var(--accent); color: white; font-weight: 700; cursor: pointer;
            transition: all 0.2s; margin-top: 10px;
        }
        .btn-run:hover { background: #3a72d4; transform: translateY(-1px); }
        .btn-run:active { transform: translateY(0); }
        .btn-run:disabled { background: var(--border); cursor: not-allowed; opacity: 0.6; }

        /* Result View */
        .result-view { position: relative; background: #000; border-radius: 12px; overflow: hidden; min-height: 400px; display: flex; align-items: center; justify-content: center; }
        #result-img { max-width: 100%; max-height: 70vh; display: block; }
        .loading-overlay {
            position: absolute; inset: 0; background: rgba(0,0,0,0.7);
            display: none; flex-direction: column; align-items: center; justify-content: center; z-index: 10;
        }
        .spinner {
            width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.1);
            border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .stats-overlay {
            position: absolute; top: 16px; left: 16px; background: rgba(15, 17, 23, 0.7);
            backdrop-filter: blur(8px); padding: 8px 12px; border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; pointer-events: none;
        }
        .stat-item { display: flex; gap: 10px; margin-bottom: 2px; }
        .stat-label { color: var(--muted); }

        .class-list { margin-top: 15px; }
        .class-pill {
            display: inline-block; padding: 4px 10px; border-radius: 15px;
            background: var(--border); font-size: 0.75rem; margin-right: 5px; margin-bottom: 5px;
        }

        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>

    <header>
        <div style="display:flex; align-items:center; gap:15px;">
            <a href="/" class="btn-back">◀ 監視画面へ</a>
            <h1 style="font-size:1.1rem; font-weight:700;">🤖 AIモデル テストツール</h1>
        </div>
        <div id="model-name" style="font-size:0.8rem; color:var(--muted);">Model: 未ロード</div>
    </header>

    <div class="container">
        <div class="grid">
            <!-- Sidebar -->
            <div class="card">
                <div class="card-header">アップロード & 設定</div>
                <div class="card-body">
                    
                    <div class="form-group">
                        <label>TFLite モデル (.tflite)</label>
                        <div class="upload-zone" id="drop-model" onclick="document.getElementById('file-model').click()">
                            <span class="icon">📦</span>
                            <p id="model-filename">モデルをドロップまたは参照</p>
                            <input type="file" id="file-model" class="file-input" accept=".tflite">
                            <span id="model-status" class="status-badge status-empty">未選択</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>テスト画像/動画 (JPG, PNG, MP4)</label>
                        <div class="upload-zone" id="drop-media" onclick="document.getElementById('file-media').click()">
                            <span class="icon">🖼️</span>
                            <p id="media-filename">メディアをドロップまたは参照</p>
                            <input type="file" id="file-media" class="file-input" accept="image/*,video/*">
                            <span id="media-status" class="status-badge status-empty">未選択</span>
                        </div>
                    </div>

                    <div class="section-title" style="margin:20px 0 10px; font-size:0.75rem; color:var(--muted); text-transform:uppercase;">パラメータ</div>
                    <div class="form-group">
                        <label>検知閾値 (Threshold)</label>
                        <div style="display:flex; align-items:center; gap:12px;">
                            <input type="range" id="param-threshold" min="0.1" max="0.95" step="0.05" value="0.5" 
                                   oninput="document.getElementById('val-threshold').textContent = this.value">
                            <span id="val-threshold" style="color:var(--accent); font-weight:700; min-width:30px;">0.5</span>
                        </div>
                    </div>

                    <button id="btn-process" class="btn-run" disabled onclick="runTest()">テスト実行</button>
                    <p style="font-size:0.68rem; color:var(--muted); margin-top:10px; text-align:center;">処理結果は tmp_test フォルダに保存されます。</p>
                </div>
            </div>

            <!-- Main Panel -->
            <div class="card" style="min-height:500px;">
                <div class="card-header">処理結果プレビュー</div>
                <div class="card-body" style="padding:0;">
                    <div class="result-view" id="result-container">
                        <div id="empty-state" style="color:var(--muted); text-align:center;">
                            <span style="font-size:3rem; display:block; margin-bottom:10px;">🔍</span>
                            <p>ファイルをアップロードしてテストを開始してください</p>
                        </div>
                        
                        <img id="result-img" style="display:none;">
                        <video id="result-video" style="display:none; max-width:100%; max-height:70vh;" controls></video>

                        <div id="stats-panel" class="stats-overlay" style="display:none;">
                            <div class="stat-item"><span class="stat-label">推論時間:</span><span id="stat-inference">-</span></div>
                            <div class="stat-item"><span class="stat-label">検知数:</span><span id="stat-count">-</span></div>
                            <div class="class-list" id="stat-classes"></div>
                        </div>

                        <div class="loading-overlay" id="loader">
                            <div class="spinner"></div>
                            <p style="margin-top:15px; font-weight:600;" id="loading-text">画像処理中...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedModel = null;
        let selectedMedia = null;

        // Drag & Drop
        ['model', 'media'].forEach(type => {
            const zone = document.getElementById('drop-' + type);
            const input = document.getElementById('file-' + type);
            
            zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                if (e.dataTransfer.files.length) handleFile(type, e.dataTransfer.files[0]);
            });
            input.addEventListener('change', () => {
                if (input.files.length) handleFile(type, input.files[0]);
            });
        });

        function handleFile(type, file) {
            if (type === 'model') {
                selectedModel = file;
                document.getElementById('model-filename').textContent = file.name;
                document.getElementById('model-status').textContent = '準備完了';
                document.getElementById('model-status').className = 'status-badge status-ready';
            } else {
                selectedMedia = file;
                document.getElementById('media-filename').textContent = file.name;
                document.getElementById('media-status').textContent = '準備完了';
                document.getElementById('media-status').className = 'status-badge status-ready';
            }
            checkReady();
        }

        function checkReady() {
            document.getElementById('btn-process').disabled = !(selectedModel && selectedMedia);
        }

        async function runTest() {
            if (!selectedModel || !selectedMedia) return;

            const loader = document.getElementById('loader');
            const empty = document.getElementById('empty-state');
            const resImg = document.getElementById('result-img');
            const resVid = document.getElementById('result-video');
            const stats = document.getElementById('stats-panel');
            const btn = document.getElementById('btn-process');

            loader.style.display = 'flex';
            empty.style.display = 'none';
            resImg.style.display = 'none';
            resVid.style.display = 'none';
            stats.style.display = 'none';
            btn.disabled = true;

            const formData = new FormData();
            formData.append('model', selectedModel);
            formData.append('media', selectedMedia);
            formData.append('threshold', document.getElementById('param-threshold').value);

            try {
                const response = await fetch('/api/test_process', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '処理に失敗しました');

                if (data.type === 'image') {
                    resImg.src = data.url + '?t=' + Date.now();
                    resImg.style.display = 'block';
                } else {
                    resVid.src = data.url + '?t=' + Date.now();
                    resVid.style.display = 'block';
                }

                document.getElementById('stat-inference').textContent = data.inference_ms.toFixed(1) + ' ms';
                document.getElementById('stat-count').textContent = data.count;
                
                const classArea = document.getElementById('stat-classes');
                classArea.innerHTML = '';
                Object.keys(data.classes).forEach(cls => {
                    const pill = document.createElement('span');
                    pill.className = 'class-pill';
                    pill.textContent = cls + ': ' + data.classes[cls];
                    classArea.appendChild(pill);
                });
                
                stats.style.display = 'block';
                document.getElementById('model-name').textContent = 'Model: ' + selectedModel.name;

            } catch (err) {
                alert('エラー: ' + err.message);
                empty.style.display = 'block';
            } finally {
                loader.style.display = 'none';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@model_test_bp.route('/model_test')
def model_test_index():
    config = {}
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    return render_template_string(MODEL_TEST_TEMPLATE, config=config)

@model_test_bp.route('/api/test_process', methods=['POST'])
def api_test_process():
    if 'model' not in request.files or 'media' not in request.files:
        return jsonify({"error": "No model or media file"}), 400
    
    model_file = request.files['model']
    media_file = request.files['media']
    threshold = float(request.form.get('threshold', 0.5))
    
    tmp_dir = current_app.config.get('TMP_TEST_FOLDER', 'tmp_test')
    os.makedirs(tmp_dir, exist_ok=True)
    
    model_path = os.path.join(tmp_dir, secure_filename(model_file.filename))
    media_path = os.path.join(tmp_dir, secure_filename(media_file.filename))
    
    model_file.save(model_path)
    media_file.save(media_path)
    
    detector = None
    try:
        detector = HumanDetector(model_path=model_path, threshold=threshold)
        ext = os.path.splitext(media_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Image Processing
            frame = cv2.imread(media_path)
            if frame is None:
                return jsonify({"error": "Failed to load image"}), 400
            
            start_t = time.time()
            detections = detector.detect(frame)
            inf_time = (time.time() - start_t) * 1000
            
            res_frame = detector.draw_detections(frame, detections)
            out_name = "res_" + secure_filename(media_file.filename)
            out_path = os.path.join(tmp_dir, out_name)
            cv2.imwrite(out_path, res_frame)
            
            class_counts = {}
            for d in detections:
                name = detector.classes.get(d[5], f"ID:{d[5]}")
                class_counts[name] = class_counts.get(name, 0) + 1
            
            return jsonify({
                "type": "image",
                "url": f"/tmp_test/{out_name}",
                "inference_ms": inf_time,
                "count": len(detections),
                "classes": class_counts
            })
            
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video Processing (Simple version for testing)
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                return jsonify({"error": "Failed to open video"}), 400
            
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 20
            
            out_name = "res_" + os.path.splitext(secure_filename(media_file.filename))[0] + ".mp4"
            out_path = os.path.join(tmp_dir, out_name)
            
            # Using H264 for browser compatibility if possible, fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            total_inf = 0
            frames = 0
            class_counts = {}
            
            # Limit to first 60 frames for quick testing in web env
            MAX_FRAMES = 60
            while frames < MAX_FRAMES:
                ret, frame = cap.read()
                if not ret: break
                
                start_t = time.time()
                detections = detector.detect(frame)
                total_inf += (time.time() - start_t)
                
                for d in detections:
                    name = detector.classes.get(d[5], f"ID:{d[5]}")
                    class_counts[name] = class_counts.get(name, 0) + 1
                
                res_frame = detector.draw_detections(frame, detections)
                out.write(res_frame)
                frames += 1
            
            cap.release()
            out.release()
            
            return jsonify({
                "type": "video",
                "url": f"/tmp_test/{out_name}",
                "inference_ms": (total_inf / max(1, frames)) * 1000,
                "count": frames,
                "classes": class_counts,
                "note": f"Processed first {frames} frames"
            })
        else:
            return jsonify({"error": "Unsupported file format"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
