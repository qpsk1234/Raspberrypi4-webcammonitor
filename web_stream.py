from flask import Flask, Response, render_template_string, request, jsonify, redirect, url_for, send_from_directory
from functools import wraps
import cv2
import json
import os
import datetime
import threading
import shutil
import time
from werkzeug.utils import secure_filename
from detector import HumanDetector

app = Flask(__name__)
camera_instance = None
logger_instance = None
detector_instance = None  # HumanDetector をここで保持
system_status = {
    "running": True,
    "detections_total": 0,
    "last_detected": "—",
    "fps": 0,
    "human_count": 0,
    "stream_width": 640,
    "stream_height": 480,
}
latest_processed_frame = None  # 加工済みフレームの共有用 (JSONシリアライズ対象外)

UPLOAD_FOLDER = 'Uploads'
TMP_TEST_FOLDER = 'tmp_test'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TMP_TEST_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TMP_TEST_FOLDER'] = TMP_TEST_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100MB limit

# ============================================================
# HTML テンプレート
# ============================================================
TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>監視カメラ管理画面</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
      --accent: #4f8ef7; --accent2: #34d399; --danger: #f87171;
      --text: #e2e8f0; --muted: #8892a4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; overflow-x: hidden; }

    /* ヘッダー */
    header {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 12px 24px; display: flex; align-items: center; gap: 12px; position: sticky; top: 0; z-index: 100;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent2); animation: pulse 1.8s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
    header h1 { font-size: 1rem; font-weight: 600; }
    .header-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
    #clock { font-size: 0.8rem; color: var(--muted); }
    .btn-nav {
      display: flex; align-items: center; gap: 6px; padding: 7px 16px;
      background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px;
      font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: all .2s;
      text-decoration: none;
    }
    .btn-nav:hover { background: var(--border); }
    .btn-settings {
      display: flex; align-items: center; gap: 6px; padding: 7px 16px;
      background: var(--accent); color: #fff; border: none; border-radius: 8px;
      font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: background .2s;
    }
    .btn-settings:hover { background: #3a72d4; }
    .btn-settings.active { background: #e25555; }

    /* メインレイアウト */
    .main { display: grid; grid-template-columns: 2fr 1fr; gap: 18px; padding: 18px 24px; }

    /* カード */
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    .card-header {
      padding: 10px 16px; font-size: 0.8rem; font-weight: 600; color: var(--muted);
      border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: .05em;
    }
    .card-body { padding: 16px; }

    /* ストリーム */
    #stream-img { width: 100%; display: block; }
    .badge-row { padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 4px; background: #12141c; }
    .badge {
      background: rgba(255,255,255,.07); border: 1px solid var(--border);
      border-radius: 5px; padding: 3px 9px; font-size: 0.73rem; color: var(--text);
    }
    .badge.alert { background: rgba(248,113,113,.15); border-color: var(--danger); color: var(--danger); animation: flash 1s infinite; }
    @keyframes flash { 0%,100%{opacity:1} 50%{opacity:.45} }

    /* ステータス */
    .stat { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid var(--border); }
    .stat:last-child { border-bottom: none; }
    .stat-label { font-size: 0.83rem; color: var(--muted); }
    .stat-value { font-size: 0.88rem; font-weight: 600; }
    .green { color: var(--accent2); } .red { color: var(--danger); } .blue { color: var(--accent); }

    /* ログテーブル */
    .log-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    .log-table th { background: #12141c; padding: 7px 10px; text-align: left; color: var(--muted); font-weight: 600; }
    .log-table td { padding: 6px 10px; border-top: 1px solid var(--border); }
    .log-table tr:hover td { background: rgba(79,142,247,.06); }

    /* ======= 設定サイドパネル ======= */
    #settings-overlay {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,.5);
      z-index: 200; backdrop-filter: blur(2px);
    }
    #settings-overlay.open { display: block; }
    #settings-panel {
      position: fixed; top: 0; right: -480px; width: 460px; max-width: 95vw;
      height: 100%; background: var(--surface); border-left: 1px solid var(--border);
      z-index: 201; overflow-y: auto; transition: right .3s cubic-bezier(.4,0,.2,1);
      padding: 0;
    }
    #settings-panel.open { right: 0; }
    .panel-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 20px; border-bottom: 1px solid var(--border);
      background: #13161f; position: sticky; top: 0; z-index: 1;
    }
    .panel-header h2 { font-size: 0.95rem; font-weight: 600; }
    .close-btn {
      background: none; border: none; color: var(--muted); font-size: 1.4rem;
      cursor: pointer; line-height: 1; padding: 2px 6px; border-radius: 6px;
    }
    .close-btn:hover { background: var(--border); color: var(--text); }

    /* タブ */
    .tab-bar { display: flex; border-bottom: 1px solid var(--border); background: #13161f; }
    .nav-item {
      flex: 1; padding: 10px 0; text-align: center; font-size: 0.8rem; font-weight: 600;
      color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent;
      transition: all .2s; background: none; border-top: none; border-left: none; border-right: none;
    }
    .nav-item:hover { background: rgba(255,255,255,0.05); }
    .nav-item.active { color: var(--accent); border-bottom-color: var(--accent); }
    .tab-content { display: none; padding: 20px; }
    .tab-content.active { display: block; }

    /* モデル情報テーブル */
    .model-info-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 10px; }
    .model-info-table th { text-align: left; color: var(--muted); padding: 6px 8px; width: 30%; }
    .model-info-table td { padding: 6px 8px; border-bottom: 1px solid var(--border); }
    .tensor-list { list-style: none; font-family: monospace; font-size: 0.75rem; background: var(--bg); padding: 8px; border-radius: 6px; }
    .tensor-item { margin-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 2px; }
    .tensor-item:last-child { border-bottom: none; }

    /* フォーム部品 */
    label { font-size: 0.78rem; color: var(--muted); display: block; margin-bottom: 5px; }
    input[type=text], input[type=number], input[type=range] {
      width: 100%; background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); padding: 8px 12px;
      font-size: 0.88rem; outline: none; transition: border-color .2s;
    }
    input:focus { border-color: var(--accent); }
    input[type=range] { padding: 4px 0; cursor: pointer; }
    .range-row { display: flex; align-items: center; gap: 10px; }
    .range-row input { flex: 1; }
    .range-val { font-size: 0.85rem; font-weight: 600; color: var(--accent); min-width: 36px; text-align: right; }
    .form-group { margin-bottom: 14px; }
    .btn-save {
      width: 100%; padding: 9px; border: none; border-radius: 8px;
      background: var(--accent); color: #fff; font-size: 0.85rem; font-weight: 600;
      cursor: pointer; transition: background .2s; margin-top: 4px;
    }
    .btn-save:hover { background: #3a72d4; }
    .save-msg { display: none; color: var(--accent2); font-size: 0.78rem; margin-top: 6px; text-align: center; }
    .section-title { font-size: 0.75rem; font-weight: 700; color: var(--muted); text-transform: uppercase;
      letter-spacing: .06em; margin: 18px 0 10px; border-top: 1px solid var(--border); padding-top: 14px; }
    .section-title:first-child { margin-top: 0; border-top: none; padding-top: 0; }

    @media (max-width: 880px) { .main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>

  <!-- ヘッダー -->
  <header>
    <div class="dot"></div>
    <h1>🎥 監視カメラ管理画面</h1>
    <div class="header-right">
      <span id="clock"></span>
      <a href="/media" class="btn-nav">📂 メディア閲覧</a>
      <button class="btn-settings" id="btn-settings" onclick="toggleSettings()">⚙ 設定</button>
    </div>
  </header>

  <!-- メイン -->
  <div class="main">
    <!-- 左列 -->
    <div style="display:flex;flex-direction:column;gap:16px;">

      <!-- ライブ映像 -->
      <div class="card">
        <div class="card-header">📡 ライブ映像</div>
        <img id="stream-img" src="{{ url_for('video_feed') }}" alt="camera stream">
        <div class="badge-row">
          <span class="badge" id="badge-fps">FPS: —</span>
          <span class="badge" id="badge-res">解像度: —</span>
          <span class="badge" id="badge-count">累計検知: —</span>
          <span class="badge" id="badge-last">最終検知: —</span>
          <span class="badge" id="badge-alert" style="display:none">⚠ 人間検知中！</span>
        </div>
      </div>

      <!-- ステータス -->
      <div class="card">
        <div class="card-header">📊 動作ステータス</div>
        <div class="card-body">
          <div class="stat"><span class="stat-label">システム状態</span><span class="stat-value green" id="st-running">稼働中</span></div>
          <div class="stat"><span class="stat-label">フレーム内の人数</span><span class="stat-value blue" id="st-humans">0</span></div>
          <div class="stat"><span class="stat-label">累計検知回数</span><span class="stat-value" id="st-total">0</span></div>
          <div class="stat"><span class="stat-label">最終検知日時</span><span class="stat-value" id="st-last">—</span></div>
          <div class="stat"><span class="stat-label">FPS</span><span class="stat-value" id="st-fps">—</span></div>
          <div class="stat"><span class="stat-label">ストリーム解像度</span><span class="stat-value" id="st-res">—</span></div>
        </div>
      </div>

      <!-- ログ -->
      <div class="card">
        <div class="card-header" style="display:flex; justify-content:space-between; align-items:center;">
          <div>📋 検知ログ履歴 <span id="log-date-display" style="font-size:0.8rem; color:var(--accent2); margin-left:10px;"></span></div>
          <div style="display:flex; gap:5px;">
            <button class="btn" style="padding:2px 8px; font-size:0.7rem;" onclick="changeLogDate(-1)">◀ 前日</button>
            <button class="btn" style="padding:2px 8px; font-size:0.7rem;" onclick="changeLogDate(0)">今日</button>
            <button class="btn" style="padding:2px 8px; font-size:0.7rem;" onclick="changeLogDate(1)">翌日 ▶</button>
          </div>
        </div>
        <div style="overflow-x:auto;">
          <table class="log-table">
            <thead><tr><th>日時</th><th>検知数</th><th>確信度</th><th>メディア</th></tr></thead>
            <tbody id="log-body">
              <tr><td colspan="4" style="text-align:center;color:var(--muted);padding:14px">データなし</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- 右列: クイックガイド -->
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="card" style="background:linear-gradient(135deg,#1a2040,#1a1d27);">
        <div class="card-header">🛡 クイックガイド</div>
        <div class="card-body" style="font-size:0.82rem;line-height:1.8;color:var(--muted);">
          <p>右上の <b style="color:var(--accent)">⚙ 設定</b> ボタンをクリックすると設定パネルが開きます。</p>
          <br>
          <p><b style="color:var(--text)">設定タブの内容</b></p>
          <ul style="padding-left:1.2em;margin-top:6px;display:flex;flex-direction:column;gap:4px;">
            <li>🔍 <b>検知設定</b> — 閾値・通知間隔・解像度</li>
            <li>📨 <b>Telegram</b> — Bot Token / Chat ID</li>
            <li>🤖 <b>モデル</b> — ロード状態・テンソル詳細</li>
            <li>🔐 <b>認証</b> — ログインID/パスワード変更</li>
          </ul>
          <br>
          <p>設定変更は即座に <code style="background:#0f1117;padding:1px 5px;border-radius:4px;">config.json</code> へ保存されます。</p>
        </div>
      </div>
    </div>
  </div>

  <!-- ======= 設定サイドパネル ======= -->
  <div id="settings-overlay" onclick="closeSettings()"></div>
  <div id="settings-panel">
    <div class="panel-header">
      <h2>⚙ 設定</h2>
      <button class="close-btn" onclick="closeSettings()">✕</button>
    </div>

    <!-- タブ -->
    <div class="tab-bar">
      <button id="nav-detect" class="nav-item active" onclick="switchTab('detect')">📹 検知</button>
      <button id="nav-classes" class="nav-item" onclick="switchTab('classes')">🍱 クラス</button>
      <button id="nav-recorder" class="nav-item" onclick="switchTab('recorder')">🎬 録画・保存</button>
      <button id="nav-telegram" class="nav-item" onclick="switchTab('telegram')">✈️ Telegram</button>
      <button id="nav-auth" class="nav-item" onclick="switchTab('auth')">🔐 認証</button>
      <button id="nav-model" class="nav-item" onclick="switchTab('model')">🤖 モデル</button>
      <button id="nav-test" class="nav-item" onclick="switchTab('test')">🧪 検証・テスト</button>
    </div>

    <!-- 検知設定タブ -->
    <div id="tab-detect" class="tab-content active">
      <form id="form-detect">
        <div class="section-title">検知パラメータ</div>
        <div class="form-group">
          <label>検知感度（レベル）</label>
          <div style="font-size:0.7rem; color:var(--muted); margin-bottom:8px;">
            高いほど敏感になります（低い確信度でも枠を表示）。
          </div>
          <div class="range-row">
            <span style="font-size:0.7rem;">鈍感</span>
            <input type="range" name="detection_threshold" min="0.1" max="0.95" step="0.05"
              value="{{ 1.05 - config.detection_threshold }}"
              oninput="this.parentElement.querySelector('.range-val').textContent = (parseFloat(this.value)*10).toFixed(1)">
            <span style="font-size:0.7rem;">敏感</span>
            <span class="range-val" style="display:none">{{ (1.05 - config.detection_threshold) * 10 }}</span>
          </div>
        </div>
        <div class="form-group">
          <label>通知間隔（秒）</label>
          <input type="number" name="notify_interval" value="{{ config.notify_interval }}" min="10" max="3600">
        </div>
        <div class="section-title">ストリーム解像度</div>
        <div class="form-group">
          <label>幅 (px)</label>
          <input type="number" name="stream_width" value="{{ config.get('stream_width', 640) }}" min="320" max="1920" step="80">
        </div>
        <div class="form-group">
          <label>高さ (px)</label>
          <input type="number" name="stream_height" value="{{ config.get('stream_height', 480) }}" min="240" max="1080" step="60">
        </div>
        <button type="button" class="btn primary" onclick="saveForm('form-detect', 'msg-detect')">保存</button>
        <div id="msg-detect" class="success-msg">✅ 保存しました</div>
      </form>
    </div>

    <!-- 🍱 クラスマップ設定 -->
    <div id="tab-classes" class="tab-content" style="max-height: 400px; overflow-y: auto;">
      <form id="form-classes">
        <div class="section-title">検知・表示設定</div>
        <div class="form-group checkbox-group">
          <label>ターゲット以外も表示</label>
          <input type="checkbox" name="show_all_detections" {% if config.show_all_detections %}checked{% endif %} 
                 style="width:auto; margin-left:10px;">
          <div style="font-size:0.7rem; color:var(--muted); margin-top:4px;">
            オフにすると、選択したターゲット以外の枠が表示されなくなります。
          </div>
        </div>

        <div class="section-title">ターゲットクラス選択</div>
        <div style="font-size:0.7rem; color:var(--muted); margin-bottom:10px;">
          チェックを入れたクラスが検知・通知・録画の対象になります。<br>
          ラベル名を日本語などに書き換えることも可能です。
        </div>

        <table style="width:100%; font-size:0.8rem; border-collapse:collapse;">
          <thead>
            <tr style="border-bottom:1px solid var(--border);">
              <th style="padding:5px; text-align:left;">対象</th>
              <th style="padding:5px; text-align:left;">ID</th>
              <th style="padding:5px; text-align:left;">ラベル名</th>
            </tr>
          </thead>
          <tbody id="classes-list-area">
            <!-- JSで動的に構築 -->
          </tbody>
        </table>
        
        <div style="margin-top:15px;">
          <button type="button" class="btn primary" onclick="saveClasses()">設定を保存</button>
          <div id="msg-classes" class="success-msg">✅ 保存しました</div>
        </div>
      </form>
    </div>

    <!-- 🎬 録画・保存設定 -->
    <div id="tab-recorder" class="tab-content">
      <form id="form-recorder">
        <div class="section-title">録画設定</div>
        <div class="form-group">
          <label>ポスト録画（秒）</label>
          <div style="font-size:0.7rem; color:var(--muted); margin-bottom:8px;">
            物体が消えた後、何秒間録画を継続するか指定します。
          </div>
          <input type="number" name="recorder_post_seconds" value="{{ config.get('recorder_post_seconds', 5) }}" min="0" max="60">
        </div>

        <div class="form-group">
          <label>録画解像度（横x縦）</label>
          <div style="font-size:0.7rem; color:var(--muted); margin-bottom:8px;">
            録画データの解像度を指定します（1280x720 推奨）。
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <input type="number" name="recorder_width" value="{{ config.get('recorder_width', 1280) }}" min="320" max="1920" step="80" style="flex:1">
            <span>x</span>
            <input type="number" name="recorder_height" value="{{ config.get('recorder_height', 720) }}" min="240" max="1080" step="60" style="flex:1">
          </div>
        </div>

        <div class="form-group">
          <label>録画開始遅延 (ミリ秒)</label>
          <div style="font-size:0.7rem; color:var(--muted); margin-bottom:8px;">
            検知した瞬間のノイズによる誤録画を防ぐため、開始を遅らせます（通常 0〜1000ms）。
          </div>
          <input type="number" name="recorder_start_delay_ms" value="{{ config.get('recorder_start_delay_ms', 0) }}" min="0" max="5000" step="100">
        </div>

        <div class="form-group">
          <label>プリ録画（フレーム枚数）</label>
          <div style="font-size:0.7rem; color:var(--muted); margin-bottom:8px;">
            検知した瞬間の何フレーム前（過去）から録画を開始するか指定します（通常 20〜100枚）。
          </div>
          <input type="number" name="recorder_pre_frames" value="{{ config.get('recorder_pre_frames', 60) }}" min="0" max="300" step="10">
        </div>

        <div class="section-title">スナップショット設定</div>
        <div class="form-group">
          <label>保存解像度（横x縦）</label>
          <div style="display:flex; gap:10px; align-items:center;">
            <input type="number" name="snapshot_width" value="{{ config.get('snapshot_width', 1280) }}" min="320" max="1920" step="80" style="flex:1">
            <span>x</span>
            <input type="number" name="snapshot_height" value="{{ config.get('snapshot_height', 720) }}" min="240" max="1080" step="60" style="flex:1">
          </div>
        </div>

        <div class="form-group">
          <label>静止画保存モード</label>
          <select name="snapshot_mode" style="width:100%; padding:8px; background:var(--bg2); color:var(--text); border:1px solid var(--border); border-radius:4px;">
            <option value="start_only" {% if config.get('snapshot_mode') == 'start_only' %}selected{% endif %}>検知開始時のみ</option>
            <option value="both" {% if config.get('snapshot_mode') == 'both' %}selected{% endif %}>開始と終了の両方</option>
          </select>
        </div>

        <button type="button" class="btn primary" onclick="saveForm('form-recorder', 'msg-recorder')">保存</button>
        <div id="msg-recorder" class="success-msg">✅ 保存しました</div>
      </form>
    </div>

    <!-- Telegram タブ -->
    <div id="tab-telegram" class="tab-content">
      <form id="form-telegram">
        <div class="section-title">Telegram Bot 設定</div>
        <div class="form-group">
          <label>Bot Token</label>
          <input type="text" name="telegram_token" value="{{ config.telegram_token }}" placeholder="123456:ABCDEF...">
        </div>
        <div class="form-group">
          <label>Chat ID</label>
          <input type="text" name="telegram_chat_id" value="{{ config.telegram_chat_id }}" placeholder="-123456789">
        </div>
        <div class="form-group">
          <label>通知モード</label>
          <select name="telegram_notify_mode" style="width:100%; padding:8px; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:8px;">
            <option value="photo" {% if config.get('telegram_notify_mode', 'photo') == 'photo' %}selected{% endif %}>静止画のみ (デフォルト)</option>
            <option value="video" {% if config.get('telegram_notify_mode') == 'video' %}selected{% endif %}>動画のみ</option>
            <option value="both" {% if config.get('telegram_notify_mode') == 'both' %}selected{% endif %}>静止画と動画の両方</option>
            <option value="none" {% if config.get('telegram_notify_mode') == 'none' %}selected{% endif %}>通知なし</option>
          </select>
          <div style="font-size:0.7rem; color:var(--muted); margin-top:6px;">
            ※ 動画通知を選択した場合、録画終了後に送信されます。
          </div>
        </div>
        <div style="margin-top:20px; display:flex; gap:10px;">
          <button type="button" class="btn primary" onclick="saveForm('form-telegram','msg-telegram')">保存</button>
          <button type="button" class="btn" style="background:var(--accent2); color:white;" onclick="sendTestNotify()">通知テスト</button>
        </div>
        <div id="msg-telegram" class="success-msg">✅ 保存しました</div>
      </form>
    </div>

    <!-- モデル情報タブ -->
    <div id="tab-model" class="tab-content">
      <div class="section-title">モデル詳細情報</div>
      <div id="model-details-area">
        <p style="color:var(--muted);font-size:0.8rem;">読み込み中...</p>
      </div>
      <button type="button" class="btn" style="margin-top:20px;" onclick="fetchModelInfo()">情報を更新</button>
    </div>
    <!-- 認証タブ -->
    <div id="tab-auth" class="tab-content">
      <form id="form-auth">
        <div class="section-title">Web管理画面 ログイン設定</div>
        <div class="form-group">
          <label>ユーザー名</label>
          <input type="text" name="web_user" value="{{ config.get('web_user','admin') }}">
        </div>
        <div class="form-group">
          <label>パスワード</label>
          <input type="text" name="web_pass" value="{{ config.get('web_pass','admin') }}">
        </div>
        <button type="button" class="btn primary" onclick="saveForm('form-auth','msg-auth')">保存</button>
        <div id="msg-auth" class="success-msg">✅ 保存しました（次回ログインから有効）</div>
      </form>
    </div>

    <!-- 🧪 検証・テストタブ -->
    <div id="tab-test" class="tab-content">
      <div class="section-title">モデル管理 (永続保持)</div>
      <div class="form-group">
        <label>新しいモデルをアップロード (.tflite)</label>
        <div style="display:flex; gap:10px;">
          <input type="file" id="model-upload-input" accept=".tflite" style="font-size:0.8rem; flex:1;">
          <button class="btn" onclick="uploadModel()" style="padding:4px 12px; background:var(--accent2); color:white;">UP</button>
        </div>
      </div>
      <div id="model-list-area" style="margin-bottom:20px;">
        <!-- JSでモデル一覧を表示 -->
      </div>

      <div class="section-title">検知テスト (一時実行)</div>
      <div class="form-group">
        <label>テスト用ファイルを選択 (画像/動画)</label>
        <input type="file" id="test-media-input" accept="image/*,video/*">
      </div>
      <div class="form-group">
        <label>使用するモデル</label>
        <select id="test-model-select" style="width:100%; padding:8px; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:8px;">
          <option value="model.tflite">初期モデル (model.tflite)</option>
        </select>
      </div>
      <button class="btn primary" id="btn-run-test" onclick="runDetectionTest()">検知テストを実行</button>
      
      <div id="test-result-area" style="margin-top:20px; display:none;">
        <div class="section-title">テスト結果</div>
        <div id="test-status-msg" style="font-size:0.85rem; margin-bottom:10px; color:var(--accent2);"></div>
        <div id="test-preview-container" style="background:#000; border-radius:8px; overflow:hidden; min-height:100px; display:flex; align-items:center; justify-content:center;">
          <!-- プレビュー表示 -->
        </div>
        <div id="test-stats-report" style="margin-top:10px; font-size:0.75rem; color:var(--muted); font-family:monospace; background:var(--bg); padding:10px; border-radius:6px; white-space:pre-wrap;"></div>
      </div>
    </div>
  </div>

  <script>
    // 時計
    function tick() { document.getElementById('clock').textContent = new Date().toLocaleString('ja-JP'); }
    setInterval(tick, 1000); tick();

    // 設定パネル
    function toggleSettings() {
      const panel = document.getElementById('settings-panel');
      const overlay = document.getElementById('settings-overlay');
      const btn = document.getElementById('btn-settings');
      const isOpen = panel.classList.contains('open');
      if (isOpen) { closeSettings(); }
      else {
        panel.classList.add('open');
        overlay.classList.add('open');
        btn.classList.add('active');
        btn.textContent = '✕ 閉じる';
      }
    }
    function closeSettings() {
      document.getElementById('settings-panel').classList.remove('open');
      document.getElementById('settings-overlay').classList.remove('open');
      const btn = document.getElementById('btn-settings');
      btn.classList.remove('active');
      btn.textContent = '⚙ 設定';
    }

    // タブ切り替え
    function switchTab(id) {
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      
      const tab = document.getElementById('tab-' + id);
      const nav = document.getElementById('nav-' + id);
      if (tab) tab.classList.add('active');
      if (nav) nav.classList.add('active');
      
      if (id === 'model') fetchModelInfo();
      if (id === 'classes') fetchClassesInfo();
      if (id === 'test') { updateModelList(); }
    }

    // --- モデル管理機能 ---
    async function updateModelList() {
      const area = document.getElementById('model-list-area');
      const select = document.getElementById('test-model-select');
      try {
        const models = await fetch('/api/test/models').then(r => r.json());
        let html = '<table class="model-info-table"><tr><th>ファイル名</th><th>操作</th></tr>';
        
        // selectメニューのリセット
        select.innerHTML = '<option value="model.tflite">初期モデル (model.tflite)</option>';
        
        models.forEach(m => {
          html += `<tr>
            <td style="font-size:0.8rem;">${m}</td>
            <td><button class="btn" style="padding:2px 8px; font-size:0.7rem; background:#e25555; color:white;" onclick="deleteModel('${m}')">削除</button></td>
          </tr>`;
          if(m !== 'model.tflite') {
            const opt = document.createElement('option');
            opt.value = 'Uploads/' + m;
            opt.textContent = m;
            select.appendChild(opt);
          }
        });
        html += '</table>';
        area.innerHTML = html;
      } catch(e) { area.innerHTML = "モデル一覧の取得に失敗"; }
    }

    async function uploadModel() {
      const input = document.getElementById('model-upload-input');
      if (!input.files[0]) return alert("ファイルを選択してください");
      const formData = new FormData();
      formData.append('file', input.files[0]);
      try {
        const res = await fetch('/api/test/model_upload', { method:'POST', body:formData }).then(r=>r.json());
        if(res.ok) { alert("アップロード完了"); updateModelList(); input.value = ''; }
        else { alert("エラー: " + res.error); }
      } catch(e) { alert("接続エラー"); }
    }

    async function deleteModel(name) {
      if(!confirm(`${name} を削除しますか？`)) return;
      try {
        const res = await fetch(`/api/test/model_delete?name=${name}`, { method:'DELETE' }).then(r=>r.json());
        if(res.ok) { updateModelList(); }
        else { alert("削除エラー"); }
      } catch(e) { alert("接続エラー"); }
    }

    // --- 検知テスト実行 ---
    async function runDetectionTest() {
      const mediaInput = document.getElementById('test-media-input');
      const modelSelect = document.getElementById('test-model-select');
      if (!mediaInput.files[0]) return alert("テスト用ファイルを選択してください");
      
      const btn = document.getElementById('btn-run-test');
      const resArea = document.getElementById('test-result-area');
      const statusMsg = document.getElementById('test-status-msg');
      const preview = document.getElementById('test-preview-container');
      const report = document.getElementById('test-stats-report');

      btn.disabled = true;
      btn.textContent = "処理中...";
      resArea.style.display = 'block';
      statusMsg.textContent = "ファイルをアップロードし、推論を実行しています...";
      preview.innerHTML = '<div class="stat-value blue" style="padding:40px;">⏳ Processing...</div>';
      report.textContent = "";

      const formData = new FormData();
      formData.append('file', mediaInput.files[0]);
      formData.append('model_path', modelSelect.value);

      try {
        const res = await fetch('/api/test/run', { method:'POST', body:formData }).then(r=>r.json());
        if(res.ok) {
          statusMsg.textContent = `完了: ${res.filename} (推論速度: ${res.avg_inf_ms}ms)`;
          const isVideo = res.result_url.match(/\.(mp4|avi)$/i);
          if (isVideo) {
            preview.innerHTML = `<video src="${res.result_url}" controls style="width:100%; max-height:400px;"></video>`;
          } else {
            preview.innerHTML = `<img src="${res.result_url}" style="width:100%; max-height:400px; object-fit:contain;">`;
          }
          report.textContent = "検知統計:\n" + JSON.stringify(res.stats, null, 2);
        } else {
          statusMsg.textContent = "エラー: " + res.error;
          preview.innerHTML = '<div class="red">Failed</div>';
        }
      } catch(e) {
        statusMsg.textContent = "接続エラーが発生しました";
      } finally {
        btn.disabled = false;
        btn.textContent = "検知テストを実行";
      }
    }

    async function sendTestNotify() {
        try {
            const res = await fetch('/api/notify_test', {method:'POST'}).then(r=>r.json());
            alert(res.message || (res.ok ? "テスト送信をリクエストしました" : "エラーが発生しました"));
        } catch(e) {
            alert("接続エラー");
        }
    }

    // ステータスポーリング
    async function pollStatus() {
      try {
        const d = await fetch('/api/status').then(r => r.json());
        document.getElementById('badge-fps').textContent = 'FPS: ' + d.fps;
        document.getElementById('badge-res').textContent = d.stream_width + 'x' + d.stream_height;
        document.getElementById('badge-count').textContent = '累計: ' + d.detections_total;
        document.getElementById('badge-last').textContent = '最終: ' + d.last_detected;
        document.getElementById('st-running').textContent = d.running ? '稼働中' : '停止中';
        document.getElementById('st-humans').textContent = d.human_count;
        document.getElementById('st-total').textContent = d.detections_total;
        document.getElementById('st-last').textContent = d.last_detected;
        document.getElementById('st-fps').textContent = d.fps;
        document.getElementById('st-res').textContent = d.stream_width + 'x' + d.stream_height;
        const alertBadge = document.getElementById('badge-alert');
        alertBadge.style.display = d.human_count > 0 ? 'inline-block' : 'none';
        alertBadge.className = d.human_count > 0 ? 'badge alert' : 'badge';
      } catch(e) {}
    }
    setInterval(pollStatus, 2000); pollStatus();

    // ログ表示対象の日付 (YYYY-MM-DD)
    let currentLogDate = new Date().toISOString().split('T')[0];

    function changeLogDate(offset) {
        const d = new Date(currentLogDate);
        if (offset === 0) {
            currentLogDate = new Date().toISOString().split('T')[0];
        } else {
            d.setDate(d.getDate() + offset);
            currentLogDate = d.toISOString().split('T')[0];
        }
        document.getElementById('log-date-display').textContent = '[' + currentLogDate + ']';
        pollLogs();
    }
    // 初期表示用
    document.getElementById('log-date-display').textContent = '[' + currentLogDate + ']';

    // ログポーリング
    async function pollLogs() {
      try {
        const rows = await fetch(`/api/logs?date=${currentLogDate}`).then(r => r.json());
        if (!rows.length) {
            document.getElementById('log-body').innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--muted);padding:14px">データなし</td></tr>';
            return;
        }
        document.getElementById('log-body').innerHTML = rows.map(r => {
          // パスからファイル名のみを抽出してリンクを作成（より堅牢に）
          const getFilename = (p) => p ? p.split(/[\\/]/).pop() : null;
          const snapFile = getFilename(r.snapshot_path);
          const videoFile = getFilename(r.video_path);

          const snapLink = snapFile ? `<a href="/records/${snapFile}" target="_blank" title="画像を表示">📷</a>` : '—';
          let videoLink = '—';
          if (videoFile) {
            const isAvi = videoFile.toLowerCase().endsWith('.avi');
            const label = isAvi ? '🎬(AVI)' : '🎬';
            const title = isAvi ? 'ダウンロードして再生' : '動画を再生';
            videoLink = `<a href="/records/${videoFile}" target="_blank" title="${title}">${label}</a>`;
          }
          return `<tr>
            <td>${r.timestamp}</td>
            <td>${r.human_count}</td>
            <td>${(parseFloat(r.confidence_max)*100).toFixed(0)}%</td>
            <td>${snapLink} ${videoLink}</td>
          </tr>`;
        }).join('');
      } catch(e) {}
    }
    setInterval(pollLogs, 5000); pollLogs();

    // 設定保存
    async function saveForm(formId, msgId) {
      const form = document.getElementById(formId);
      const data = {};
      form.querySelectorAll('input').forEach(i => {
        let val = i.value;
        if (i.name === 'detection_threshold') {
            val = 1.05 - parseFloat(i.value);
        }
        if (i.type === 'checkbox') {
            data[i.name] = i.checked;
        } else {
            data[i.name] = (i.type === 'number' || i.type === 'range') ? Number(val) : val;
        }
      });
      // フォーム内の select 要素も収集
      form.querySelectorAll('select').forEach(s => {
        data[s.name] = s.value;
      });
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      });
      if (res.ok) {
        const msg = document.getElementById(msgId);
        msg.style.display = 'block';
        setTimeout(() => msg.style.display = 'none', 2500);
      }
    }

    async function fetchClassesInfo() {
      const area = document.getElementById('classes-list-area');
      try {
        const model = await fetch('/api/model').then(r => r.json());
        const config = await fetch('/api/config').then(r => r.json());
        if (!model || !config) return;

        const globalClassesMap = model.classes || {};
        const globalTargetClasses = config.target_classes || [];

        let html = '';
        Object.keys(globalClassesMap).sort((a,b)=>Number(a)-Number(b)).forEach(id => {
          const name = globalClassesMap[id];
          const checked = globalTargetClasses.includes(Number(id)) ? 'checked' : '';
          html += `
            <tr style="border-bottom:1px solid var(--border);">
              <td style="padding:5px;"><input type="checkbox" class="cls-target" data-id="${id}" ${checked}></td>
              <td style="padding:5px; color:var(--muted)">${id}</td>
              <td style="padding:5px;"><input type="text" class="cls-name" data-id="${id}" value="${name}" 
                  style="padding:2px 5px; height:24px; font-size:0.75rem;"></td>
            </tr>
          `;
        });
        area.innerHTML = html;
      } catch(e) {
        area.innerHTML = `<tr><td colspan="3" class="red">接続エラー</td></tr>`;
      }
    }

    async function saveClasses() {
      const showAll = document.querySelector('#form-classes [name="show_all_detections"]').checked;
      const targets = [];
      document.querySelectorAll('.cls-target:checked').forEach(i => targets.push(Number(i.dataset.id)));
      
      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            "target_classes": targets,
            "show_all_detections": showAll
        })
      });

      const newClasses = {};
      document.querySelectorAll('.cls-name').forEach(i => {
        newClasses[i.dataset.id] = i.value;
      });

      const res = await fetch('/api/classes', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(newClasses)
      });

      if (res.ok) {
        const msg = document.getElementById('msg-classes');
        msg.style.display = 'block';
        setTimeout(() => msg.style.display = 'none', 2500);
      }
    }

    // モデル情報取得
    async function fetchModelInfo() {
      const area = document.getElementById('model-details-area');
      try {
        const d = await fetch('/api/model').then(r => r.json());
        if (!d) return;

        const renderTensors = (list) => {
          return `<ul class="tensor-list">` + (list || []).map(t => 
            `<li class="tensor-item"><b>${t.name}</b><br><span style="color:var(--accent2)">[${t.shape.join(', ')}]</span> <span style="color:var(--muted)">${t.dtype}</span></li>`
          ).join('') + `</ul>`;
        };

        area.innerHTML = `
          <table class="model-info-table">
            <tr><th>ステータス</th><td class="${d.status==='Loaded'?'green':'red'}">${d.status}</td></tr>
            <tr><th>モデルパス</th><td style="word-break:break-all;font-size:0.75rem">${d.path}</td></tr>
          </table>
          
          <div style="margin-top:15px; font-size:0.75rem; color:var(--muted); font-weight:700;">入力テンソル</div>
          ${renderTensors(d.input)}

          <div style="margin-top:15px; font-size:0.75rem; color:var(--muted); font-weight:700;">出力テンソル</div>
          ${renderTensors(d.output)}

          <div style="margin-top:15px; font-size:0.75rem; color:var(--muted); font-weight:700;">インデックス判別結果</div>
          <table class="model-info-table">
            <tr><th>Boxes</th><td>${d.indices.boxes}</td></tr>
            <tr><th>Classes</th><td>${d.indices.classes}</td></tr>
            <tr><th>Scores</th><td>${d.indices.scores}</td></tr>
            <tr><th>Count</th><td>${d.indices.count}</td></tr>
          </table>
        `;
      } catch(e) {
        area.innerHTML = `<p class="red">接続エラーが発生しました</p>`;
      }
    }
  </script>
</body>
</html>
"""

MEDIA_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>メディア閲覧 - 監視カメラ</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
      --accent: #4f8ef7; --text: #e2e8f0; --muted: #8892a4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); padding-bottom: 40px; }
    header {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 12px 24px; display: flex; align-items: center; gap: 12px; position: sticky; top: 0; z-index: 100;
    }
    header h1 { font-size: 1rem; font-weight: 600; }
    .btn-back {
      text-decoration: none; color: var(--text); background: var(--border);
      padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    }
    .container { padding: 24px; max-width: 1200px; margin: 0 auto; }
    .media-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px;
    }
    .media-card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      overflow: hidden; cursor: pointer; transition: transform 0.2s;
    }
    .media-card:hover { transform: translateY(-3px); border-color: var(--accent); }
    .media-thumb { width: 100%; height: 140px; background: #000; object-fit: cover; }
    .media-info { padding: 10px; }
    .media-name { font-size: 0.75rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .media-meta { font-size: 0.7rem; color: var(--muted); margin-top: 4px; display: flex; justify-content: space-between; }
    
    /* モーダル表示 */
    #viewer {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9);
      z-index: 1000; flex-direction: column; align-items: center; justify-content: center;
      padding: 20px;
    }
    #viewer.open { display: flex; }
    #viewer-content { max-width: 90%; max-height: 80%; border-radius: 8px; background: #000; }
    .viewer-close { position: absolute; top: 20px; right: 20px; color: #fff; font-size: 2rem; cursor: pointer; }
    .viewer-title { margin-top: 15px; font-size: 0.9rem; color: #fff; }
    .btn-download { margin-top: 10px; background: var(--accent); color: #fff; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; text-decoration: none; font-size: 0.8rem; }
  </style>
</head>
<body>
  <header>
    <a href="/" class="btn-back">◀ 戻る</a>
    <h1>📂 保存済みメディア閲覧</h1>
  </header>
  <div class="container">
    <div id="media-list" class="media-grid">
      <p style="color:var(--muted)">読み込み中...</p>
    </div>
  </div>

  <div id="viewer" onclick="closeViewer()">
    <span class="viewer-close">✕</span>
    <div id="viewer-main" onclick="event.stopPropagation()">
        <!-- 動画または画像がここに挿入される -->
    </div>
    <div class="viewer-title" id="viewer-title"></div>
    <a id="download-link" class="btn-download" href="#" download>ダウンロードして保存</a>
  </div>

  <script>
    async function loadMedia() {
      try {
        const files = await fetch('/api/media_list').then(r => r.json());
        const listArea = document.getElementById('media-list');
        if (!files.length) {
          listArea.innerHTML = '<p style="color:var(--muted)">保存されたファイルはありません。</p>';
          return;
        }
        listArea.innerHTML = files.map(f => {
          const isVideo = f.name.match(/\.(mp4|avi)$/i);
          const icon = isVideo ? '🎬' : '📷';
          const thumbSrc = isVideo ? '' : `/records/${f.name}`;
          const thumbHtml = isVideo 
            ? `<div class="media-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:2rem;">${icon}</div>`
            : `<img class="media-thumb" src="${thumbSrc}" loading="lazy">`;
          
          return `
            <div class="media-card" onclick="openViewer('${f.name}', ${!!isVideo})">
              ${thumbHtml}
              <div class="media-info">
                <div class="media-name">${f.name}</div>
                <div class="media-meta">
                  <span>${f.size}</span>
                  <span>${f.date}</span>
                </div>
              </div>
            </div>
          `;
        }).join('');
      } catch(e) {
        document.getElementById('media-list').innerHTML = '<p class="red">エラーが発生しました</p>';
      }
    }

    function openViewer(name, isVideo) {
      const viewer = document.getElementById('viewer');
      const main = document.getElementById('viewer-main');
      const title = document.getElementById('viewer-title');
      const dl = document.getElementById('download-link');
      
      const fileUrl = `/records/${name}`;
      title.textContent = name;
      dl.href = fileUrl;
      
      if (isVideo) {
        main.innerHTML = `<video id="viewer-content" src="${fileUrl}" controls autoplay></video>`;
      } else {
        main.innerHTML = `<img id="viewer-content" src="${fileUrl}">`;
      }
      viewer.classList.add('open');
    }

    function closeViewer() {
      const viewer = document.getElementById('viewer');
      const main = document.getElementById('viewer-main');
      main.innerHTML = '';
      viewer.classList.remove('open');
    }

    loadMedia();
  </script>
</body>
</html>
"""

# ============================================================
# Basic 認証ヘルパー
# ============================================================
def check_auth(username, password):
    config = load_config()
    return (username == config.get('web_user', 'admin') and
            password == config.get('web_pass', 'admin'))

def authenticate():
    return Response(
        '認証が必要です。', 401,
        {'WWW-Authenticate': 'Basic realm="Monitoring Camera"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# ============================================================
# 設定ロード/保存ヘルパー
# ============================================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(new_values: dict):
    config = load_config()
    config.update(new_values)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    return config

# ============================================================
# ルート
# ============================================================
@app.route('/')
@requires_auth
def index():
    config = load_config()
    return render_template_string(TEMPLATE, config=config)

@app.route('/api/status')
@requires_auth
def api_status():
    return jsonify(system_status)

@app.route('/api/logs')
@requires_auth
def api_logs():
    date_str = request.args.get('date')
    if not date_str:
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    rows = []
    if logger_instance:
        rows = logger_instance.read_by_date(date_str)
    return jsonify(rows)

@app.route('/api/notify_test', methods=['POST'])
@requires_auth
def api_notify_test():
    if not notifier_instance:
        return jsonify({"ok": False, "message": "Notifier not initialized"})
    
    config = load_config()
    # 最新の設定で送り直すためにインスタンスを一時的に更新（または config から直接送るNotifier側の機能が必要だが、今はインスタンスの値を更新する）
    notifier_instance.token = config.get('telegram_token')
    notifier_instance.chat_id = config.get('telegram_chat_id')
    notifier_instance.api_url = f"https://api.telegram.org/bot{notifier_instance.token}/"
    
    notifier_instance.send_message("🔔 これは監視カメラシステムからのテスト通知です。")
    return jsonify({"ok": True, "message": "テスト通知を送信しました。コンソールログを確認してください。"})

@app.route('/api/model')
@requires_auth
def api_model():
    if detector_instance:
        return jsonify(detector_instance.get_model_info())
    return jsonify({"status": "error", "message": "Detector not found"})

@app.route('/api/classes', methods=['POST'])
@requires_auth
def api_classes():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"})
    
    # 外部ファイルに保存
    json_path = os.path.join(os.path.dirname(__file__), 'coco_classes.json')
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Detector 側のキャッシュも更新
        if detector_instance:
            detector_instance.refresh_classes()
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/config', methods=['GET', 'POST'])
@requires_auth
def api_config():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        
        allowed_keys = {
            'detection_threshold', 'notify_interval',
            'telegram_token', 'telegram_chat_id',
            'stream_width', 'stream_height',
            'web_user', 'web_pass',
            'target_classes', 'show_all_detections',
            'recorder_post_seconds', 'recorder_start_delay_ms',
            'recorder_width', 'recorder_height', 'recorder_pre_frames',
            'snapshot_width', 'snapshot_height', 'snapshot_mode'
        }
        filtered = {k: v for k, v in data.items() if k in allowed_keys}
        save_config(filtered)
        
        if 'stream_width' in filtered:
            system_status['stream_width'] = filtered['stream_width']
        if 'stream_height' in filtered:
            system_status['stream_height'] = filtered['stream_height']
            
        return jsonify({"ok": True})
    
    # GET の場合は現在の設定を返す
    return jsonify(load_config())

# ============================================================
# モデルテスト・管理用 API
# ============================================================

@app.route('/api/test/models', methods=['GET'])
@requires_auth
def api_test_models():
    """保存されているモデルの一覧を返す"""
    files = []
    # デフォルトモデル
    if os.path.exists('model.tflite'):
        files.append('model.tflite')
    
    # アップロードされたモデル
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if f.endswith('.tflite'):
                files.append(f)
    return jsonify(files)

@app.route('/api/test/model_upload', methods=['POST'])
@requires_auth
def api_test_model_upload():
    """モデルファイルをアップロードする"""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No filename"}), 400
    if not file.filename.endswith('.tflite'):
        return jsonify({"ok": False, "error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return jsonify({"ok": True})

@app.route('/api/test/model_delete', methods=['DELETE'])
@requires_auth
def api_test_model_delete():
    """モデルファイルを削除する"""
    name = request.args.get('name')
    if not name: return jsonify({"ok": False}), 400
    
    # 安全のためパスを制限
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(name))
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.route('/api/test/run', methods=['POST'])
@requires_auth
def api_test_run():
    """アップロードされたメディアに対して検知テストを実行する"""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400
    
    file = request.files['file']
    model_path = request.form.get('model_path', 'model.tflite')
    
    # セキュアなパス処理
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    input_filename = f"test_{timestamp}_{filename}"
    input_path = os.path.join(app.config['TMP_TEST_FOLDER'], input_filename)
    file.save(input_path)
    
    # 出力ファイル名
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_result{ext}"
    output_path = os.path.join(app.config['TMP_TEST_FOLDER'], output_filename)
    
    # 検知器の初期化（指定されたモデルを使用）
    try:
        # モデルパスの検証（Uploads/ または 直下）
        safe_model_path = model_path
        if model_path.startswith('Uploads/'):
            safe_model_path = os.path.join(os.getcwd(), 'Uploads', secure_filename(os.path.basename(model_path)))
        
        test_detector = HumanDetector(model_path=safe_model_path, threshold=0.4)
        
        is_video = ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']
        stats = {}
        avg_inf_ms = 0
        
        if is_video:
            # 動画処理 (Tools/model_test.py のロジックを流用)
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            inf_times = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                t1 = time.time()
                detections = test_detector.detect(frame)
                inf_times.append(time.time() - t1)
                
                for d in detections:
                    class_name = test_detector.classes.get(d[5], f"ID:{d[5]}")
                    stats[class_name] = stats.get(class_name, 0) + 1
                
                res_frame = test_detector.draw_detections(frame, detections)
                out.write(res_frame)
            
            cap.release()
            out.release()
            avg_inf_ms = round((sum(inf_times) / max(1, len(inf_times))) * 1000, 1)
        else:
            # 静止画処理
            frame = cv2.imread(input_path)
            if frame is not None:
                t1 = time.time()
                detections = test_detector.detect(frame)
                avg_inf_ms = round((time.time() - t1) * 1000, 1)
                
                for d in detections:
                    class_name = test_detector.classes.get(d[5], f"ID:{d[5]}")
                    stats[class_name] = stats.get(class_name, 0) + 1
                    
                res_frame = test_detector.draw_detections(frame, detections)
                cv2.imwrite(output_path, res_frame)
        
        # 結果のURL (一時ファイルの配信)
        result_url = f"/test_files/{output_filename}"
        
        # 一時ファイルのクリーンアップ（スレッドで遅延実行）
        def cleanup_task():
            time.sleep(600) # 10分後に削除
            try:
                if os.path.exists(input_path): os.remove(input_path)
                if os.path.exists(output_path): os.remove(output_path)
            except: pass
        threading.Thread(target=cleanup_task).start()
        
        return jsonify({
            "ok": True,
            "filename": filename,
            "result_url": result_url,
            "avg_inf_ms": avg_inf_ms,
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/test_files/<path:filename>')
@requires_auth
def serve_test_file(filename):
    """一時テストファイルの配信"""
    return send_from_directory(os.path.abspath(TMP_TEST_FOLDER), filename)

@app.route('/records/<path:filename>')
@requires_auth
def serve_record(filename):
    config = load_config()
    save_dir = config.get('save_directory', 'records')
    # 絶対パスを構築
    abs_save_dir = os.path.abspath(save_dir)
    return send_from_directory(abs_save_dir, filename)

def _draw_osd(frame):
    """フレームに検知状態・FPS・日時を重畳する。"""
    import datetime
    h, w = frame.shape[:2]
    human_count = system_status.get('human_count', 0)
    fps          = system_status.get('fps', 0)
    now_str      = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 上部バー（半透明）
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # FPS と日時
    cv2.putText(frame, f"FPS: {fps}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 1)
    cv2.putText(frame, now_str, (w - 8 - cv2.getTextSize(now_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0], 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 255), 1)

    # 下部ステータスバー
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 38), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0)

    if human_count > 0:
        status_text  = f"DETECTED: {human_count}"
        status_color = (50, 80, 255)   # 赤
        # 検知時は枠で警告強調
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (50, 80, 255), 3)
    else:
        status_text  = "Monitoring..."
        status_color = (80, 220, 80)   # 緑

    cv2.putText(frame, status_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    last = system_status.get('last_detected', '—')
    last_str = f"Last: {last}"
    cv2.putText(frame, last_str,
                (w - 8 - cv2.getTextSize(last_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0], h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 255), 1)

    return frame

def generate_frames():
    import time
    prev_time = time.time()
    while True:
        # main.py で加工されたフレームがあればそれを優先
        frame = latest_processed_frame
        
        # なければカメラから直接取得（フォールバック）
        if frame is None and camera_instance:
            frame = camera_instance.get_frame()

        if frame is not None:
            now = time.time()
            system_status['fps'] = round(1.0 / max(now - prev_time, 1e-6), 1)
            prev_time = now

            w = system_status.get('stream_width', 640)
            h = system_status.get('stream_height', 480)
            display = cv2.resize(frame, (w, h))

            # OSD 描画
            display = _draw_osd(display)

            ret, buffer = cv2.imencode('.jpg', display)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.01) # 少し待機してループ

@app.route('/video_feed')
@requires_auth
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/media')
@requires_auth
def media_browser():
    return render_template_string(MEDIA_TEMPLATE)

@app.route('/api/media_list')
@requires_auth
def api_media_list():
    config = load_config()
    save_dir = config.get('save_directory', 'records')
    if not os.path.exists(save_dir):
        return jsonify([])
    
    files = []
    for filename in os.listdir(save_dir):
        if filename.lower().endswith(('.jpg', '.mp4', '.avi')):
            path = os.path.join(save_dir, filename)
            stat = os.stat(path)
            files.append({
                "name": filename,
                "size": f"{stat.st_size / (1024*1024):.1f} MB" if stat.st_size > 1024*1024 else f"{stat.st_size / 1024:.0f} KB",
                "mtime": stat.st_mtime,
                "date": datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    # 日付の降順でソート
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify(files)

def run_server(cam, logger=None, detector=None, notifier=None):
    global camera_instance, logger_instance, detector_instance, notifier_instance
    camera_instance = cam
    logger_instance = logger
    detector_instance = detector
    notifier_instance = notifier
    config = load_config()
    system_status['stream_width'] = config.get('stream_width', 640)
    system_status['stream_height'] = config.get('stream_height', 480)
    app.run(host='0.0.0.0', port=5000, threaded=True)
