from flask import Flask, Response, render_template_string, request, jsonify, redirect, url_for, send_from_directory
from functools import wraps
import cv2
import json
import os

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
    .tab {
      flex: 1; padding: 10px 0; text-align: center; font-size: 0.8rem; font-weight: 600;
      color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent;
      transition: all .2s;
    }
    .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
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
      <button class="nav-item active" onclick="switchTab('detect')">📹 検知</button>
      <button class="nav-item" onclick="switchTab('classes')">🍱 クラス</button>
      <button class="nav-item" onclick="switchTab('recorder')">🎬 録画・保存</button>
      <button class="nav-item" onclick="switchTab('telegram')">✈️ Telegram</button>
      <button class="nav-item" onclick="switchTab('auth')">🔐 認証</button>
      <button class="nav-item" onclick="switchTab('model')">🤖 モデル</button>
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
      document.getElementById('tab-' + id).classList.add('active');
      const idx_map = {'detect':0, 'classes':1, 'recorder':2, 'telegram':3, 'auth':4, 'model':5};
      document.querySelectorAll('.nav-item')[idx_map[id] || 0].classList.add('active');
      if (id === 'model') fetchModelInfo();
      if (id === 'classes') fetchClassesInfo();
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
        if (!rows.length) return;
        document.getElementById('log-body').innerHTML = rows.map(r => {
          const snapLink = r.snapshot_path ? `<a href="/records/${r.snapshot_path.split(/[\\\\/]/).pop()}" target="_blank" title="画像を表示">📷</a>` : '—';
          const videoLink = r.video_path ? `<a href="/records/${r.video_path.split(/[\\\\/]/).pop()}" target="_blank" title="動画を再生">🎬</a>` : '—';
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
            'recorder_post_seconds', 'snapshot_width', 'snapshot_height', 'snapshot_mode'
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
