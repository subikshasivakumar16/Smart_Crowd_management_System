"""
Smart Crowd and Environmental Monitoring Web Dashboard
Flask Backend - Main Application
"""

import os
import csv
import json
import base64
import threading
import time
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import numpy as np
import pandas as pd

# Configuration
app = Flask(__name__)
app.secret_key = 'smart-crowd-monitor-secret-key-2025'
app.config['SESSION_TYPE'] = 'filesystem'

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
VIDEOS_DIR = BASE_DIR / 'videos'
CSV_PATH = DATA_DIR / 'sensor_data.csv'
EVENTS_FILE = DATA_DIR / 'events.json'

# Thread-safe data storage
events_lock = threading.Lock()
last_event_log = {'crowd': {}, 'risk': {}}
data_lock = threading.Lock()
sensor_history = []
MAX_HISTORY = 200
zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
current_sensor_data = {
    'A': {'temperature': 25, 'humidity': 65, 'oxygen': 21, 'sound': 45},
    'B': {'temperature': 25, 'humidity': 66, 'oxygen': 21, 'sound': 44},
    'C': {'temperature': 26, 'humidity': 64, 'oxygen': 21, 'sound': 46},
    'D': {'temperature': 25, 'humidity': 65, 'oxygen': 21, 'sound': 45}
}

# LSTM model and scaler references (loaded lazily)
lstm_model = None
scaler = None
yolo_model = None


def load_or_create_models():
    """Load LSTM model and MinMaxScaler, or create fallback if not present."""
    global lstm_model, scaler
    
    lstm_path = MODELS_DIR / 'lstm_model.h5'
    scaler_path = MODELS_DIR / 'scaler.save'
    
    try:
        if scaler_path.exists():
            import joblib
            scaler = joblib.load(scaler_path)
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            sample_data = np.array([[25, 65, 21, 45], [30, 80, 20, 70], [20, 50, 21.5, 30]])
            scaler.fit(sample_data)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(scaler, scaler_path)
    except Exception as e:
        print(f"Scaler load/create error: {e}")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array([[25, 65, 21, 45], [30, 80, 20, 70]]))
    
    try:
        lstm_path_keras = MODELS_DIR / 'lstm_model.keras'
        lstm_path_h5 = MODELS_DIR / 'lstm_model.h5'
        if lstm_path_keras.exists():
            import tensorflow as tf
            lstm_model = tf.keras.models.load_model(lstm_path_keras)
        elif lstm_path_h5.exists():
            import tensorflow as tf
            lstm_model = tf.keras.models.load_model(lstm_path_h5)
        else:
            lstm_model = None
    except Exception as e:
        print(f"LSTM load error: {e}")
        lstm_model = None


def predict_risk(temperature, humidity, oxygen, sound):
    """Predict risk score using LSTM or fallback heuristic."""
    global lstm_model, scaler
    
    if lstm_model is None or scaler is None:
        # Heuristic fallback: higher temp, humidity, sound = higher risk
        norm_t = (temperature - 20) / 15
        norm_h = (humidity - 40) / 50
        norm_o = 1 - (oxygen - 19) / 3
        norm_s = (sound - 30) / 80
        score = 0.25 * norm_t + 0.25 * norm_h + 0.3 * norm_o + 0.2 * min(norm_s, 1)
        return float(np.clip(score, 0, 1))
    
    try:
        raw = np.array([[temperature, humidity, oxygen, sound]], dtype=np.float32)
        scaled = scaler.transform(raw)
        scaled = scaled.reshape(1, 1, 4)
        pred = lstm_model.predict(scaled, verbose=0)
        return float(np.clip(pred[0][0], 0, 1))
    except Exception:
        return 0.25


def load_events():
    """Load events from JSON file."""
    if not EVENTS_FILE.exists():
        return []
    try:
        with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_events(events):
    """Save events to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(EVENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(events[-500:], f, indent=2)
    except Exception:
        pass


def log_event(event_type, message, zone=None, value=None):
    """Log an event for calendar view. Throttled to avoid spam."""
    from datetime import datetime
    key = f"{zone or 'all'}"
    last = last_event_log.get(event_type, {}).get(key, 0)
    if time.time() - last < 60:
        return
    last_event_log.setdefault(event_type, {})[key] = time.time()
    with events_lock:
        events = load_events()
        events.append({
            'id': len(events) + 1,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'datetime': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'zone': zone,
            'value': value
        })
        save_events(events)


def load_sensor_csv():
    """Load sensor data from CSV file."""
    global current_sensor_data
    if not CSV_PATH.exists():
        return
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return
        zones = df['zone'].unique() if 'zone' in df.columns else ['A', 'B', 'C', 'D']
        for z in ['A', 'B', 'C', 'D']:
            zone_df = df[df['zone'] == z] if 'zone' in df.columns else df
            if len(zone_df) > 0:
                row = zone_df.iloc[-1]
                current_sensor_data[z] = {
                    'temperature': float(row.get('temperature', 25)),
                    'humidity': float(row.get('humidity', 65)),
                    'oxygen': float(row.get('oxygen', 21)),
                    'sound': float(row.get('sound', 45))
                }
    except Exception as e:
        print(f"CSV load error: {e}")


def sensor_update_thread():
    """Background thread to update sensor data every 2 seconds."""
    global sensor_history, current_sensor_data
    csv_rows = []
    
    if CSV_PATH.exists():
        try:
            with open(CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
        except Exception:
            pass
    
    idx = 0
    while True:
        try:
            with data_lock:
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                if csv_rows:
                    for z in ['A', 'B', 'C', 'D']:
                        zone_rows = [r for r in csv_rows if r.get('zone') == z]
                        if zone_rows:
                            row = zone_rows[idx % len(zone_rows)]
                        else:
                            row = csv_rows[idx % len(csv_rows)]
                        data = {
                            'temperature': float(row.get('temperature', 25)),
                            'humidity': float(row.get('humidity', 65)),
                            'oxygen': float(row.get('oxygen', 21)),
                            'sound': float(row.get('sound', 45))
                        }
                        current_sensor_data[z] = data
                        risk = predict_risk(data['temperature'], data['humidity'], data['oxygen'], data['sound'])
                        if risk > 0.6:
                            log_event('risk', f'High environmental risk in Zone {z}', zone=z, value=round(risk, 2))
                        sensor_history.append({
                            'timestamp': row.get('timestamp', ts),
                            'zone': z,
                            'temperature': data['temperature'],
                            'humidity': data['humidity'],
                            'oxygen': data['oxygen'],
                            'sound': data['sound'],
                            'risk': risk
                        })
                    idx += 1
                else:
                    for z in ['A', 'B', 'C', 'D']:
                        d = current_sensor_data[z]
                        d['temperature'] = round(d['temperature'] + np.random.uniform(-0.5, 0.5), 1)
                        d['humidity'] = round(d['humidity'] + np.random.uniform(-1, 1), 1)
                        d['sound'] = round(d['sound'] + np.random.uniform(-2, 2), 1)
                        d['oxygen'] = round(21 + np.random.uniform(-0.1, 0.1), 2)
                        risk = predict_risk(d['temperature'], d['humidity'], d['oxygen'], d['sound'])
                        sensor_history.append({
                            'timestamp': ts,
                            'zone': z,
                            'temperature': d['temperature'],
                            'humidity': d['humidity'],
                            'oxygen': d['oxygen'],
                            'sound': d['sound'],
                            'risk': risk
                        })
                if len(sensor_history) > MAX_HISTORY:
                    sensor_history = sensor_history[-MAX_HISTORY:]
        except Exception as e:
            print(f"Sensor update error: {e}")
        time.sleep(2)


def get_yolo_model():
    """Load YOLOv8 model lazily."""
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    try:
        from ultralytics import YOLO
        model_path = MODELS_DIR / 'yolov8s.pt'
        if model_path.exists():
            yolo_model = YOLO(str(model_path))
        else:
            yolo_model = YOLO('yolov8s.pt')
        return yolo_model
    except Exception as e:
        print(f"YOLO load error: {e}")
        return None


# Video capture cache per zone (thread-safe)
video_caps = {}
video_cap_lock = threading.Lock()


def get_video_path(zone):
    """Get video file path for zone. Uses zone_X.mp4 or first .mp4 in videos folder."""
    zone_file = VIDEOS_DIR / f'zone_{zone.lower()}.mp4'
    if zone_file.exists():
        return str(zone_file.resolve())
    mp4_files = list(VIDEOS_DIR.glob('*.mp4'))
    if mp4_files:
        return str(mp4_files[0].resolve())
    return None


def generate_video_frame(zone):
    """Generate MJPEG frame from videos folder only. YOLO draws boxes around people and counts them."""
    import cv2
    video_path = get_video_path(zone)
    
    if not video_path:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f'Zone {zone} - No video in folder', (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        with data_lock:
            zone_counts[zone] = 0
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    try:
        with video_cap_lock:
            if zone not in video_caps or not video_caps[zone].isOpened():
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    cap.release()
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f'Zone {zone} - Cannot open video', (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    with data_lock:
                        zone_counts[zone] = 0
                    _, jpeg = cv2.imencode('.jpg', frame)
                    return jpeg.tobytes()
                video_caps[zone] = cap
        
        cap = video_caps[zone]
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f'Zone {zone} - No frame', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            with data_lock:
                zone_counts[zone] = 0
        else:
            model = get_yolo_model()
            count = 0
            if model is not None:
                results = model(frame, classes=[0], verbose=False)
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:
                                count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, 'person', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            with data_lock:
                zone_counts[zone] = count
            if count >= 35:
                log_event('crowd', f'High crowd density in Zone {zone}: {count} people', zone=zone, value=count)
        
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    except Exception as e:
        print(f"Video frame error: {e}")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f'Zone {zone} - Error', (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        with data_lock:
            zone_counts[zone] = 0
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def video_stream(zone):
    """MJPEG stream generator for zone video."""
    import time
    while True:
        frame = generate_video_frame(zone)
        if not frame:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


# Auth decorator
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def login():
    """Login page."""
    if session.get('logged_in'):
        return redirect(url_for('crowd_dashboard'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def do_login():
    """Handle login form submission."""
    username = request.form.get('username', '')
    password = request.form.get('password', '')
    if username == 'admin' and password == '12345':
        session['logged_in'] = True
        return redirect(url_for('crowd_dashboard'))
    return render_template('login.html', error='Invalid username or password')


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for('login'))


@app.route('/index.html')
@login_required
def crowd_dashboard():
    """Crowd monitoring dashboard."""
    return render_template('index.html')


@app.route('/dashboard')
@login_required
def env_dashboard():
    """Environment monitoring dashboard."""
    return render_template('dashboard.html')


@app.route('/get_data')
@login_required
def get_data():
    """Return latest sensor data as JSON."""
    with data_lock:
        data = {z: dict(current_sensor_data[z]) for z in ['A', 'B', 'C', 'D']}
        avg_risk = 0.25
        if sensor_history:
            avg_risk = float(np.mean([h['risk'] for h in sensor_history[-4:]]))
        risk_level = 'Normal' if avg_risk < 0.3 else 'Medium' if avg_risk < 0.6 else 'Danger'
        return jsonify({
            'zones': data,
            'risk_score': avg_risk,
            'risk_level': risk_level
        })


@app.route('/get_history')
@login_required
def get_history():
    """Return sensor history (last 200 points) as JSON."""
    with data_lock:
        hist = list(sensor_history[-MAX_HISTORY:])
    return jsonify(hist)


@app.route('/counts')
@login_required
def counts():
    """Return zone crowd counts."""
    with data_lock:
        c = dict(zone_counts)
    return jsonify(c)


@app.route('/calendar/events')
@login_required
def calendar_events():
    """Return events for calendar by date. ?date=YYYY-MM-DD for specific date, else all."""
    date = request.args.get('date', '')
    with events_lock:
        events = load_events()
    if date:
        events = [e for e in events if e.get('date') == date]
    return jsonify(events[-100:])


@app.route('/calendar/dates')
@login_required
def calendar_dates_with_events():
    """Return list of dates that have events (for calendar highlighting)."""
    with events_lock:
        events = load_events()
    dates = list(set(e.get('date') for e in events if e.get('date')))
    return jsonify(sorted(dates))


@app.route('/download/video/<zone>')
@login_required
def download_video(zone):
    """Download source video file for zone."""
    zone = zone.upper()
    if zone not in ['A', 'B', 'C', 'D']:
        return '', 404
    video_path = get_video_path(zone)
    if not video_path or not Path(video_path).exists():
        return 'Video not found', 404
    from flask import send_file
    return send_file(video_path, as_attachment=True, download_name=f'zone_{zone.lower()}.mp4')


@app.route('/download/snapshot/<zone>')
@login_required
def download_snapshot(zone):
    """Download current frame as JPEG snapshot."""
    zone = zone.upper()
    if zone not in ['A', 'B', 'C', 'D']:
        return '', 404
    frame_bytes = generate_video_frame(zone)
    if not frame_bytes:
        return 'Error capturing frame', 500
    from io import BytesIO
    from flask import send_file
    return send_file(
        BytesIO(frame_bytes),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f'zone_{zone}_snapshot.jpg'
    )


@app.route('/download/report')
@login_required
def download_report():
    """Download sensor/crowd report as CSV."""
    date = request.args.get('date', '')
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Time', 'Zone', 'Temperature', 'Humidity', 'Oxygen', 'Sound', 'Risk', 'Crowd Count'])
    with data_lock:
        hist = list(sensor_history[-MAX_HISTORY:])
        counts = dict(zone_counts)
    for h in hist:
        row_date = h.get('timestamp', '')[:10] if h.get('timestamp') else ''
        if date and row_date != date:
            continue
        writer.writerow([
            row_date,
            h.get('timestamp', '')[11:19] if len(str(h.get('timestamp', ''))) > 10 else '',
            h.get('zone', ''),
            h.get('temperature', ''),
            h.get('humidity', ''),
            h.get('oxygen', ''),
            h.get('sound', ''),
            round(h.get('risk', 0), 2),
            counts.get(h.get('zone', ''), 0)
        ])
    from flask import send_file
    output.seek(0)
    return send_file(
        BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'monitor_report_{date or "all"}.csv'
    )


@app.route('/video/<zone>')
@login_required
def video_feed(zone):
    """MJPEG video stream for zone A/B/C/D."""
    zone = zone.upper()
    if zone not in ['A', 'B', 'C', 'D']:
        return '', 404
    return Response(
        video_stream(zone),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache'}
    )


if __name__ == '__main__':
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    load_sensor_csv()
    load_or_create_models()
    
    t = threading.Thread(target=sensor_update_thread, daemon=True)
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
