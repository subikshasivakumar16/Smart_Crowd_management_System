from flask import Flask, Response, request, jsonify
import cv2
from ultralytics import YOLO
import requests
import time
from flask_cors import CORS
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

print("🚀 SafeEye Server Starting...")

# LOAD MODEL
model = YOLO("yolov8n.pt")

# GLOBALS
camera_urls = []
cameras = []
counts = []
LIMIT = 3
history = []
bots = []
last_alert_time = []
alert_count = 0
ALERT_INTERVAL = 30   # ✅ EVERY 30 SECONDS


@app.route('/')
def home():
    return "✅ SafeEye Running"


# ✅ SET CAMERAS
@app.route('/set_cameras', methods=['POST'])
def set_cameras():
    global camera_urls, cameras, counts, LIMIT, bots, last_alert_time

    data = request.json
    camera_urls = data.get("ips", [])
    LIMIT = int(data.get("limit", 3))
    bots = data.get("bots", [])

    # release old cameras
    for cam in cameras:
        cam.release()

    cameras = []
    counts = []

    for i, url in enumerate(camera_urls):
        print(f"📷 Connecting Camera {i+1}: {url}")
        cam = cv2.VideoCapture(url)

        if not cam.isOpened():
            print(f"❌ Camera {i+1} FAILED")
        else:
            print(f"✅ Camera {i+1} CONNECTED")
            cameras.append(cam)
            counts.append(0)

    last_alert_time = [0] * len(cameras)

    return jsonify({"status": "ok"})


# ✅ COUNTS API
@app.route('/counts')
def get_counts():
    total = sum(counts)

    if total < LIMIT * len(counts) * 0.5:
        density = "Low"
    elif total < LIMIT * len(counts):
        density = "Medium"
    else:
        density = "High"

    return jsonify({
        "counts": counts,
        "total": total,
        "density": density,
        "alerts": alert_count
    })

# ✅ SETTINGS
@app.route('/set_settings', methods=['POST'])
def set_settings():
    global LIMIT, bots

    data = request.json
    LIMIT = int(data.get("limit", LIMIT))

    bot_token = data.get("bot_token")
    chat_id = data.get("chat_id")

    if bot_token and chat_id:
        bots = [{"token": bot_token, "chat": chat_id}]

    return jsonify({"status": "updated"})
@app.route('/history')
def get_history():
    return jsonify(history)

def send_alert(frame, cam_index, count, level):
    import requests
    from datetime import datetime

    success, img = cv2.imencode('.jpg', frame)
    if not success:
        return

    # 🔥 TIME + DATE
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    # 🔥 MESSAGE FORMAT (WITH LIMIT)
    message = f"""🚨 Crowd Alert

📷 Camera: {cam_index+1}
👥 People Detected: {count}
📊 Crowd Level: {level}
🚧 Crowd Limit: {LIMIT}
🕒 Time: {time_str}
📅 Date: {date_str}
"""

    for bot in bots:
        try:
            url = f"https://api.telegram.org/bot{bot['token']}/sendPhoto"

            files = {
                "photo": ("alert.jpg", img.tobytes())
            }

            data = {
                "chat_id": bot['chat'],
                "caption": message
            }

            res = requests.post(url, data=data, files=files)
            print("📩 Telegram:", res.status_code)

        except Exception as e:
            print("❌ Telegram Error:", e)
# ✅ PROCESS FRAME (BOUNDING BOX + ALERT TIMER)
def process_frame(frame, index):
    global counts, last_alert_time, LIMIT, alert_count, history

    count = 0

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "Person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    counts[index] = count

    # LEVEL
    if count < LIMIT * 0.5:
        level = "Low"
    elif count < LIMIT:
        level = "Medium"
    else:
        level = "High"

    current_time = time.time()

    # ✅ FIXED BLOCK
    if current_time - last_alert_time[index] > ALERT_INTERVAL:

        if count >= LIMIT:   # ONLY WHEN LIMIT EXCEEDED

            alert_count += 1

            history.append({
                "time": time.strftime("%H:%M:%S"),
                "camera": f"Camera {index+1}",
                "people": count,
                "level": level
            })

            send_alert(frame, index, count, level)

        last_alert_time[index] = current_time

    return frame
# ✅ VIDEO STREAM
def generate_frames(index):
    while True:
        if index >= len(cameras):
            break

        cam = cameras[index]
        success, frame = cam.read()

        if not success:
            continue

        frame = process_frame(frame, index)

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/video/<int:index>')
def video(index):
    return Response(generate_frames(index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True) 
