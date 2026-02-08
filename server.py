import os
import csv
import datetime
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from deepface import DeepFace

app = FastAPI()

DB_PATH = "./database/"
MODEL_PATH = "liveness.model"
CSV_FILE = "attendance.csv"

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Liveness model not found at: {MODEL_PATH}")

print("Loading liveness model...")
liveness_model = tf.keras.models.load_model(MODEL_PATH)
print(" Liveness model loaded")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["NAME", "DATE", "CHECK_IN", "CHECK_OUT"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(" Could not open camera")

last_detected = {"name": None, "is_live": False}

def mark_checkin(name):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    with open(CSV_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))

    for i in range(1, len(rows)):
        if rows[i][0] == name and rows[i][1] == today:
            return "⚠️ Already checked in today"

    rows.append([name, today, now_time, ""])

    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    return f" Check-IN marked at {now_time}"

def mark_checkout(name):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    with open(CSV_FILE, "r", newline="") as f:
        rows = list(csv.reader(f))

    for i in range(1, len(rows)):
        if rows[i][0] == name and rows[i][1] == today:
            if rows[i][3] == "":
                rows[i][3] = now_time
                with open(CSV_FILE, "w", newline="") as f:
                    csv.writer(f).writerows(rows)
                return f" Check-OUT marked at {now_time}"
            else:
                return "⚠️ Already checked out today"

    return " You have not checked in today"

def gen_frames():
    global last_detected

    while True:
        success, frame = cap.read()
        if not success:
            continue

        name = None
        is_live = False

        try:
            res = DeepFace.find(frame, DB_PATH, enforce_detection=False, model_name="Facenet512")
            if isinstance(res, list) and len(res) > 0 and "identity" in res[0] and len(res[0]["identity"]) > 0:
                identity_path = res[0]["identity"][0]
                name = os.path.basename(os.path.dirname(identity_path))

                xmin = int(res[0]["source_x"][0])
                ymin = int(res[0]["source_y"][0])
                w = int(res[0]["source_w"][0])
                h = int(res[0]["source_h"][0])

                face = frame[ymin:ymin+h, xmin:xmin+w]

                if face.size > 0:
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float32") / 255.0
                    face = np.expand_dims(face, axis=0)

                    pred = liveness_model.predict(face, verbose=0)
                    is_live = np.argmax(pred[0]) == 1

                    last_detected["name"] = name
                    last_detected["is_live"] = is_live

                    color = (0, 255, 0) if is_live else (0, 0, 255)
                    label = f"{name} ({'LIVE' if is_live else 'FAKE'})"

                    cv2.rectangle(frame, (xmin, ymin), (xmin+w, ymin+h), color, 2)
                    cv2.putText(frame, label, (xmin, ymin-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("DeepFace error:", e)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Face Attendance</title>
      <style>
        body { font-family: Arial; text-align: center; background:#f5f5f5; }
        h1 { font-size: 36px; }
        img { border: 3px solid #333; border-radius: 10px; margin-top: 20px; }
        button { font-size: 22px; padding: 15px 30px; margin: 20px; cursor: pointer; }
        #status { font-size: 20px; margin-top: 20px; color: green; }
      </style>
    </head>
    <body>
      <h1> chai street </h1>

      <img src="/video_feed" width="640" height="480"/>

      <div>
        <button onclick="checkin()"> CHECK IN</button>
        <button onclick="checkout()"> CHECK OUT</button>
      </div>

      <div id="status"></div>

      <script>
        async function checkin() {
          const res = await fetch('/checkin', { method: 'POST' });
          const data = await res.json();
          document.getElementById('status').innerText = data.message;
        }

        async function checkout() {
          const res = await fetch('/checkout', { method: 'POST' });
          const data = await res.json();
          document.getElementById('status').innerText = data.message;
        }
      </script>
    </body>
    </html>
    """

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/checkin")
def checkin():
    if last_detected["name"] and last_detected["is_live"]:
        msg = mark_checkin(last_detected["name"])
        return {"message": msg}
    return {"message": " No live face detected"}

@app.post("/checkout")
def checkout():
    if last_detected["name"] and last_detected["is_live"]:
        msg = mark_checkout(last_detected["name"])
        return {"message": msg}
    return {"message": " No live face detected"}
