Face Detection Attendance System (with Liveness Check)
A smart AI-powered face recognition attendance system built using Python, FastAPI, OpenCV, DeepFace, and TensorFlow.
It detects faces in real-time from a camera, verifies identity from a local database, checks liveness, and automatically marks attendance.
This system helps prevent proxy attendance by ensuring that only a live person (not a photo or video) can mark attendance.

Features
✅ Real-time face detection using OpenCV
✅ Face recognition using DeepFace (Facenet512)
✅ Liveness detection using a trained TensorFlow model
✅ Automatic attendance logging to CSV
✅ FastAPI backend for streaming video
✅ Folder-based face database
✅ Prevents fake attendance using photos/videos

Tech Stack
Python 3.8+
FastAPI
OpenCV
DeepFace
TensorFlow / Keras
NumPy
Uvicorn

Project Structure
face_detection_model/
│
├── server.py           # Main FastAPI server
├── liveness.model      # Liveness detection model (not included in repo)
├── attendance.csv      # Attendance log file (ignored in git)
├── database/           # Face images database (ignored in git)
│   ├── Person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── Person2/
│       ├── img1.jpg
│       └── img2.jpg
└── .gitignore
⚠️ Note: database/, attendance.csv are ignored for privacy and size reasons.

Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/Rishabhpal07/face_detection_model.git
cd face_detection_model
2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On macOS/Linux
# venv\Scripts\activate    # On Windows
3️⃣ Install dependencies
pip install -r requirements.txt
(If you don’t have requirements.txt, install manually:)
pip install fastapi uvicorn opencv-python deepface tensorflow numpy
Prepare Face Database
Create a folder named database/ and inside it create folders for each person:
database/
├── Rishabh/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
└── shivam/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
Use clear face images
Add 5–10 images per person for better accuracy

▶️ Run the Server
uvicorn server:app --reload
Then open in browser:
http://127.0.0.1:8000
📝 Attendance Output
Attendance is saved in:
attendance.csv
Format:
NAME, DATE, CHECK_IN
Rishabh, 2026-02-09, 10:15:32
