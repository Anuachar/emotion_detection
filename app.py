from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
emotion_model = load_model("emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

emotion_emojis = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_emotion(face_gray):
    face_gray = cv2.equalizeHist(face_gray)
    face_gray = cv2.resize(face_gray, (64, 64))
    face_gray = face_gray / 255.0
    face_gray = np.reshape(face_gray, (1, 64, 64, 1))

    preds = emotion_model.predict(face_gray, verbose=0)[0]
    idx = np.argmax(preds)

    return emotion_labels[idx], int(preds[idx] * 100)


# Splash Page
@app.route("/")
def splash():
    return render_template("splash.html")


# Home Page
@app.route("/home")
def home():
    return render_template("index.html")


# 🔥 AJAX Emotion Analyze Route
@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files.get("image")

    if not file or file.filename == "":
        return jsonify({"error": "No file selected"})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    if img is None:
        return jsonify({"error": "Invalid image"})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return jsonify({"error": "No Face Detected"})

    (x, y, w, h) = faces[0]
    face_gray = gray[y:y+h, x:x+w]

    emotion, confidence = predict_emotion(face_gray)
    emoji = emotion_emojis[emotion]

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "emoji": emoji,
        "image": "uploads/" + file.filename
    })


# Webcam Stream
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            emotion, confidence = predict_emotion(face_gray)
            emoji = emotion_emojis[emotion]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} {emoji}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


if __name__ == "__main__":
    app.run(debug=True)