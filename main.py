import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model

# ===============================
# Load model & face detector
# ===============================

print("🔄 Loading emotion model...")
emotion_model = load_model("emotion_model.h5")

print("🔄 Loading face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Error loading Haar Cascade file")
    exit()

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

# ===============================
# Emotion prediction function
# ===============================
def predict_emotion(face_gray):
    face_gray = cv2.equalizeHist(face_gray)
    face_gray = cv2.resize(face_gray, (64, 64))
    face_gray = face_gray / 255.0
    face_gray = np.reshape(face_gray, (1, 64, 64, 1))

    preds = emotion_model.predict(face_gray, verbose=0)[0]
    idx = np.argmax(preds)

    return emotion_labels[idx], int(preds[idx] * 100)


# ===============================
# Get Largest Face Only
# ===============================
def get_largest_face(faces):
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    return faces[0]


# ===============================
# IMAGE MODE
# ===============================
def image_mode():
    image_path = input("Enter image path: ").strip()
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return

    # Resize large images
    h, w = img.shape[:2]
    if w > 1000:
        scale = 1000 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    face = get_largest_face(faces)

    if face is None:
        print("❌ No face detected")
        return

    (x, y, w, h) = face
    face_gray = gray[y:y+h, x:x+w]

    emotion, confidence = predict_emotion(face_gray)
    emoji = emotion_emojis[emotion]

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        img,
        f"{emotion} {emoji} ({confidence}%)",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Image Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===============================
# WEBCAM MODE
# ===============================
def webcam_mode():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    emotion_window = deque(maxlen=15)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        face = get_largest_face(faces)

        if face is not None:
            (x, y, w, h) = face
            face_gray = gray[y:y+h, x:x+w]

            emotion, confidence = predict_emotion(face_gray)
            emotion_window.append(emotion)

            # Smooth emotion output
            stable_emotion = max(set(emotion_window), key=emotion_window.count)
            emoji = emotion_emojis[stable_emotion]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{stable_emotion} {emoji} ({confidence}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # FPS Calculation
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Webcam Emotion Detection (Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# MAIN MENU
# ===============================
print("\nChoose mode:")
print("1 - Image emotion detection")
print("2 - Live webcam emotion detection")

choice = input("Enter choice (1/2): ").strip()

if choice == "1":
    image_mode()
elif choice == "2":
    webcam_mode()
else:
    print("❌ Invalid choice")