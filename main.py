
import face_recognition
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os

# Create/connect to SQLite database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        time TEXT NOT NULL,
        date TEXT NOT NULL
    )
""")
conn.commit()

# Load known faces
known_face_encodings = []
known_face_names = []

images_path = "images"
for filename in os.listdir(images_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img = face_recognition.load_image_file(os.path.join(images_path, filename))
        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Initialize webcam
video_capture = cv2.VideoCapture(0)

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, time, date))
    conn.commit()

print("[INFO] Starting FaceTend...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)

        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    cv2.imshow("FaceTend - Press 'q' to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
conn.close()
print("[INFO] Attendance recording complete.")
