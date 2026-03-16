import cv2
import face_recognition
import numpy as np
import os
import pickle
import sqlite3
from datetime import datetime
import time
import pandas as pd

# Config

DATASET_DIR = "dataset"                # Stores captured face images per person
ENCODINGS_PATH = "encodings.pickle"    # Stores known face encodings
DB_PATH = "attendance.db"              # SQLite DB for users & attendance
CAPTURE_COUNT = 20                     # number of images to capture when registering
CAMERA_INDEX = 0                       # default webcam index


# Database init

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            reg_time TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            timestamp TEXT,
            date TEXT,
            UNIQUE(user_id, date),   -- ensures one entry per user per date
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def add_user_db(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    try:
        c.execute("INSERT INTO users (name, reg_time) VALUES (?, ?)", (name, now))
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        c.execute("SELECT id FROM users WHERE name = ?", (name,))
        row = c.fetchone()
        user_id = row[0] if row else None
    conn.commit()
    conn.close()
    return user_id

def mark_attendance_db(user_id, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute("INSERT OR IGNORE INTO attendance (user_id, name, timestamp, date) VALUES (?, ?, ?, ?)",
                  (user_id, name, ts, date_str))
        conn.commit()
    finally:
        conn.close()

def export_attendance_csv(out_path="attendance_export.csv"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    df.to_csv(out_path, index=False)
    conn.close()
    return out_path

# Dataset registration

def register_user(name, capture_count=CAPTURE_COUNT):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"[INFO] Starting registration for '{name}'. Press 'q' to abort early.")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    count = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    needed = capture_count
    print(f"[INFO] Already {count} images for user. Capturing {needed} new images...")

    captured = 0
    try:
        while captured < needed:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            display = frame.copy()
            h, w = display.shape[:2]
            cv2.putText(display, f"Capturing {captured+1}/{needed} - Press 'q' to cancel",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Register - Press space to capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                filename = os.path.join(person_dir, f"{int(time.time())}_{captured}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Saved {filename}")
                captured += 1
                time.sleep(0.2)
            elif key == ord('q'):
                print("[INFO] Registration aborted by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    user_id = add_user_db(name)
    print(f"[INFO] Registration complete. DB user_id = {user_id}")
    return user_id


# Encoding generation part

def generate_encodings(dataset_dir=DATASET_DIR, encodings_path=ENCODINGS_PATH):
    print("[INFO] Generating encodings from dataset...")
    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []:
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_folder, img_name)
            image = face_recognition.load_image_file(img_path)
            faces = face_recognition.face_encodings(image)
            if len(faces) == 0:
                print(f"[WARN] No face found in {img_path} - skipping")
                continue
            encoding = faces[0]
            known_encodings.append(encoding)
            known_names.append(person_name)
            print(f"[INFO] Encoded {img_path} for {person_name}")

    if len(known_encodings) == 0:
        print("[WARN] No encodings generated. Is your dataset empty or contains no faces?")
        return False

    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved encodings to {encodings_path} ({len(known_encodings)} faces)")
    return True

# Real-time recognition

def recognize_and_mark(encodings_path=ENCODINGS_PATH):
    if not os.path.exists(encodings_path):
        print("[ERROR] Encodings file not found. Run encoding generation first.")
        return

    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    known_encodings = data.get("encodings", [])
    known_names = data.get("names", [])
    print(f"[INFO] Loaded {len(known_encodings)} encodings for {len(set(known_names))} people.")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    name_to_id = {}
    c.execute("SELECT id, name FROM users")
    for uid, name in c.fetchall():
        name_to_id[name] = uid
    conn.close()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    process_every_n_frames = 2
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            if frame_count % process_every_n_frames == 0:
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                names_in_frame = []

                for face_encoding, face_loc in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        best_idx = np.argmin(face_distances)
                        name = known_names[best_idx]

                        user_id = name_to_id.get(name)
                        if user_id is None:
                            user_id = add_user_db(name)
                            name_to_id[name] = user_id
                            print(f"[INFO] Added user {name} to DB with id {user_id}")
                        mark_attendance_db(user_id, name)
                    names_in_frame.append((name, face_loc))

                for name, (top, right, bottom, left) in [(n, (t*2, r*2, b*2, l*2)) for n, (t,r,b,l) in names_in_frame]:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Attendance - Press 'q' to quit", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Recognition stopped.")

# Utility / Menu

def print_menu():
    print("\n==== Face Recognition Attendance System ====")
    print("1. Register new user (capture images)")
    print("2. Generate encodings from dataset")
    print("3. Start real-time recognition (mark attendance)")
    print("4. Export attendance to CSV")
    print("5. Exit")
    print("============================================")

def main_loop():
    init_db()
    while True:
        print_menu()
        choice = input("Enter choice (1-5): ").strip()
        if choice == "1":
            name = input("Enter the person's NAME (no slashes): ").strip()
            if name == "":
                print("[WARN] Name cannot be empty.")
                continue
            register_user(name)
        elif choice == "2":
            success = generate_encodings()
            if not success:
                print("[WARN] Encoding generation didn't produce any encodings.")
        elif choice == "3":
            print("[INFO] Starting recognition. Make sure encodings.pickle exists (option 2).")
            recognize_and_mark()
        elif choice == "4":
            out = export_attendance_csv()
            print(f"[INFO] Attendance exported to {out}")
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("[WARN] Invalid option. Choose 1-5.")

if __name__ == "__main__":
    main_loop()
