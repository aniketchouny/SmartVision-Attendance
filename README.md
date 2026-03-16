# SmartVision-Attendance
This is a basic face recognition attendance system made using Python, OpenCV, and the face_recognition library. It uses a webcam to detect faces and mark attendance automatically in a database.
I made this as a simple project to understand how face recognition works in real-time and how it can be used for something practical like attendance.

---

## What it does

- Lets you register a person using your webcam  
- Stores their face images in a dataset  
- Converts images into face encodings  
- Recognizes faces in real time  
- Marks attendance automatically  
- Saves everything in a SQLite database  
- Can export attendance to a CSV file  

---

## Requirements

- Python 3  
- opencv-python  
- face_recognition  
- numpy  
- pandas  

Install them using:

pip install opencv-python face_recognition numpy pandas

Note: face_recognition needs dlib, which can be a bit tricky to install depending on your system.

---

## How to run

python main.py

You will see a menu like this:

1. Register new user  
2. Generate encodings  
3. Start recognition  
4. Export attendance  
5. Exit  

---

## How to use

### 1. Register a user

- Enter the person's name  
- Press SPACE to capture images  
- Press Q to stop  

Images will be saved in:

dataset/"name of person"/

---

### 2. Generate encodings

This processes the saved images and creates face encodings.  
It saves them in a file called:

encodings.pickle

---

### 3. Start recognition

- Opens webcam  
- Detects and recognizes faces  
- Marks attendance automatically  

Each person is only marked once per day.

---

### 4. Export attendance

This will create:

attendance_export.csv

---

## Files created

- dataset/ → stores face images  
- encodings.pickle → saved face data  
- attendance.db → database  
- attendance_export.csv → exported attendance  

---

## Notes

- Works better with good lighting  
- Try to keep only one face during registration  
- Webcam index may need to be changed if camera does not open  
- Accuracy depends on image quality  

---

## Why I made this

Just a small project to learn:
- face recognition  
- working with OpenCV  
- basic database handling  
