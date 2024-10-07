from flask import Flask, request, jsonify, render_template
import os
import cv2
import dlib
import numpy as np
from datetime import datetime
from PIL import Image

app = Flask(__name__)

# Initialize dlib's face detector and models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory containing images of known individuals
known_faces_dir = "known_faces"
attendance_file = "attendance.csv"

# Load known faces and their encodings
known_faces = []
known_names = []
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(known_faces_dir, filename))
        faces = detector(img)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(img, face)
            encoding = np.array(face_rec_model.compute_face_descriptor(img, landmarks))
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])


def encode_face(image):
    img = np.array(image)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(rgb_frame, face)
    encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, landmarks))
    return encoding


def mark_attendance(name):
    with open(attendance_file, "a") as file:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        file.write(f"{name},{dt_string}\n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'image' not in request.files:
        return jsonify({'status': 'No image sent'})

    image_file = request.files['image']
    image = Image.open(image_file.stream)

    encoding = encode_face(image)
    if encoding is None:
        return jsonify({'status': 'No face detected'})

    matches = []
    for known_face in known_faces:
        distance = np.linalg.norm(known_face - encoding)
        matches.append(distance)

    if matches:
        min_distance_index = np.argmin(matches)
        if matches[min_distance_index] < 0.6:
            name = known_names[min_distance_index]
            mark_attendance(name)
            response = {'status': f'Attendance marked for {name}'}
        else:
            response = {'status': 'Unknown'}
    else:
        response = {'status': 'Unknown'}

    return jsonify(response)

# Add this new route to your existing Flask app
@app.route('/get_all_students')
def get_all_students():
    return jsonify(known_names)
@app.route('/attendance')
def attendance():
    with open(attendance_file, 'r') as file:
        records = file.readlines()
    return render_template('attendance.html', records=records)


if __name__ == '__main__':
    app.run(debug=True)
