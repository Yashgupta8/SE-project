from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import dlib 
import numpy as np
from datetime import datetime
import json
import pandas as pd
from PIL import Image
from collections import defaultdict


from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all domains by default

CORS(app, origins=["http://localhost:5500"])
from flask_cors import cross_origin



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

# Dictionary to track when a student enters, for calculating exit times
daily_attendance = defaultdict(dict)

def get_date_string():
    return datetime.now().strftime('%Y-%m-%d')

def get_attendance_list():
    # Read the CSV file and process it into unique daily records
    daily_records = defaultdict(lambda: defaultdict(dict))
    
    try:
        with open(attendance_file, "r") as file:
            for line in file:
                name, entry_time, exit_time = line.strip().split(',')
                entry_datetime = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
                exit_datetime = datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S')
                date = entry_datetime.strftime('%Y-%m-%d')
                
                # If this person doesn't have a record for this date yet
                if name not in daily_records[date]:
                    daily_records[date][name] = {
                        "name": name,
                        "date": date,
                        "entry_time": entry_time,
                        "exit_time": exit_time
                    }
                else:
                    # Update entry time if this entry is earlier
                    current_entry = datetime.strptime(daily_records[date][name]["entry_time"], 
                                                    '%Y-%m-%d %H:%M:%S')
                    if entry_datetime < current_entry:
                        daily_records[date][name]["entry_time"] = entry_time
                    
                    # Update exit time if this exit is later
                    current_exit = datetime.strptime(daily_records[date][name]["exit_time"], 
                                                   '%Y-%m-%d %H:%M:%S')
                    if exit_datetime > current_exit:
                        daily_records[date][name]["exit_time"] = exit_time

    except FileNotFoundError:
        print(f"No attendance file found at {attendance_file}")
    except Exception as e:
        print(f"Error reading attendance file: {str(e)}")

    # Convert the nested defaultdict to a list of records
    attendance_list = []
    for date in daily_records:
        attendance_list.extend(list(daily_records[date].values()))
    
    # Sort by date and then by name
    attendance_list.sort(key=lambda x: (x["date"], x["name"]))
    return attendance_list

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    current_date = get_date_string()

    if name in daily_attendance[current_date]:
        # Update exit time for existing entry
        entry_time = daily_attendance[current_date][name]
        with open(attendance_file, "a") as file:
            file.write(f"{name},{entry_time},{dt_string}\n")
        # Clear the entry from daily_attendance after recording exit
        del daily_attendance[current_date][name]
    else:
        # Mark new entry time
        daily_attendance[current_date][name] = dt_string
    
    # Get and print the updated attendance list
    attendance_list = get_attendance_list()
    print("\nCurrent Attendance List:")
    print(json.dumps(attendance_list, indent=2))
    return attendance_list

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
            attendance_list = mark_attendance(name)
            response = {
                'status': f'Attendance marked for {name}',
                'attendance_list': attendance_list
            }
        else:
            response = {'status': 'Unknown'}
    else:
        response = {'status': 'Unknown'}

    return jsonify(response)

@app.route('/get_all_students')
def get_all_students():
    return jsonify(known_names)

@app.route('/attendance')
def attendance():
    date_filter = request.args.get('date', None)
    attendance_list = get_attendance_list()
    
    # Apply date filter if provided
    if date_filter:
        attendance_list = [record for record in attendance_list if record['date'] == date_filter]
    
    # Sort records by date and time
    sorted_records = sorted(
        attendance_list,
        key=lambda x: (x['date'], x['entry_time']),
        reverse=True  # Most recent first
    )
    
    # Get current date for statistics
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template(
        'attendance.html',
        records=sorted_records,
        current_date=current_date
    )

@app.route('/get_attendance_json')
@cross_origin()
def get_attendance_json():
    attendance_list = get_attendance_list()
    return jsonify(attendance_list)

@app.route('/get_daily_attendance')
def get_daily_attendance():
    date = request.args.get('date', get_date_string())
    attendance_list = get_attendance_list()
    daily_records = [record for record in attendance_list if record['date'] == date]
    return jsonify(daily_records)

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/download_attendance', methods=['GET'])
def download_attendance():
    # Get the attendance list
    attendance_list = get_attendance_list()

    # Save the attendance list to an Excel file
    output_path = 'final_attendance.xlsx'
    if attendance_list:
        df = pd.DataFrame(attendance_list)
        df.to_excel(output_path, index=False)
        print(f"Final attendance saved to {output_path}")
    
    # Send the file as an attachment for download
    return send_file(output_path, as_attachment=True, download_name="final_attendance.xlsx")

@app.route('/admin/stop_server', methods=['POST'])
def stop_server():
    # Get and print the final attendance list
    attendance_list = get_attendance_list()
    print("\nFinal Attendance List:")
    print(json.dumps(attendance_list, indent=2))
    
    # Save attendance list to an Excel file
    if attendance_list:
        df = pd.DataFrame(attendance_list)
        output_path = 'final_attendance.xlsx'
        df.to_excel(output_path, index=False)
        print(f"Final attendance saved to {output_path}")
    
    # Shutdown the server
    shutdown_server()
    return jsonify({
        "status": "Server shutting down",
        "final_attendance": attendance_list
    })

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()

if __name__ == '__main__':
    app.run(debug=True, port=8000)
