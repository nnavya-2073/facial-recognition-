import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load face encodings for known people
faculty_face_names = ["amritha"]
student_face_names = ["navya", "nireeksha", "shraddha"]

# Combine both lists for encoding
known_face_names = faculty_face_names + student_face_names
known_face_encodings = []

# Load encodings for faculty
for name in faculty_face_names:
    image = face_recognition.load_image_file(f"photos/{name}.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

# Load encodings for students
for name in student_face_names:
    image = face_recognition.load_image_file(f"photos/{name}.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

# Initialize attendance tracking
current_time = datetime.now().strftime("%H:%M:%S")
attendance = {
    name: {"Attendance": "absent", "Time": current_time}
    for name in known_face_names
}
# Function to get the current month name
def get_current_month_name():
    return datetime.now().strftime("%B")

# Function to create the CSV file with headers if needed
def ensure_csv_headers():
    file_name = "attendance.csv"
    if not os.path.isfile(file_name):  # Create the file if it doesn't exist
        with open(file_name, 'w', newline='') as f:
            lnwriter = csv.writer(f)
            lnwriter.writerow([get_current_month_name()])  # Add current month as header
            lnwriter.writerow(["Date", "Name", "Attendance", "Time"])  # Column headers
    else:
        # Check if headers are already written
        with open(file_name, 'r') as f:
            first_line = f.readline().strip()
            if get_current_month_name() not in first_line:
                # Add new headers with the current month if not present
                with open(file_name, 'a', newline='') as f_write:
                    lnwriter = csv.writer(f_write)
                    lnwriter.writerow([get_current_month_name()])  # Month header
                    lnwriter.writerow(["Date", "Name", "Attendance", "Time"])  # Column headers

ensure_csv_headers()  # Ensure headers are in place

# Flag to control when others' attendance can be taken
faculty_recognized = False

# Process video frames for face recognition
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    scale_factor = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        name = known_face_names[best_match_index] if matches[best_match_index] else None

        # Check if Amritha (faculty) is recognized
        if name == "amritha":
            top, right, bottom, left = face_location
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            left = int(left / scale_factor)

            # Draw a red rectangle around Amritha's face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Mark attendance for Amritha if not already marked
            if attendance[name]["Attendance"] == "absent":
                current_time = datetime.now().strftime("%H:%M:%S")
                attendance[name]["Attendance"] = "present"
                attendance[name]["Time"] = current_time
                faculty_recognized = True
            if faculty_recognized:
                cv2.putText(frame, "[Name of the Faculty]/[Subject]", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check for students only if faculty is recognized
        elif faculty_recognized and name in student_face_names:
            top, right, bottom, left = face_location
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            left = int(left / scale_factor)

            # Draw a green rectangle around the student's face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Mark attendance for the student if not already marked
            if attendance[name]["Attendance"] == "absent":
                current_time = datetime.now().strftime("%H:%M:%S")
                attendance[name]["Attendance"] = "present"
                attendance[name]["Time"] = current_time
            # Collect recognized student names
                recognized_students = [name for name in student_face_names if attendance[name]["Attendance"] == "present"]

            # Display recognized student names on the frame
            for i, recognized_name in enumerate(recognized_students):
                cv2.putText(frame, f"{recognized_name} ", (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    # Display the frame
    cv2.imshow("Face Recognition Attendance", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()

# Append new data to the CSV file
file_name = "attendance.csv"
with open(file_name, 'a', newline='') as f:
    lnwriter = csv.writer(f)
    for name, details in attendance.items():
        lnwriter.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            name,
            details["Attendance"],
            details["Time"]
        ])
