import os
import cv2
import threading
from deepface import DeepFace


# Function to check if the file exists and has the correct permissions
def check_file(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return False
    if not os.access(file_path, os.R_OK):
        print(f"Error: The file '{file_path}' is not readable.")
        return False
    return True

# Path to the reference video
file_path = "/Users/apple/Documents/WS/ai_401/myenv/liveFaceRecognition/vid.mov" #Replace vid.mov with your video.

# Check if the file exists and is readable
if not check_file(file_path):
    exit()

# Load reference video
cap_reference = cv2.VideoCapture(file_path)
if not cap_reference.isOpened():
    print("Error: Could not open reference video.")
    exit()

# Extract frames from the reference video
reference_frames = []
while True:
    ret, frame = cap_reference.read()
    if not ret:
        break
    reference_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap_reference.release()

# Function to check face match
def check_face(frame):
    global face_match
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for reference_img_rgb in reference_frames:
            result = DeepFace.verify(frame_rgb, reference_img_rgb, model_name='VGG-Face', detector_backend='opencv')
            if result['verified']:
                face_match = True
                print(f"Face match result: {result}")  # Debugging statement
                return
        face_match = False
    except Exception as e:
        print(f"Face verification error: {e}")
        face_match = False

# Open webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set video frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        # Check face match every 30 frames using threading
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        counter += 1

        # Display match result on the frame
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
