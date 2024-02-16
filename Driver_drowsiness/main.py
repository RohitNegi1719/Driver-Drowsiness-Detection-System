import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to convert dlib shape object to NumPy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    e_a_r = (A + B) / (2.0 * C)
    return e_a_r

# Set the EAR threshold for drowsiness detection
EAR_THRESHOLD = 0.20
CONSECUTIVE_FRAMES_THRESHOLD = 35
consecutive_frames_closed = 0

# Initialize the webcam
capture = cv2.VideoCapture(0)

# Initialize Pygame for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")  # Replace with your sound file

drowsiness_detected = False  # Track the presence of "Drowsiness Detected" text

# Start capturing frames and perform drowsiness detection
while True:
    ret, frame = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Reset the drowsiness_detected flag at the beginning of each loop iteration
    drowsiness_detected = False

    for face in faces:
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate EAR for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Calculate average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Draw the eyes and compute the convex hull for visualization
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        # Check if the EAR is below the threshold
        if ear < EAR_THRESHOLD:
            consecutive_frames_closed += 1
            if consecutive_frames_closed >= CONSECUTIVE_FRAMES_THRESHOLD:
                cv2.putText(frame, "Drowsiness Detected", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_sound.play()  # Play the alert sound
                drowsiness_detected = True
        else:
            consecutive_frames_closed = 0

    # Stop the alert sound if "Drowsiness Detected" text is not present in the current frame
    if not drowsiness_detected:
        alert_sound.stop()

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windowsA
capture.release()
cv2.destroyAllWindows()