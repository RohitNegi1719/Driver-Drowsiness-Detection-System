# Drowsiness Detection System

This project aims to detect drowsiness in a person's eyes using a webcam feed. When drowsiness is detected, an alert sound is played to notify the user. The system utilizes facial landmarks detection and eye aspect ratio (EAR) calculation to determine the state of the eyes.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- dlib
- NumPy
- SciPy
- Pygame

## Installation

1. Clone the repository:

2. Navigate to the project directory:

3. Install the required libraries:

4. Download the pre-trained facial landmarks predictor file (`shape_predictor_68_face_landmarks.dat`) and place it in the project directory.

## Usage

1. Run the `main.py` script:

2. The webcam feed will open, and the system will start detecting drowsiness in real-time.
3. If drowsiness is detected (i.e., when the average eye aspect ratio falls below a certain threshold for a specified number of consecutive frames), an alert sound will be played, and "Drowsiness Detected" text will be displayed on the frame.
4. Press the 'q' key to exit the program and release the webcam resources.

## Customization

- Adjust the EAR threshold and consecutive frames threshold in the script to fine-tune the sensitivity of the drowsiness detection algorithm.
- Replace the alert sound file (`alert.mp3`) with your preferred sound file.

## Acknowledgments

- The project utilizes the dlib library for face detection and facial landmarks detection. Refer to the [dlib documentation](http://dlib.net/) for more information.
- Pygame is used to play the alert sound. Visit the [Pygame website](https://www.pygame.org/) for documentation and tutorials.
