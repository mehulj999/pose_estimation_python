import cv2
import numpy as np
from mediapipe import solutions
import time

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks on an image."""
    mp_drawing = solutions.drawing_utils
    mp_pose = solutions.pose
    annotated_image = rgb_image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        detection_result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    )
    return annotated_image

def open_camera(index=0, retries=5, delay=1.0):
    """Open the camera with retries."""
    for i in range(retries):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("✅ Camera opened successfully")
            return cap
        print(f"❌ Camera open attempt {i+1}/{retries} failed. Retrying...")
        time.sleep(delay)
    return None

def is_landmark_visible(landmark, confidence_threshold=0.3):
    """Check if landmark is visible enough to be reliable."""
    return landmark.visibility > confidence_threshold if hasattr(landmark, 'visibility') else True