import cv2
import time
import threading
from mediapipe import solutions
from app.detectors.utils import calculate_angle, draw_landmarks_on_image, open_camera, is_landmark_visible
from app.detectors.right_arm_bicep_curl import right_arm_bicep_curl_tracker

# Shared state
stats = {
    "right_arm_bicep_curl": {
        "reps": 0,
        "direction": None,
        "last_angle": None,
        "status": "waiting"
    },
    "left_arm_bicep_curl": {
        "reps": 0,
        "direction": None,
        "last_angle": None,
        "status": "waiting"
    }
}
running = True  # Camera and tracker state
lock = threading.Lock()

def choose_exercise(exercise_name):
    """Choose the exercise to track."""
    global running
    if exercise_name == "right_arm_bicep_curl":
        return right_arm_bicep_curl_tracker(running=running, stats=stats["right_arm_bicep_curl"])
    elif exercise_name == "left_arm_bicep_curl":
        from app.detectors.left_arm_bicep_curl import left_arm_bicep_curl_tracker
        return left_arm_bicep_curl_tracker(running=running, stats=stats["left_arm_bicep_curl"])
    else:
        raise ValueError(f"Exercise '{exercise_name}' not supported.")
    
def stop_tracker():
    """Stop the tracker gracefully."""
    global running
    running = False

def get_stats(exercise_name=None):
    """Return current stats with thread safety."""
    with lock:
        if exercise_name:
            if exercise_name in stats:
                return stats[exercise_name]
            return None
        return stats

def get_right_arm_bicep_curl_stats():
    """Get right arm bicep curl specific stats."""
    with lock:
        return stats["right_arm_bicep_curl"]

def get_left_arm_bicep_curl_stats():
    """Get left arm bicep curl specific stats."""
    with lock:
        return stats["left_arm_bicep_curl"]

# choose_exercise("left_arm_bicep_curl")  # Example usage, replace with actual exercise name