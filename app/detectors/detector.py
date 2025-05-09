import cv2
import time
import threading
from mediapipe import solutions
from app.detectors.utils import calculate_angle, draw_landmarks_on_image, open_camera, is_landmark_visible
from app.detectors.bicep_curl import bicep_curl_tracker

# Shared state
stats = {
    "bicep_curl": {
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
    if exercise_name == "bicep_curl":
        global running
        return bicep_curl_tracker(running=running, stats=stats["bicep_curl"])
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

def get_bicep_curl_stats():
    """Get bicep curl specific stats."""
    with lock:
        return stats["bicep_curl"]

# choose_exercise("bicep_curl")  # Example usage, replace with actual exercise name