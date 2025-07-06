from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import threading
import os
import signal
import cv2
import time
from app.detectors.detector import choose_exercise, get_stats, get_right_arm_bicep_curl_stats, get_left_arm_bicep_curl_stats, stop_tracker
from app.backend_client import BackendClient
import requests

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
tracker_running = False
tracker_thread = None
backend_client = BackendClient("http://localhost:8000")
workout_start_time = None
current_exercise_type = None
current_arm_used = None

@app.get("/")
def root():
    return {"message": "Pose detection API is running."}

@app.get("/stats")
def read_stats():
    """Get all exercise stats."""
    try:
        stats = get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/right_arm_bicep_curl")
def read_right_arm_bicep_curl_stats():
    """Get right arm bicep curl specific stats."""
    try:
        stats = get_right_arm_bicep_curl_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/left_arm_bicep_curl")
def read_left_arm_bicep_curl_stats():
    """Get left arm bicep curl specific stats."""
    try:
        stats = get_left_arm_bicep_curl_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/right_arm_bicep_curl")
def start_right_arm_bicep_curl():
    """Start the right arm bicep curl tracker."""
    global tracker_running, tracker_thread, workout_start_time, current_exercise_type, current_arm_used
    
    if tracker_running:
        return {"message": "Tracker is already running."}
    
    print("Starting right arm bicep curl tracker...")
    
    # Set tracking parameters
    workout_start_time = time.time()
    current_exercise_type = "bicep_curl"
    current_arm_used = "right"
    
    tracker_thread = threading.Thread(target=choose_exercise, args=("right_arm_bicep_curl",), daemon=True)
    tracker_thread.start()
    tracker_running = True
    return {"message": "Right arm bicep curl tracking started."}

@app.get("/left_arm_bicep_curl")
def start_left_arm_bicep_curl():
    """Start the left arm bicep curl tracker."""
    global tracker_running, tracker_thread, workout_start_time, current_exercise_type, current_arm_used
    
    if tracker_running:
        return {"message": "Tracker is already running."}
    
    print("Starting left arm bicep curl tracker...")
    
    # Set tracking parameters
    workout_start_time = time.time()
    current_exercise_type = "bicep_curl"
    current_arm_used = "left"
    
    tracker_thread = threading.Thread(target=choose_exercise, args=("left_arm_bicep_curl",), daemon=True)
    tracker_thread.start()
    tracker_running = True
    return {"message": "Left arm bicep curl tracking started."}

@app.get("/shutdown")
def shutdown():
    """Shutdown the tracker and post final stats to backend."""
    global tracker_running, workout_start_time, current_exercise_type, current_arm_used
    
    # Stop the tracker first to ensure stats are finalized
    stop_tracker()
    tracker_running = False
    
    # Small delay to ensure stats are updated
    time.sleep(0.5)
    
    # Get final stats after stopping
    try:
        if current_arm_used == "right":
            stats = get_right_arm_bicep_curl_stats()
        elif current_arm_used == "left":
            stats = get_left_arm_bicep_curl_stats()
        else:
            stats = {"reps": 0, "last_angle": 0}
        
        reps_completed = stats.get("reps", 0)
        duration = time.time() - workout_start_time if workout_start_time else 0
        min_angle = stats.get("last_angle", 0)
        max_angle = stats.get("last_angle", 0)
        
        print(f"📊 Final stats - Reps: {reps_completed}, Duration: {duration:.2f}s, Angle: {min_angle}")
        
        # Create workout session and exercise set with final stats
        if backend_client.authenticate():
            if backend_client.create_workout_session(f"{current_arm_used} arm bicep curl workout"):
                # Create exercise set with final stats
                exercise_data = {
                    "exercise_type": current_exercise_type or "bicep_curl",
                    "arm_used": current_arm_used or "right",
                    "reps_completed": reps_completed,
                    "set_number": 1,
                    "duration": duration,
                    "avg_angle_range": max_angle - min_angle,
                    "form_quality_score": 0,
                    "rep_consistency_score": 0,
                    "avg_rep_speed": 0,
                    "min_angle_achieved": min_angle,
                    "max_angle_achieved": max_angle
                }
                
                headers = backend_client.get_headers()
                endpoint = f"{backend_client.base_url}/workouts/sessions/{backend_client.current_session_id}/exercises"
                
                try:
                    response = requests.post(endpoint, json=exercise_data, headers=headers)
                    if response.status_code == 200 or response.status_code == 201:
                        print(f"✅ Posted workout data: {reps_completed} reps, {duration:.2f}s duration")
                    else:
                        print(f"⚠️ Failed to post exercise data: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ Error posting exercise data: {e}")
            else:
                print("⚠️ Failed to create workout session")
        else:
            print("⚠️ Failed to authenticate with backend")
            
    except Exception as e:
        print(f"⚠️ Error getting final stats: {e}")
    
    print("🛑 Shutting down the application...")
    os.kill(os.getpid(), signal.SIGINT)

@app.on_event("startup")
def start_background_tracker():
    """Initialize the app without starting any tracker automatically."""
    global tracker_running
    
    if not tracker_running:
        # Only authenticate with backend on startup
        if backend_client.authenticate():
            print("✅ Backend authentication successful")
        else:
            print("⚠️ Backend authentication failed, but continuing without backend integration")
        
        print("🚀 App initialized. Call /right_arm_bicep_curl or /left_arm_bicep_curl to start tracking.")