from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import threading
import os
import signal
import cv2  # Import cv2 at the top level to catch import errors early
from app.detectors.detector import choose_exercise, get_stats, get_bicep_curl_stats, stop_tracker

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to track if the tracker is running
tracker_running = False
tracker_thread = None

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

@app.get("/stats/bicep_curl")
def read_bicep_curl_stats():
    """Get bicep curl specific stats."""
    try:
        stats = get_bicep_curl_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bicep_curl")
def start_bicep_curl():
    """Start the bicep curl tracker."""
    global tracker_running, tracker_thread
    
    if tracker_running:
        return {"message": "Tracker is already running."}
    
    print("Starting bicep curl tracker...")
    tracker_thread = threading.Thread(target=choose_exercise, args=("bicep_curl",), daemon=True)
    tracker_thread.start()
    tracker_running = True
    return {"message": "Bicep curl tracking started."}

@app.get("/shutdown")
def shutdown():
    """Shutdown the tracker and the server."""
    global tracker_running
    stop_tracker()
    tracker_running = False
    print("🛑 Shutting down the application...")
    os.kill(os.getpid(), signal.SIGINT)

@app.on_event("startup")
def start_background_tracker():
    """Start the tracker when the server starts."""
    global tracker_running, tracker_thread
    if not tracker_running:
        tracker_thread = threading.Thread(target=choose_exercise, args=("bicep_curl",), daemon=True)
        tracker_thread.start()
        tracker_running = True