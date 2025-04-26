from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import threading
import os
import signal
from app.detector import bicep_curl_tracker, get_stats, stop_tracker

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Pose detection API is running."}

@app.get("/stats")
def read_stats():
    stats = get_stats()
    print(f"Returning stats: Reps={stats['reps']}, Direction={stats['direction']}")  # Debugging
    return JSONResponse(content=stats)

@app.get("/shutdown")
def shutdown():
    stop_tracker()
    print("🛑 Shutting down the application...")
    os.kill(os.getpid(), signal.SIGINT)

@app.on_event("startup")
def start_background_tracker():
    thread = threading.Thread(target=bicep_curl_tracker, daemon=True)
    thread.start()