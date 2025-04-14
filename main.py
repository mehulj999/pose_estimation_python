# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = FastAPI()

# Enable CORS if Android client has origin restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

@app.get("/")
def root():
    return {"message": "Pose detection API is running."}

@app.post("/detect_pose/")
async def detect_pose(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file
