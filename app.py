import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    for pose_landmarks in pose_landmarks_list:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image, proto, solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Load the pose model with VIDEO mode
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0
total_detection_time = 0
start_time = time.time()

print("Starting pose detection. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # Start timing
    detection_start = time.time()
    result = detector.detect_for_video(mp_image, timestamp)
    detection_end = time.time()

    # Calculate detection time
    detection_time = (detection_end - detection_start) * 1000  # ms
    total_detection_time += detection_time
    frame_count += 1

    # Draw landmarks
    annotated_image = draw_landmarks_on_image(rgb_frame, result)
    bgr_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Calculate and display FPS and average detection time
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_detection_time = total_detection_time / frame_count if frame_count > 0 else 0

    cv2.putText(bgr_annotated, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(bgr_annotated, f"Avg Detect Time: {avg_detection_time:.2f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Pose Detection", bgr_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
