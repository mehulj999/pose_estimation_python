import cv2
import time
import numpy as np
import threading
from mediapipe import solutions
from app.detectors.utils import calculate_angle, draw_landmarks_on_image, open_camera, is_landmark_visible

def bicep_curl_tracker(running, stats):
    """Background function to track bicep curls using live camera."""
    # Initialize MediaPipe Pose
    mp_pose = solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = open_camera()
    if cap is None:
        print("❌ Cannot open camera. Exiting background thread.")
        running = False
        return

    print("🎥 Running bicep curl detector — Press 'q' in the video window to stop")
    cv2.namedWindow("Bicep Curl Tracker", cv2.WINDOW_NORMAL)
    
    # Variables for stability tracking
    valid_frames_count = 0
    REQUIRED_VALID_FRAMES = 2  # Require consecutive valid frames before counting reps
    REQUIRED_VISIBLE_FRAMES = 3  # Require consecutive frames with all landmarks visible
    REQUIRED_FACE_VISIBLE_FRAMES = 3  # Require consecutive frames with face visible
    visible_frames_count = 0
    face_visible_frames_count = 0
    ready_to_count = False
    tracking_active = False
    status_text = "Waiting for pose..."

    while running:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ Empty frame received. Skipping.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Default status if no pose is detected
        status_text = "Waiting for pose..."
        tracking_active = False
        all_visible = False
        face_visible = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Landmarks for RIGHT arm (from person's perspective)
            shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
            nose_idx = mp_pose.PoseLandmark.NOSE.value
            
            # Check if landmarks exist in the detected pose
            if (shoulder_idx < len(landmarks) and 
                elbow_idx < len(landmarks) and 
                wrist_idx < len(landmarks) and
                nose_idx < len(landmarks)):
                # Check visibility of all three arm landmarks
                if (is_landmark_visible(landmarks[shoulder_idx]) and
                    is_landmark_visible(landmarks[elbow_idx]) and
                    is_landmark_visible(landmarks[wrist_idx])):
                    visible_frames_count += 1
                    all_visible = True
                else:
                    visible_frames_count = 0
                    ready_to_count = False
                # Check visibility of face (nose)
                if is_landmark_visible(landmarks[nose_idx]):
                    face_visible_frames_count += 1
                    face_visible = True
                else:
                    face_visible_frames_count = 0
                    ready_to_count = False

                if visible_frames_count >= REQUIRED_VISIBLE_FRAMES and face_visible_frames_count >= REQUIRED_FACE_VISIBLE_FRAMES:
                    ready_to_count = True
                else:
                    ready_to_count = False

                valid_frames_count += 1
                
                if valid_frames_count >= REQUIRED_VALID_FRAMES and ready_to_count:
                    tracking_active = True
                    
                    def get_point(index):
                        return (int(landmarks[index].x * w), int(landmarks[index].y * h))

                    right_shoulder = get_point(shoulder_idx)
                    right_elbow = get_point(elbow_idx)
                    right_wrist = get_point(wrist_idx)
                    
                    # Calculate angle for right arm
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    # Update stats
                    stats["last_angle"] = right_angle
                    stats["status"] = "tracking"

                    # Improved bicep curl logic for right arm
                    # Only count a rep when going from down (angle > 130) to up (angle < 90)
                    if right_angle > 130:
                        stats["direction"] = "down"
                        status_text = "Down position detected"
                    elif right_angle < 90:
                        if stats["direction"] == "down":
                            stats["reps"] += 1
                            stats["direction"] = "up"
                            status_text = "Up position detected (Rep counted)"
                        else:
                            stats["direction"] = "up"
                            status_text = "Up position detected"
                    else:
                        status_text = f"Moving... Angle: {int(right_angle)}"

                    # Draw visuals
                    cv2.putText(frame, f"Angle: {int(right_angle)}", right_elbow,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 51), 2)
                elif not ready_to_count:
                    if not face_visible:
                        status_text = "Show your face to start counting reps."
                    else:
                        status_text = "Show your full arm and hand to start counting reps."
                    stats["status"] = "not_ready"
                else:
                    status_text = f"Stabilizing... ({valid_frames_count}/{REQUIRED_VALID_FRAMES})"
                    stats["status"] = "stabilizing"
            else:
                valid_frames_count = 0
                visible_frames_count = 0
                face_visible_frames_count = 0
                ready_to_count = False
                status_text = "Landmarks not found"
                stats["status"] = "waiting"
        else:
            valid_frames_count = 0
            visible_frames_count = 0
            face_visible_frames_count = 0
            ready_to_count = False
            stats["status"] = "waiting"

        # Draw status box (enlarged for more info)
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)

        cv2.putText(frame, f"RIGHT ARM | Curls: {stats['reps']}", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Annotate image with pose landmarks
        annotated_image = draw_landmarks_on_image(rgb_frame, results)
        output_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        output_frame = cv2.addWeighted(output_frame, 0.7, frame, 0.3, 0)

        # Show tracking status indicator
        if tracking_active:
            cv2.putText(output_frame, "TRACKING ACTIVE", (output_frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(output_frame, "TRACKING INACTIVE", (output_frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS overlay
        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-6)
        cv2.putText(output_frame, f"FPS: {int(fps)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Bicep Curl Tracker", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Tracker stopped.")