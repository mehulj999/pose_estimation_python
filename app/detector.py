import cv2
import time
import threading
from mediapipe import solutions
from app.utils import calculate_angle, draw_landmarks_on_image, open_camera

# Shared state
rep_count = 0
direction = None  # 'up' or 'down'
running = True  # Camera and tracker state
lock = threading.Lock()

def stop_tracker():
    """Stop the tracker gracefully."""
    global running
    running = False

def get_stats():
    """Return current stats with thread safety."""
    with lock:
        return {"reps": rep_count, "direction": direction}

def is_landmark_visible(landmark, confidence_threshold=0.3):
    """Check if landmark is visible enough to be reliable."""
    # Lower threshold to be more lenient with visibility
    return landmark.visibility > confidence_threshold if hasattr(landmark, 'visibility') else True

def bicep_curl_tracker():
    """Background function to track bicep curls using live camera."""
    global rep_count, direction, running

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

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Landmarks for RIGHT arm (from person's perspective)
            shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
            
            # Check if landmarks exist in the detected pose
            if (shoulder_idx < len(landmarks) and 
                elbow_idx < len(landmarks) and 
                wrist_idx < len(landmarks)):
                
                # # For debugging - print landmarks visibility

                # print(f"Right Shoulder visibility: {landmarks[shoulder_idx].visibility if hasattr(landmarks[shoulder_idx], 'visibility') else 'N/A'}")
                # print(f"Right Elbow visibility: {landmarks[elbow_idx].visibility if hasattr(landmarks[elbow_idx], 'visibility') else 'N/A'}")
                # print(f"Right Wrist visibility: {landmarks[wrist_idx].visibility if hasattr(landmarks[wrist_idx], 'visibility') else 'N/A'}")
                
                # Attempt to track regardless of visibility scores initially
                valid_frames_count += 1
                
                if valid_frames_count >= REQUIRED_VALID_FRAMES:
                    tracking_active = True
                    
                    def get_point(index):
                        return (int(landmarks[index].x * w), int(landmarks[index].y * h))

                    right_shoulder = get_point(shoulder_idx)
                    right_elbow = get_point(elbow_idx)
                    right_wrist = get_point(wrist_idx)
                    
                    # Calculate angle for right arm
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    # Print angle for debugging
                    print(f"Right arm angle: {right_angle}")

                    # Draw visual indicators for points used in angle calculation
                    cv2.circle(frame, right_shoulder, 10, (0, 255, 0), -1)  # Green for shoulder
                    cv2.circle(frame, right_elbow, 10, (255, 0, 0), -1)     # Blue for elbow
                    cv2.circle(frame, right_wrist, 10, (0, 0, 255), -1)     # Red for wrist

                    # Bicep curl logic for right arm
                    with lock:
                        if right_angle < 90:  # Adjusted threshold for more lenient detection
                            if direction != 'up':
                                rep_count += 1
                                direction = 'up'
                                status_text = "Up position detected"
                        elif right_angle > 130:  # Adjusted threshold for more lenient detection
                            if direction == 'up':
                                direction = 'down'
                                status_text = "Down position detected"
                        else:
                            status_text = f"Moving... Angle: {int(right_angle)}"

                    # Draw visuals
                    cv2.putText(frame, f"Angle: {int(right_angle)}", right_elbow,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    status_text = f"Stabilizing... ({valid_frames_count}/{REQUIRED_VALID_FRAMES})"
            else:
                # Reset stability counter if key landmarks aren't available
                valid_frames_count = 0
                status_text = "Landmarks not found"
        else:
            # Reset stability counter if no pose detected
            valid_frames_count = 0

        # Draw status box (enlarged for more info)
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)

        with lock:
            cv2.putText(frame, f"RIGHT ARM | Curls: {rep_count}", (20, 40),
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