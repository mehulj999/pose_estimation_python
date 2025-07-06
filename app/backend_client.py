import requests
import json
from datetime import datetime
from typing import Dict, Optional

class BackendClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.current_session_id: Optional[int] = None
        self.current_exercise_set_id: Optional[int] = None
        
    def authenticate(self, username: str = "test@example.com", password: str = "testpassword123") -> bool:
        """Authenticate with the backend and get access token."""
        try:
            auth_data = {
                "grant_type": "password",
                "username": username,
                "password": password,
                "scope": "",
                "client_id": "string",
                "client_secret": "string"
            }
            
            response = requests.post(f"{self.base_url}/auth/token", data=auth_data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            print("✅ Successfully authenticated with backend")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def create_workout_session(self, notes: str = "Pose estimation workout") -> Optional[int]:
        """Create a new workout session."""
        try:
            session_data = {"notes": notes}
            headers = self.get_headers()
            
            response = requests.post(
                f"{self.base_url}/workouts/sessions",
                json=session_data,
                headers=headers
            )
            response.raise_for_status()
            
            session_info = response.json()
            self.current_session_id = session_info["id"]
            print(f"✅ Created workout session with ID: {self.current_session_id}")
            return self.current_session_id
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create workout session: {e}")
            return None
    
    def create_exercise_set(self, exercise_type: str, arm_used: str, set_number: int = 1) -> Optional[int]:
        """Create a new exercise set."""
        if not self.current_session_id:
            print("❌ No active workout session. Create session first.")
            return None
            
        try:
            exercise_data = {
                "exercise_type": exercise_type,
                "arm_used": arm_used,
                "reps_completed": 0,
                "set_number": set_number,
                "duration": 0,
                "avg_angle_range": 0,
                "form_quality_score": 0,
                "rep_consistency_score": 0,
                "avg_rep_speed": 0,
                "min_angle_achieved": 0,
                "max_angle_achieved": 0
            }
            
            headers = self.get_headers()
            
            # Use the correct endpoint for adding exercises to sessions
            endpoint = f"{self.base_url}/workouts/sessions/{self.current_session_id}/exercises"
            
            try:
                print(f"Trying endpoint: {endpoint}")
                response = requests.post(endpoint, json=exercise_data, headers=headers)
                if response.status_code == 200 or response.status_code == 201:
                    exercise_info = response.json()
                    self.current_exercise_set_id = exercise_info["id"]
                    print(f"✅ Created exercise set with ID: {self.current_exercise_set_id}")
                    return self.current_exercise_set_id
                else:
                    print(f"❌ Endpoint {endpoint} returned status {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to reach {endpoint}: {e}")
                return None
            
        except Exception as e:
            print(f"❌ Failed to create exercise set: {e}")
            return None
    
    def update_exercise_set(self, reps_completed: int, duration: float, 
                          min_angle: float, max_angle: float) -> bool:
        """Update the current exercise set with final stats."""
        if not self.current_exercise_set_id:
            print("❌ No active exercise set to update.")
            return False
            
        try:
            update_data = {
                "reps_completed": reps_completed,
                "duration": duration,
                "avg_angle_range": max_angle - min_angle,
                "min_angle_achieved": min_angle,
                "max_angle_achieved": max_angle
            }
            
            headers = self.get_headers()
            
            # Use the correct endpoint for updating exercises in sessions
            endpoint = f"{self.base_url}/workouts/sessions/{self.current_session_id}/exercises/{self.current_exercise_set_id}"
            
            try:
                print(f"Trying update endpoint: {endpoint}")
                response = requests.put(endpoint, json=update_data, headers=headers)
                if response.status_code == 200 or response.status_code == 201:
                    print(f"✅ Updated exercise set with {reps_completed} reps")
                    return True
                else:
                    print(f"❌ Update endpoint {endpoint} returned status {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to reach update endpoint {endpoint}: {e}")
                return False
            
        except Exception as e:
            print(f"❌ Failed to update exercise set: {e}")
            return False
    
    def post_pose_frame(self, frame_number: int, joint_positions: Dict[str, Dict[str, float]]) -> bool:
        """Post a pose frame with joint positions."""
        if not self.current_exercise_set_id:
            print("❌ No active exercise set for pose frame.")
            return False
            
        try:
            frame_data = {
                "exercise_set_id": self.current_exercise_set_id,
                "timestamp": datetime.now().isoformat(),
                "frame_number": frame_number,
                "joint_positions": joint_positions
            }
            
            headers = self.get_headers()
            response = requests.post(
                f"{self.base_url}/workouts/pose-frames",
                json=frame_data,
                headers=headers
            )
            response.raise_for_status()
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to post pose frame: {e}")
            return False
    
    def start_workout(self, exercise_type: str, arm_used: str, notes: str = "Pose estimation workout") -> bool:
        """Start a complete workout session with exercise set."""
        if not self.authenticate():
            return False
            
        if not self.create_workout_session(notes):
            return False
            
        if not self.create_exercise_set(exercise_type, arm_used):
            return False
            
        print("✅ Workout session started successfully")
        return True
    
    def end_workout(self, reps_completed: int, duration: float, 
                   min_angle: float, max_angle: float) -> bool:
        """End the workout by updating the exercise set."""
        success = self.update_exercise_set(reps_completed, duration, min_angle, max_angle)
        if success:
            print("✅ Workout session ended successfully")
        return success 