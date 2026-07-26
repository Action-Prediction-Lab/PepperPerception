import mediapipe as mp
import cv2
from .base import BaseDetector

class MediaPipeDetector(BaseDetector):
    def __init__(self, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initialize the MediaPipe Holistic detector.
        """
        print("Loading MediaPipe Holistic...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        print("MediaPipe Holistic loaded successfully.")

    def _process_landmarks(self, landmarks):
        """Helper to convert NormalizedLandmarkList to list of dicts."""
        if not landmarks:
            return None
        return [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in landmarks.landmark]

    def detect(self, image):
        """Run Holistic on a BGR image; returns pose/face/left_hand/right_hand landmark lists, None when absent."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.holistic.process(image_rgb)
        
        return {
            "pose_landmarks": self._process_landmarks(results.pose_landmarks),
            "face_landmarks": self._process_landmarks(results.face_landmarks),
            "left_hand_landmarks": self._process_landmarks(results.left_hand_landmarks),
            "right_hand_landmarks": self._process_landmarks(results.right_hand_landmarks)
        }
