from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image):
        """Detect on a BGR ndarray; returns a list (yolo) or dict (mediapipe), per implementation."""
        pass
