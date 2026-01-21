from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image):
        """
        Process the image and return results.
        
        Args:
            image (numpy.ndarray): Input image in BGR format.
            
        Returns:
            dict or list: Detection results (format depends on implementation).
        """
        pass
