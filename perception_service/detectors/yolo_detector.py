from ultralytics import YOLO
from .base import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_name (str): The name or path of the YOLO model to load.
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        print("Model loaded successfully.")

    def detect(self, image):
        """
        Run inference on an image.

        Args:
            image (numpy.ndarray): The input image (BGR format from OpenCV).

        Returns:
            list: A list of dictionaries containing detection results.
                  Each dict has: 'class', 'confidence', 'bbox' [x1, y1, x2, y2].
        """
        # Run inference
        results = self.model(image, verbose=False)
        
        detections = []
        
        # Parse results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence
                conf = float(box.conf[0])
                
                # Get class ID and name
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                
        return detections
