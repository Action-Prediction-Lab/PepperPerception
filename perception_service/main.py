import zmq
import cv2
import numpy as np
import json
import os
import sys
import argparse

# Import new detectors
from .detectors.yolo_detector import YOLODetector
from .detectors.base import BaseDetector

class CombinedDetector(BaseDetector):
    def __init__(self, model_name='yolov8n.pt'):
        from .detectors.mediapipe_detector import MediaPipeDetector
        print("Initializing Combined Backend...")
        self.yolo = YOLODetector(model_name=model_name)
        self.mediapipe = MediaPipeDetector()
        print("Combined Backend Ready.")

    def detect(self, image):
        # Run both sequentially
        det_yolo = self.yolo.detect(image)
        det_mp = self.mediapipe.detect(image)
        
        # Merge results
        return {
            "detections": det_yolo,
            **det_mp # Unpack pose, face, hand landmarks
        }

def get_detector(backend, model_name):
    if backend == 'yolo':
        return YOLODetector(model_name=model_name)
    elif backend == 'mediapipe':
        from .detectors.mediapipe_detector import MediaPipeDetector
        return MediaPipeDetector()
    elif backend == 'combined':
        return CombinedDetector(model_name=model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def main():
    import time # Re-import locally to force availability
    parser = argparse.ArgumentParser(description="Pepper Perception Service")
    parser.add_argument("--port", type=int, default=5557, help="ZMQ REP port")
    parser.add_argument("--backend", type=str, default="yolo", choices=['yolo', 'mediapipe', 'combined'], help="Detection backend")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model name (for YOLO)")
    args = parser.parse_args()

    print(f"Starting Perception Service on port {args.port} using {args.backend}...")
    print(f"Time module loaded: {time}")

    # Initialize ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Initialize Detector
    try:
        import traceback
        detector = get_detector(args.backend, args.model)
    except Exception as e:
        print(f"CRITICAL Error loading backend {args.backend}: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Service Ready. Waiting for requests...")

    try:
        while True:
            # Wait for next request
            msg_parts = socket.recv_multipart()
            
            if not msg_parts:
                socket.send_json({"error": "Empty message"})
                continue

            # First part is optional metadata, Last part is image
            img_data = msg_parts[-1]
            
            try:
                # Decode image
                np_arr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Could not decode image")
                
                # Run detection
                start_t = time.time()
                results = detector.detect(image)
                end_t = time.time()
                
                inference_ms = (end_t - start_t) * 1000
                print(f"Inference: {inference_ms:.2f}ms (FPS: {1000/inference_ms:.1f})")
                
                # Send reply
                response = {
                    "status": "success",
                    "backend": args.backend,
                    "inference_ms": inference_ms,
                    "data": results
                }
                
                socket.send_json(response)
                
            except Exception as e:
                print(f"Error processing request: {e}")
                socket.send_json({"status": "error", "message": str(e)})

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
