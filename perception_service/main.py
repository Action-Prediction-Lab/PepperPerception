import zmq
import cv2
import numpy as np
import json
import os
import sys
import argparse
from .detector import ObjectDetector

def main():
    parser = argparse.ArgumentParser(description="Pepper Perception Service")
    parser.add_argument("--port", type=int, default=5557, help="ZMQ REP port")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model name")
    args = parser.parse_args()

    print(f"Starting Perception Service on port {args.port}...")

    # Initialize ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    # Initialize Detector
    detector = ObjectDetector(model_name=args.model)

    print("Service Ready. Waiting for requests...")

    try:
        while True:
            # Wait for next request from client
            # We expect two frames: [metadata_json, image_bytes] or just image_bytes if simple
            # But simpler for now: recv_string assuming JSON? 
            # Or recv_pyobj?
            # Let's assume the standard ZMQ pattern for images: recv bytes
            
            # Simple protocol: Receive an image buffer directly? 
            # Or receive a multipart message: [header_json, image_bytes]
            
            # Let's implement a robust multipart handler
            msg_parts = socket.recv_multipart()
            
            if not msg_parts:
                socket.send_json({"error": "Empty message"})
                continue

            # First part is optional metadata, Last part is image
            # Ideally: [b'{"request_id": "123"}', b'<image_data>']
            
            # If single part, assume it's an encoded image
            img_data = msg_parts[-1]
            
            try:
                # Decode image
                np_arr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Could not decode image")
                
                # Run detection
                detections = detector.detect(image)
                
                # Send reply
                response = {
                    "status": "success",
                    "detections": detections
                }
                socket.send_json(response)
                print(f"Processed image: {len(detections)} detections found.")
                
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
