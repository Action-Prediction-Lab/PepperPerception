import zmq
import cv2
import json
import sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)
        
    # Encode image to buffer
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()

    # Connect to service
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5557")
    
    print("Sending request...")
    # Send image as multipart message (simulating expected protocol)
    socket.send_multipart([b'{"meta": "data"}', img_bytes])
    
    # Receive reply
    reply = socket.recv_json()
    print("Received reply:")
    print(json.dumps(reply, indent=2))
    
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
