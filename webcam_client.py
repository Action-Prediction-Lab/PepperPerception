import cv2
import zmq
import numpy as np
import time

def main():
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5557")

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Starting Webcam Stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize for speed (optional, YOLO is fast though)
        # frame = cv2.resize(frame, (640, 480))

        # Encode frame
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # Measure latency
        start_time = time.time()
        
        # Send
        socket.send_multipart([b'{"type":"webcam"}', img_encoded.tobytes()])
        
        # Receive
        response = socket.recv_json()
        
        latency = (time.time() - start_time) * 1000 # ms
        
        # Draw Detections
        if "detections" in response:
            for det in response["detections"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = f"{det['class']} {det['confidence']:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Latency
        cv2.putText(frame, f"Latency: {latency:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show
        cv2.imshow('PepperPerception Demo', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
