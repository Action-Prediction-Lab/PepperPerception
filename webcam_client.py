import cv2
import zmq
import time

# MediaPipe Pose Connections (Simplified)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_yolo(frame, detections):
    h, w, _ = frame.shape
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class']} {det['confidence']:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Determine text position (inside box if near top)
        if y1 - 10 > 10:
            text_y = y1 - 10
        else:
            text_y = y1 + text_h + 10
            
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_pose_skeleton(frame, landmarks, color=(255, 0, 0), thickness=2):
    if not landmarks:
        return
    
    h, w, _ = frame.shape
    points = {}
    
    # Draw points and cache coordinates
    for i, lm in enumerate(landmarks):
        if 'visibility' in lm and lm['visibility'] < 0.5:
             continue
        cx, cy = int(lm['x'] * w), int(lm['y'] * h)
        points[i] = (cx, cy)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        
    # Draw lines
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(frame, points[start_idx], points[end_idx], color, thickness)

def draw_landmarks(frame, landmarks, color=(0, 255, 255), thickness=2):
    if not landmarks:
        return
    h, w, _ = frame.shape
    for lm in landmarks:
        cx, cy = int(lm['x'] * w), int(lm['y'] * h)
        if 'visibility' in lm and lm['visibility'] < 0.5:
             continue
        cv2.circle(frame, (cx, cy), thickness, color, -1)

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

        # Resize for speed (optional)
        # frame = cv2.resize(frame, (640, 480))

        # Encode frame
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # Measure latency
        start_time = time.time()
        
        # Send
        try:
            socket.send_multipart([b'{"type":"webcam"}', img_encoded.tobytes()])
            response = socket.recv_json()
        except zmq.ZMQError as e:
            print(f"ZMQ Error: {e}")
            break
        
        latency = (time.time() - start_time) * 1000 # ms
        
        # Backend detection
        backend = response.get("backend", "unknown")
        data = response.get("data", None)

        if data:
            if backend == "yolo":
                draw_yolo(frame, data)
            elif backend == "mediapipe":
                # Use dedicated skeleton drawer for pose
                draw_pose_skeleton(frame, data.get("pose_landmarks"), color=(255, 100, 0)) 
                
                # Simple dots for face (too many lines otherwise)
                draw_landmarks(frame, data.get("face_landmarks"), color=(0, 255, 255), thickness=1)
                
                # Simple dots for hands (could add connections later)
                draw_landmarks(frame, data.get("left_hand_landmarks"), color=(0, 255, 0))
                draw_landmarks(frame, data.get("right_hand_landmarks"), color=(0, 255, 0))
            
            elif backend == "combined":
                 # YOLO part
                 if "detections" in data:
                     draw_yolo(frame, data["detections"])
                 
                 # MediaPipe part
                 draw_pose_skeleton(frame, data.get("pose_landmarks"), color=(255, 100, 0))
                 draw_landmarks(frame, data.get("face_landmarks"), color=(0, 255, 255), thickness=1)
                 # draw_landmarks(frame, data.get("left_hand_landmarks"), color=(0, 255, 0))
                 # draw_landmarks(frame, data.get("right_hand_landmarks"), color=(0, 255, 0))
                 
            elif "detections" in response: # Fallback
                 draw_yolo(frame, response["detections"])

        # Draw Latency and Backend
        cv2.putText(frame, f"Backend: {backend}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Latency: {latency:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
