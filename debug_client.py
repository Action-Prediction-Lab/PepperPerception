import zmq
import cv2
import numpy as np
import time
import json
import threading

def main():
    # Context
    context = zmq.Context()
    
    # 1. Connect to Video Stream (PUB) from PepperBox
    video_sub = context.socket(zmq.SUB)
    video_sub.connect("tcp://localhost:5559")
    video_sub.setsockopt_string(zmq.SUBSCRIBE, "video")
    print("Connected to Video Stream (5559)")

    # 2. Connect to Perception Service (REP) from PepperPerception
    def connect_perception():
        req = context.socket(zmq.REQ)
        req.connect("tcp://localhost:5557")
        req.setsockopt(zmq.RCVTIMEO, 5000) # 5s timeout
        req.setsockopt(zmq.SNDTIMEO, 5000)
        return req

    perception_req = connect_perception()
    print("Connected to Perception Service (5557)")
    
    # GUI Setup
    window_name = "Pepper Vision Debugger"
    cv2.namedWindow(window_name)
    
    # Trackbar for Confidence
    def nothing(x): pass
    cv2.createTrackbar("Confidence %", window_name, 50, 100, nothing)

    print("Starting visual debugger...")
    
    while True:
        try:
            # A. Receive Frame
            if video_sub.poll(100):
                try:
                    topic, msg = video_sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue

                # Check Resolution
                if len(msg) == 230400:
                    h, w = 240, 320
                elif len(msg) == 921600:
                    h, w = 480, 640
                else:
                    print(f"Unknown frame size: {len(msg)}")
                    continue
                    
                # Decode to Numpy
                frame = np.frombuffer(msg, dtype=np.uint8).reshape((h, w, 3))
                
                # Convert RGB to BGR for OpenCV display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # B. Send to Perception (Needs JPG bytes)
                # CV2.imencode expects BGR input.
                _, jpg_encoded = cv2.imencode('.jpg', display_frame)
                
                # REQ/REP is blocking.
                try:
                    perception_req.send_multipart([b'{}', jpg_encoded.tobytes()])
                    # Wait for reply
                    result_json = perception_req.recv_json()
                except (zmq.Again, zmq.ZMQError) as e:
                    print(f"Warning: Perception timed out or error ({e}). Reconnecting...")
                    # Lazy Pirate: Close and Reopen
                    perception_req.close()
                    perception_req = connect_perception()
                    result_json = {}
                except Exception as e:
                    print(f"Perception Error: {e}")
                    result_json = {}
                
                # C. Overlay Results
                # Service returns: {"status": "success", "data": [...]}
                detections = result_json.get("data", [])
                
                # Get threshold from slider
                current_thresh = cv2.getTrackbarPos("Confidence %", window_name) / 100.0
                
                filtered_detections = []
                
                # Draw Box
                for det in detections:
                    class_name = det["class"]
                    conf = det["confidence"]
                    bbox = det["bbox"] # [x1, y1, x2, y2]
                    
                    if conf < current_thresh:
                        continue
                    
                    filtered_detections.append(det)
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Fancy colors per class (hash based)
                    color_seed = sum(map(ord, class_name))
                    color = ((color_seed * 50) % 255, (color_seed * 80) % 255, (color_seed * 110) % 255)
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label with background
                    label = f"{class_name} {conf:.2f}"
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw Detection Count on Frame
                info_text = f"Detections: {len(filtered_detections)} (Thresh: {current_thresh:.2f})"
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # D. Show
                cv2.imshow(window_name, display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                pass

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
            
    cv2.destroyAllWindows()
    context.term()

if __name__ == "__main__":
    main()
