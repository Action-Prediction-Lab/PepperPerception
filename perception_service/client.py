import zmq
import json
import cv2

class PerceptionClient:
    def __init__(self, service_uri="tcp://localhost:5557"):
        self.service_uri = service_uri
        self.context = zmq.Context()
        self.socket = self._new_socket()
        print(f"PerceptionClient: Connected to {service_uri}")

    def _new_socket(self):
        """REQ socket with LINGER 0; the default (-1) makes close() block on an abandoned request."""
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.service_uri)
        return socket

    def detect(self, img_bgr, target_label=None):
        """Send a BGR frame and return the service's `data` payload, or None on timeout/error.

        `data` is a list for the yolo backend and a dict for mediapipe and combined.
        """
        try:
            # Encode to JPG
            _, img_jpg = cv2.imencode('.jpg', img_bgr)
            
            # Send Request
            # Protocol: [MetadataJSON, ImageBytes]
            meta = {}
            if target_label:
                meta["target"] = target_label
            
            self.socket.send_multipart([json.dumps(meta).encode(), img_jpg.tobytes()])
            
            # Wait for Reply (Blocking but fast)
            if self.socket.poll(1000): # 1s timeout
                result = self.socket.recv_json()
                return result.get("data", {})
            else:
                print("PerceptionClient: Timeout")
                # REQ is strict-alternating, so a timed-out socket cannot be reused.
                self.socket.close()
                self.socket = self._new_socket()
                return None
                
        except Exception as e:
            print(f"PerceptionClient Error: {e}")
            return None

    def close(self):
        self.socket.close()
        self.context.term()
