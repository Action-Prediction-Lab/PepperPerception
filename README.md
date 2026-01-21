# PepperPerception

PepperPerception is a lightweight perception service designed to provide object detection capabilities to the Pepper robot (or other applications) via a ZeroMQ interface. 
- It utilizes [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for robust and fast object detection.

This service is designed to run in a Docker container, preferably on a machine with an NVIDIA GPU.

## Features

- **ZeroMQ Interface**
- **Dockerised**
- **GPU Accelerated**

## Capabilities
 - **Object Detection** YOLOv8 Integration


## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Required for GPU support)

## Installation & Running

**Start the service:**
   ```bash
   docker-compose up --build
   ```

   The service will start and listen on port `5557`.

## API Documentation

The service uses a **ZeroMQ REP (Reply)** socket. Clients should use a **REQ (Request)** socket to communicate.

### Request Format
Send a multipart ZMQ message where **the last frame** contains the encoded image bytes (e.g., JPEG or PNG).
- Frame 0 (Optional): Metadata JSON string (currently unused, but reserved for future flags).
- Frame 1 (Last): Encoded Image Bytes.

### Response Format
The service replies with a JSON object string.

**Success Response:**
```json
{
  "status": "success",
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [100.0, 50.0, 250.0, 400.0] 
    },
    {
      "class": "cup",
      "confidence": 0.88,
      "bbox": [300.5, 200.0, 350.0, 280.0]
    }
  ]
}
```
*`bbox` format is `[x1, y1, x2, y2]` (Top-Left, Bottom-Right).*

**Error Response:**
```json
{
  "status": "error",
  "message": "Error description..."
}
```

### Example Client (Python)

```python
import zmq
import cv2
import json

# Load image
img = cv2.imread("test_image.jpg")
_, buffer = cv2.imencode('.jpg', img)
img_bytes = buffer.tobytes()

# Connect
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5557")

# Send Request
socket.send_multipart([b'{}', img_bytes])

# Receive Response
response = socket.recv_json()
print(json.dumps(response, indent=2))
```

## Tools & Benchmarking

The repository includes a benchmarking utility to test throughput and latency.

```bash
# Run benchmark (requires service to be running)
python benchmark.py
```

This will run a series of requests with a random image and report:
- Total Throughput (FPS)
- Min/Max/Mean/Median Latency
- P95 and P99 Latency

## Configuration

You can configure the model and device settings in `docker-compose.yml`.

- **Model Selection**: Change the command argument to use a different model (e.g., `yolov8n.pt`, `yolov8l.pt`).
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--model", "yolov8m.pt" ]
  ```
- **Ports**: Map the internal `5557` port to a different host port if needed.

## License

[MIT License](LICENSE)
