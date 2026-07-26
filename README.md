# PepperPerception

PepperPerception is a perception service designed to provide human and object detection capabilities to the Pepper robot (or other applications) via a ZeroMQ interface. 

It supports multiple backends:
- **Ultralytics YOLOv8**: For object detection.
- **Google MediaPipe Holistic**: For face, hand, and pose tracking.
- **Combined**: Runs **both** YOLO and MediaPipe simultaneously.

This service is designed to run in a Docker container, preferably on a machine with an NVIDIA GPU.

## Features

- **Multi-Backend**: Switch between YOLO (Object Detection), MediaPipe (Holistic Tracking), or Combined.
- **ZeroMQ Interface**: Exposes a fast and language-agnostic API using ZMQ (REP/REQ pattern).
- **Dockerised**: Easy to deploy with all dependencies encapsulated.
- **GPU Accelerated**: Configured to leverage NVIDIA GPUs for inference (YOLO).

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

## Configuration

By default, `docker-compose.yml` runs the **combined** backend. Change its `command` to switch:

- **Combined** (default):
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--backend", "combined" ]
  ```

- **YOLO only**:
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--backend", "yolo" ]
  ```

- **MediaPipe only**:
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--backend", "mediapipe" ]
  ```

With no flags the module runs `--backend yolo --model yolov8n.pt`. Pass `--model` to select a different YOLO checkpoint. Note that `data` changes shape with the backend, as below.

## API Documentation

### Request Format
Send a multipart ZMQ message where **the last frame** contains the encoded image bytes.
- Frame 0 (Optional): Metadata JSON string.
- Frame 1 (Last): Encoded Image Bytes.

### Response Format
The service replies with a JSON object carrying `status`, `backend`, `inference_ms` and `data`. A request whose payload cannot be decoded returns `{"status": "error", "message": "..."}` instead, and the service stays up.

**YOLO Response:**
```json
{
  "status": "success",
  "backend": "yolo",
  "inference_ms": 12.4,
  "data": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [100.0, 50.0, 250.0, 400.0] 
    }
  ]
}
```

**MediaPipe Response:**
```json
{
  "status": "success",
  "backend": "mediapipe",
  "inference_ms": 12.4,
  "data": {
    "pose_landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}, ...],
    "face_landmarks": [...],
    "left_hand_landmarks": [...],
    "right_hand_landmarks": [...]
  }
}
```

**Combined Response:**
```json
{
  "status": "success",
  "backend": "combined",
  "inference_ms": 12.4,
  "data": {
     "detections": [...], // YOLO results
     "pose_landmarks": [...], // MediaPipe results
     "face_landmarks": [...],
     ...
  }
}
```

## Tests

Unit tests run inside the image, (the image does not ship with pytest, so the wrapper installs it at startup):

```bash
./tests/run-in-docker.sh tests/test_yolo_parsing.py -v
```

The integration guard checks the live ZMQ contract and needs the service running:

```bash
docker compose up -d
PYTHONPATH=tests python3 tests/test_service_contract.py
```

## Tools & Benchmarking

The repository includes a benchmarking utility.

```bash
# Run benchmark (requires service to be running)
python benchmark.py
```

## License

This project is licensed under the Apache License 2.0, see the [LICENSE](LICENSE) file for details.
