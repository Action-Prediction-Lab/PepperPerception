# PepperPerception

PepperPerception is a lightweight perception service designed to provide object detection capabilities to the Pepper robot (or other applications) via a ZeroMQ interface. 

It supports multiple backends:
- **Ultralytics YOLOv8**: For object detection.
- **Google MediaPipe Holistic**: For face, hand, and pose tracking.
- **Combined**: Runs **both** YOLO and MediaPipe simultaneously.

This service is designed to run in a Docker container, preferably on a machine with an NVIDIA GPU.

## Features

- **Multi-Backend**: Switch between YOLO (Object Detection), MediaPipe (Holistic Tracking), or Combined.
- **ZeroMQ Interface**: Exposes a fast and language-agnostic API using ZMQ (REP/REQ pattern).
- **Dockerized**: Easy to deploy with all dependencies encapsulated.
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

You can configure the backend in `docker-compose.yml`.

- **YOLO Backend (Default)**:
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--backend", "yolo", "--model", "yolov8m.pt" ]
  ```

- **MediaPipe Backend**:
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--backend", "mediapipe" ]
  ```

- **Combined Backend**:
  ```yaml
  command: [ "python", "-m", "perception_service.main", "--backend", "combined" ]
  ```

## API Documentation

### Request Format
Send a multipart ZMQ message where **the last frame** contains the encoded image bytes.
- Frame 0 (Optional): Metadata JSON string.
- Frame 1 (Last): Encoded Image Bytes.

### Response Format
The service replies with a JSON object.

**YOLO Response:**
```json
{
  "status": "success",
  "backend": "yolo",
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
  "data": {
     "detections": [...], // YOLO results
     "pose_landmarks": [...], // MediaPipe results
     "face_landmarks": [...],
     ...
  }
}
```

## Tools & Benchmarking

The repository includes a benchmarking utility.

```bash
# Run benchmark (requires service to be running)
python benchmark.py
```

## License

[MIT License](LICENSE)
