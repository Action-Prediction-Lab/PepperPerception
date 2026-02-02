# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless numpy && \
    pip install "numpy<2.0.0" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "perception_service.main"]
