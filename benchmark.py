import zmq
import cv2
import time
import numpy as np
import sys
import statistics

def main():
    # Configuration
    endpoint = "tcp://localhost:5557"
    num_requests = 100
    warmup_requests = 10
    image_size = (640, 480) # Standard VGA

    print(f"Benchmarking PepperPerception at {endpoint}")
    print(f"Image Size: {image_size}, Requests: {num_requests}, Warmup: {warmup_requests}")

    # Generate random test image
    print("Generating test image...")
    img = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(endpoint)

    # Warmup
    print("Warming up...")
    for i in range(warmup_requests):
        socket.send_multipart([b'{"type":"warmup"}', img_bytes])
        socket.recv_json()
        sys.stdout.write(".")
        sys.stdout.flush()
    print("\nWarmup complete.")

    # Benchmark
    latencies = []
    print("Running benchmark...")
    
    start_total = time.time()
    
    for i in range(num_requests):
        req_start = time.time()
        
        socket.send_multipart([b'{"type":"benchmark"}', img_bytes])
        socket.recv_json()
        
        latencies.append((time.time() - req_start) * 1000) # ms
        
        if i % 10 == 0:
            sys.stdout.write("*")
            sys.stdout.flush()
            
    total_time = time.time() - start_total
    
    # Analyze
    print("\n\n--- Results ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {num_requests / total_time:.2f} FPS")
    print(f"Min Latency: {min(latencies):.2f} ms")
    print(f"Max Latency: {max(latencies):.2f} ms")
    print(f"Mean Latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median Latency: {statistics.median(latencies):.2f} ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")

    socket.close()
    context.term()

if __name__ == "__main__":
    main()
