from ultralytics import YOLO

def main():
    print("Running validation on COCO128 dataset...")
    # Load the model (assumes 'yolov8n.pt' is downloaded relative to this script or in cache)
    model = YOLO('yolov8m.pt')
    
    # Run validation
    # This will automatically download coco128.yaml and the dataset if not present
    metrics = model.val(data='coco128.yaml')
    
    print("\n--- Validation Results ---")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
