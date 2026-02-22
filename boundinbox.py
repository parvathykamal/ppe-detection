import torch
import cv2
from pathlib import Path

# Configuration
VIDEO_PATH = r"C:\ppe_unified_dataset_clean\resultvideo.mp4"
MODEL_PATH = r"C:\ppe_unified_dataset_clean\runs\ppe\advanced_balanced_v1\weights\epoch60.pt"
OUTPUT_PATH = r"C:\ppe_unified_dataset_clean\output_video_with_detections.mp4"

# Detection parameters
CONF_THRESHOLD = 0.25  # Confidence threshold
IOU_THRESHOLD = 0.45   # NMS IOU threshold

def process_video_with_yolo():
    """Process video using YOLOv5/v8 model"""
    
    # Load the model using torch.hub (for YOLOv5) or ultralytics (for YOLOv8)
    print(f"Loading model from: {MODEL_PATH}")
    
    # Try YOLOv5 approach first
    try:
        # For YOLOv5 custom trained model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
        model.conf = CONF_THRESHOLD
        model.iou = IOU_THRESHOLD
        print("Model loaded successfully using YOLOv5")
    except Exception as e1:
        print(f"YOLOv5 loading failed: {e1}")
        try:
            # Try YOLOv8/ultralytics approach
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            print("Model loaded successfully using YOLOv8/Ultralytics")
        except Exception as e2:
            print(f"YOLOv8 loading failed: {e2}")
            raise ValueError("Could not load model with either YOLOv5 or YOLOv8 methods")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("Processing video...")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IOU threshold: {IOU_THRESHOLD}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame)
        
        # Render results on frame
        if hasattr(results, 'render'):
            # YOLOv5 style
            annotated_frame = results.render()[0]
        elif hasattr(results[0], 'plot'):
            # YOLOv8 style
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\nProcessing complete!")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    try:
        process_video_with_yolo()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()