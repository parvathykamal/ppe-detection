from ultralytics import YOLO
import os
from pathlib import Path

# Paths
model_path = r"C:\ppe_unified_dataset_clean\runs\ppe\advanced_balanced_v1\weights\best.pt"
test_data_path = r"C:\ppe_unified_dataset_clean\test"

# Load the trained model
print("Loading model...")
model = YOLO(model_path)

# Run validation/testing on the test dataset
print("\nRunning model evaluation on test set...")
print("=" * 60)

# Perform validation which calculates comprehensive metrics
results = model.val(
    data=test_data_path,  # Path to test data
    split='test',          # Use test split
    save_json=True,        # Save results in COCO JSON format
    save_hybrid=True,      # Save hybrid labels (for analysis)
    conf=0.25,             # Confidence threshold
    iou=0.6,               # IoU threshold for NMS
    max_det=300,           # Maximum detections per image
    plots=True,            # Generate plots
    verbose=True           # Verbose output
)

# Display comprehensive metrics
print("\n" + "=" * 60)
print("EVALUATION METRICS SUMMARY")
print("=" * 60)

# Overall metrics
print("\n1. OVERALL PERFORMANCE:")
print(f"   mAP@0.5        : {results.box.map50:.4f}")
print(f"   mAP@0.5:0.95   : {results.box.map:.4f}")
print(f"   mAP@0.75       : {results.box.map75:.4f}")

# Precision and Recall
print("\n2. PRECISION & RECALL:")
print(f"   Precision      : {results.box.mp:.4f}")
print(f"   Recall         : {results.box.mr:.4f}")

# Per-class metrics
print("\n3. PER-CLASS METRICS:")
if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
    class_names = model.names
    for i, class_idx in enumerate(results.box.ap_class_index):
        class_name = class_names[int(class_idx)]
        print(f"\n   Class: {class_name}")
        print(f"   - AP@0.5       : {results.box.ap50[i]:.4f}")
        print(f"   - AP@0.5:0.95  : {results.box.ap[i]:.4f}")

# Speed metrics
print("\n4. INFERENCE SPEED:")
print(f"   Preprocess time : {results.speed['preprocess']:.2f} ms")
print(f"   Inference time  : {results.speed['inference']:.2f} ms")
print(f"   Postprocess time: {results.speed['postprocess']:.2f} ms")

# Additional statistics
print("\n5. DATASET STATISTICS:")
print(f"   Total images    : {results.box.nc if hasattr(results.box, 'nc') else 'N/A'}")
print(f"   Classes detected: {len(results.box.ap_class_index) if hasattr(results.box, 'ap_class_index') else 'N/A'}")

print("\n" + "=" * 60)
print("Evaluation complete! Results saved to runs/detect/val/")
print("=" * 60)

# Generate confusion matrix and other plots
print("\nGenerating detailed plots...")
results.confusion_matrix.plot(save_dir='runs/detect/val', names=model.names)

# Save metrics to file
output_file = "runs/detect/val/metrics_summary.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    f.write("OBJECT DETECTION MODEL EVALUATION METRICS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Test Data: {test_data_path}\n\n")
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Precision: {results.box.mp:.4f}\n")
    f.write(f"Recall: {results.box.mr:.4f}\n")

print(f"\nMetrics summary saved to: {output_file}")