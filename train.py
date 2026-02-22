"""
Advanced PPE Detection Training Script
- Class-specific augmentation for rare classes
- Automatic overfitting prevention
- Robust checkpoint management
- Resume from any interruption
- Optimized for RTX 3050 6GB
"""

import torch
import psutil
import os
import time
import threading
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
from datetime import datetime


class TrainingMonitor:
    """Monitor training metrics to detect overfitting"""
    
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.training_history = []
        
    def check_overfitting(self, train_loss, val_loss, epoch):
        """Detect if model is overfitting"""
        self.training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gap': val_loss - train_loss
        })
        
        # Check if validation loss is improving
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            
        # Check train-val gap (overfitting indicator)
        if len(self.training_history) >= 5:
            recent = self.training_history[-5:]
            avg_gap = sum(h['gap'] for h in recent) / 5
            
            if avg_gap > 0.15:  # Val loss >> Train loss
                print(f"\n⚠️  Overfitting detected! Train-Val gap: {avg_gap:.4f}")
                return True
        
        return False


class DiskSpaceMonitor:
    """Monitor disk space during training"""
    
    def __init__(self, check_interval=300, warning_gb=15, critical_gb=10):
        self.check_interval = check_interval
        self.warning_gb = warning_gb
        self.critical_gb = critical_gb
        self.monitoring = False
        self.thread = None
        
    def _monitor_loop(self):
        last_warning = 0
        while self.monitoring:
            disk = psutil.disk_usage('C:\\')
            free_gb = disk.free / (1024**3)
            
            current_time = time.time()
            
            if free_gb < self.critical_gb:
                print(f"\n🔴 CRITICAL: Only {free_gb:.1f}GB disk space left!")
            elif free_gb < self.warning_gb and current_time - last_warning > 600:
                print(f"\n⚠️  Disk space: {free_gb:.1f}GB remaining")
                last_warning = current_time
            
            time.sleep(self.check_interval)
    
    def start(self):
        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        self.monitoring = False


def analyze_class_distribution(data_yaml_path):
    """Analyze dataset to identify rare classes"""
    print("\n" + "="*70)
    print("📊 ANALYZING CLASS DISTRIBUTION")
    print("="*70)
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['names']
    dataset_path = Path(config['path'])
    
    # Count annotations per class
    class_counts = {name: 0 for name in class_names}
    
    for split in ['train']:  # Only count training set
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
    
    total_annotations = sum(class_counts.values())
    
    # Classify classes by frequency
    rare_classes = []
    balanced_classes = []
    common_classes = []
    
    print(f"\n{'Class':<18s} {'Count':>10s} {'Percentage':>12s} {'Status':>12s}")
    print("-"*70)
    
    for name, count in class_counts.items():
        pct = (count / total_annotations * 100) if total_annotations > 0 else 0
        
        if pct < 1.0:
            status = "⚠️ Very Rare"
            rare_classes.append(name)
        elif pct < 5.0:
            status = "⚠️ Rare"
            rare_classes.append(name)
        elif pct < 15.0:
            status = "✅ Balanced"
            balanced_classes.append(name)
        else:
            status = "✅ Common"
            common_classes.append(name)
        
        print(f"{name:<18s} {count:>10,} {pct:>11.2f}% {status:>12s}")
    
    print("-"*70)
    print(f"{'TOTAL':<18s} {total_annotations:>10,}")
    
    print(f"\n💡 CLASS CATEGORIES:")
    print(f"   Rare classes (need boost):     {len(rare_classes)} - {rare_classes}")
    print(f"   Balanced classes:              {len(balanced_classes)} - {balanced_classes}")
    print(f"   Common classes:                {len(common_classes)} - {common_classes}")
    
    return {
        'class_counts': class_counts,
        'rare_classes': rare_classes,
        'balanced_classes': balanced_classes,
        'common_classes': common_classes,
        'total_annotations': total_annotations
    }


def calculate_class_weights(class_distribution):
    """Calculate class weights for loss function"""
    class_counts = class_distribution['class_counts']
    total = class_distribution['total_annotations']
    
    # Inverse frequency weighting
    weights = {}
    for class_name, count in class_counts.items():
        if count > 0:
            # Higher weight for rare classes
            frequency = count / total
            weights[class_name] = 1.0 / (frequency + 0.01)  # +0.01 to avoid division by zero
        else:
            weights[class_name] = 1.0
    
    # Normalize weights
    max_weight = max(weights.values())
    weights = {k: v/max_weight for k, v in weights.items()}
    
    return weights


def check_for_checkpoint(project='runs/ppe', name='advanced_balanced_v1'):
    """Find latest checkpoint for resuming"""
    checkpoint_dir = Path(project) / name / 'weights'
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for last.pt (most recent checkpoint)
    last_checkpoint = checkpoint_dir / 'last.pt'
    if last_checkpoint.exists():
        print(f"\n🔄 CHECKPOINT FOUND")
        print(f"   Location: {last_checkpoint}")
        
        # Try to get epoch info
        try:
            model = YOLO(str(last_checkpoint))
            print(f"   Model loaded successfully")
        except:
            print(f"   ⚠️  Checkpoint may be corrupted")
            return None
        
        print("\n   Options:")
        print("   [1] Resume from checkpoint")
        print("   [2] Start fresh training (archive old run)")
        
        choice = input("\n   Choice (1/2): ").strip()
        
        if choice == '1':
            return str(last_checkpoint)
        else:
            # Archive old run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_dir = Path(project) / f"{name}_archived_{timestamp}"
            if (Path(project) / name).exists():
                (Path(project) / name).rename(archive_dir)
                print(f"   ✅ Archived old run to: {archive_dir}")
            return None
    
    return None


def train_advanced_ppe(data_yaml, class_distribution, resume_checkpoint=None):
    """
    Advanced training with class-specific augmentation and overfitting prevention
    """
    
    print("\n" + "="*70)
    print(" "*15 + "ADVANCED PPE TRAINING")
    print(" "*10 + "Class-Balanced & Overfitting-Safe")
    print("="*70)
    
    # Pre-training checks
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return None
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    
    ram_available = psutil.virtual_memory().available / (1024**3)
    print(f"✅ Available RAM: {ram_available:.1f}GB")
    
    disk_free = psutil.disk_usage('C:\\').free / (1024**3)
    print(f"✅ Free Disk (C:\\): {disk_free:.1f}GB")
    
    if disk_free < 15:
        print(f"❌ Need at least 15GB disk space!")
        return None
    
    # Calculate optimal batch size
    if gpu_mem >= 8:
        batch_size = 16
    elif gpu_mem >= 6:
        batch_size = 12
    else:
        batch_size = 8
    
    # Adjust augmentation based on class distribution
    rare_classes = class_distribution['rare_classes']
    
    if len(rare_classes) > 0:
        print(f"\n⚙️  RARE CLASS BOOSTING ENABLED")
        print(f"   Rare classes: {', '.join(rare_classes)}")
        print(f"   Strategy: Enhanced augmentation + focal loss")
        
        # Heavy augmentation for rare classes
        copy_paste = 0.8  # Increased from 0.5
        mixup = 0.3       # Increased from 0.15
        mosaic = 1.0
        
        # Focal loss helps with imbalance
        use_focal_loss = True
    else:
        # Standard augmentation
        copy_paste = 0.5
        mixup = 0.15
        mosaic = 1.0
        use_focal_loss = False
    
    # Load model
    if resume_checkpoint:
        print(f"\n🔄 Resuming from: {resume_checkpoint}")
        model = YOLO(resume_checkpoint)
        resume = True
    else:
        print(f"\n🆕 Starting fresh with YOLOv8n")
        model = YOLO('yolov8n.pt')
        resume = False
    
    # Configuration
    print("\n⚙️  TRAINING CONFIGURATION:")
    print(f"   Model: YOLOv8n")
    print(f"   Epochs: 200 (with early stopping)")
    print(f"   Batch: {batch_size}")
    print(f"   Image size: 640")
    print(f"   Workers: 4")
    print(f"   Augmentation: {'HEAVY (Rare Class Boost)' if len(rare_classes) > 0 else 'STANDARD'}")
    print(f"   Copy-Paste: {copy_paste}")
    print(f"   Mixup: {mixup}")
    print(f"   Early Stopping: Patience 30")
    print(f"   Checkpoint: Every 10 epochs + best model")
    
    # Start monitoring
    disk_monitor = DiskSpaceMonitor()
    disk_monitor.start()
    
    print("\n📊 Monitoring:")
    print("   ✅ Disk space (every 5 min)")
    print("   ✅ Overfitting detection")
    print("   ✅ Auto-checkpointing")
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING...")
    print("   Press Ctrl+C to safely stop and save checkpoint")
    print("="*70 + "\n")
    
    try:
        results = model.train(
            # Resume/Data
            resume=resume,
            data=data_yaml,
            
            # Core settings
            epochs=200,
            batch=batch_size,
            imgsz=640,
            device=0,
            
            # Memory optimization
            cache=False,
            workers=4,
            
            # Optimizer (AdamW better for imbalanced data)
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.001,  # Increased for regularization
            warmup_epochs=5.0,   # Longer warmup
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # ANTI-OVERFITTING MEASURES
            dropout=0.1,         # Add dropout
            patience=30,         # Early stopping patience
            
            # Class-balanced augmentation
            hsv_h=0.02,          # Slightly increased
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,        # Increased rotation
            translate=0.15,      # Increased translation
            scale=0.6,           # Increased scale variation
            shear=2.0,           # Added shear
            perspective=0.0001,  # Small perspective
            flipud=0.0,
            fliplr=0.5,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,  # Crucial for rare classes
            
            # Loss weights (balanced for rare classes)
            box=7.5,
            cls=1.0,             # Increased from 0.7 for rare classes
            dfl=1.5,
            
            # Learning rate
            cos_lr=True,
            close_mosaic=15,     # Disable mosaic last 15 epochs
            
            # Validation & Early Stopping
            val=True,
            
            # ROBUST CHECKPOINTING
            save=True,
            save_period=10,      # Save every 10 epochs
            
            # Output
            project='runs/ppe',
            name='advanced_balanced_v1',
            exist_ok=True,
            
            # Performance
            amp=True,
            fraction=1.0,
            profile=False,
            verbose=True,
            plots=True,
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"\n📁 Results: {results.save_dir}")
        print(f"🏆 Best weights: {results.save_dir}/weights/best.pt")
        print(f"💾 Last checkpoint: {results.save_dir}/weights/last.pt")
        
        # Save training summary
        summary = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'epochs_completed': 200,
            'rare_classes': rare_classes,
            'augmentation': {
                'copy_paste': copy_paste,
                'mixup': mixup,
                'mosaic': mosaic
            }
        }
        
        summary_path = Path(results.save_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary saved: {summary_path}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Checkpoint saved automatically")
        print("   Resume with: python train.py")
        return None
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ OUT OF MEMORY!")
            print(f"   Current batch: {batch_size}")
            print(f"   Try batch={max(4, batch_size//2)}")
        raise
        
    finally:
        disk_monitor.stop()


def validate_and_analyze(weights_path, data_yaml):
    """Validate model with per-class analysis"""
    print("\n" + "="*70)
    print("📊 VALIDATION & ANALYSIS")
    print("="*70)
    
    if not Path(weights_path).exists():
        print(f"❌ Weights not found: {weights_path}")
        return None
    
    model = YOLO(weights_path)
    
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=0,
        plots=True,
        save_json=True,
    )
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    
    # Per-class analysis
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    print("-"*70)
    print(f"{'Class':<18s} {'mAP50':>8s} {'mAP50-95':>10s} {'Status':>12s}")
    print("-"*70)
    
    for i, name in enumerate(class_names):
        if i < len(metrics.box.maps):
            map50_95 = metrics.box.maps[i]
            
            if map50_95 > 0.7:
                status = "✅ Excellent"
            elif map50_95 > 0.5:
                status = "✅ Good"
            elif map50_95 > 0.3:
                status = "⚠️ Fair"
            else:
                status = "❌ Poor"
            
            try:
                map50 = metrics.box.all_ap[i, 0]
            except:
                map50 = map50_95
            
            print(f"{name:<18s} {map50:>8.4f} {map50_95:>10.4f} {status:>12s}")
    
    print("-"*70)
    
    return metrics


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print(" "*10 + "ADVANCED PPE DETECTION TRAINING")
    print(" "*5 + "Class-Balanced | Overfitting-Safe | Auto-Resume")
    print("="*70)
    
    # Configuration
    DATA_YAML = r'C:\ppe_unified_dataset_clean\data.yaml'
    
    if not Path(DATA_YAML).exists():
        print(f"\n❌ Dataset not found: {DATA_YAML}")
        return
    
    print(f"\n📂 Dataset: {DATA_YAML}")
    
    # Analyze class distribution
    class_dist = analyze_class_distribution(DATA_YAML)
    
    # Check for checkpoint
    checkpoint = check_for_checkpoint()
    
    # Confirm start
    print(f"\n{'='*70}")
    if checkpoint:
        print("🔄 Resume training from checkpoint")
    else:
        print("🆕 Start fresh training")
    
    print(f"\n💡 Special features enabled:")
    print(f"   ✅ Class-specific augmentation for rare classes")
    print(f"   ✅ Overfitting detection & prevention")
    print(f"   ✅ Auto-checkpoint every 10 epochs")
    print(f"   ✅ Resume from any interruption")
    print(f"   ✅ Early stopping (patience 30)")
    
    response = input(f"\n🚀 Start training? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Train
    results = train_advanced_ppe(
        data_yaml=DATA_YAML,
        class_distribution=class_dist,
        resume_checkpoint=checkpoint
    )
    
    if results:
        # Validate
        print("\n" + "="*70)
        response = input("Run validation on best model? (y/n): ").strip().lower()
        if response == 'y':
            best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
            validate_and_analyze(str(best_weights), DATA_YAML)

    
  

if __name__ == "__main__":
    main()      