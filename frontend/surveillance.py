"""
Integrated Safety Surveillance System - CLEAN OUTPUT VERSION
Combines PPE Detection + Fire/Smoke Detection with Threading
Dashboard overlay removed for clean web UI output

Features:
- Parallel inference using threading
- Fixed PPE compliance checking with enhanced association logic
- Improved label visibility
- Fire and smoke detection
- Multi-level alert system
- Comprehensive logging
- Optional dashboard overlay (disabled by default for web UI)

Author: AI Assistant
Date: 2026
Modified: Dashboard overlay removed for cleaner web UI display
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import threading
from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple, Set, Optional
import sys

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ERROR: ultralytics not found. Install with: pip install ultralytics")
    sys.exit(1)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not found. Audio alerts disabled. Install with: pip install pygame")


# ==================== CONFIGURATION ====================
class Config:
    """System configuration parameters"""
    
    # PPE Detection Configuration
    REQUIRED_PPE = ['Hardhat', 'Safety_Vest', 'Safety_Boots']
    PPE_CLASSES = ['Hardhat', 'Person', 'Safety_Boots', 'Safety_Gloves', 'Safety_Mask', 'Safety_Vest']
    
    # Fire/Smoke Detection Configuration
    HAZARD_CLASSES = ['fire', 'smoke']
    
    # Alert Severity Levels
    SEVERITY_CRITICAL = 3
    SEVERITY_HIGH = 2
    SEVERITY_MEDIUM = 1
    SEVERITY_LOW = 0
    
    # Color Schemes (BGR format)
    COLOR_SAFE = (0, 255, 0)
    COLOR_WARNING = (0, 165, 255)
    COLOR_DANGER = (0, 0, 255)
    COLOR_CRITICAL = (255, 0, 255)
    COLOR_FIRE = (0, 0, 255)
    COLOR_SMOKE = (128, 128, 128)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    
    # Detection Thresholds
    PPE_CONFIDENCE = 0.5
    FIRE_CONFIDENCE = 0.4
    IOU_THRESHOLD = 0.1
    PROXIMITY_THRESHOLD = 300
    
    # Alert Thresholds
    FIRE_SIZE_CRITICAL = 50000
    SMOKE_SIZE_WARNING = 30000
    ALERT_COOLDOWN = 5.0
    
    # System Performance
    MAX_FRAME_QUEUE = 5
    STATS_WINDOW = 100


# ==================== DETECTION RESULT CLASSES ====================
class Detection:
    """Single detection result"""
    def __init__(self, bbox, confidence, class_name, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.class_id = class_id
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


class PersonCompliance:
    """Person with PPE compliance status"""
    def __init__(self, person_id, bbox, detected_ppe, missing_ppe, status):
        self.person_id = person_id
        self.bbox = bbox
        self.detected_ppe = detected_ppe
        self.missing_ppe = missing_ppe
        self.status = status
        self.severity = self._calculate_severity()
    
    def _calculate_severity(self):
        if self.status == 'No_PPE':
            return Config.SEVERITY_HIGH
        elif self.status == 'Partial_PPE':
            return Config.SEVERITY_MEDIUM
        return Config.SEVERITY_LOW


class HazardDetection:
    """Fire/Smoke hazard detection"""
    def __init__(self, bbox, confidence, hazard_type):
        self.bbox = bbox
        self.confidence = confidence
        self.hazard_type = hazard_type
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.severity = self._calculate_severity()
    
    def _calculate_severity(self):
        if self.hazard_type == 'fire':
            if self.area > Config.FIRE_SIZE_CRITICAL:
                return Config.SEVERITY_CRITICAL
            return Config.SEVERITY_HIGH
        else:
            if self.area > Config.SMOKE_SIZE_WARNING:
                return Config.SEVERITY_HIGH
            return Config.SEVERITY_MEDIUM


# ==================== ALERT SYSTEM ====================
class AlertSystem:
    """Multi-level alert management system"""
    
    def __init__(self, enable_audio=True, enable_email=False, enable_logging=True):
        self.enable_audio = enable_audio and PYGAME_AVAILABLE
        self.enable_email = enable_email
        self.enable_logging = enable_logging
        
        self.last_alert_time = {}
        self.alert_history = deque(maxlen=1000)
        
        if self.enable_audio:
            self._init_audio()
        
        if self.enable_logging:
            self._init_logging()
    
    def _init_audio(self):
        """Initialize audio alerts"""
        try:
            pygame.mixer.init()
            self.sounds = {
                'critical': self._generate_beep(880, 0.5, 3),
                'high': self._generate_beep(660, 0.4, 2),
                'medium': self._generate_beep(440, 0.3, 1),
            }
            print("✓ Audio alerts enabled")
        except Exception as e:
            print(f"⚠ Audio initialization failed: {e}")
            self.enable_audio = False
    
    def _generate_beep(self, frequency, duration, count):
        """Generate beep sound"""
        sample_rate = 22050
        n_samples = int(round(duration * sample_rate))
        
        t = np.arange(n_samples) / sample_rate
        beep = np.sin(2 * np.pi * frequency * t)
        
        gap_samples = int(0.1 * sample_rate)
        full_sound = []
        
        for i in range(count):
            full_sound.extend(beep)
            if i < count - 1:
                full_sound.extend(np.zeros(gap_samples))
        
        full_sound = np.array(full_sound)
        full_sound = (full_sound * 32767).astype(np.int16)
        stereo = np.column_stack((full_sound, full_sound))
        return pygame.sndarray.make_sound(stereo)
    
    def _init_logging(self):
        """Initialize logging system"""
        self.log_dir = Path("surveillance_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"alerts_{timestamp}.json"
        self.alert_images_dir = self.log_dir / "alert_snapshots"
        self.alert_images_dir.mkdir(exist_ok=True)
        
        print(f"✓ Logging enabled: {self.log_file}")
    
    def trigger_alert(self, alert_type, severity, message, frame=None, frame_number=None):
        """Trigger an alert with cooldown management"""
        current_time = datetime.now()
        
        cooldown_key = f"{alert_type}_{severity}"
        if cooldown_key in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[cooldown_key]).total_seconds()
            if time_diff < Config.ALERT_COOLDOWN:
                return
        
        self.last_alert_time[cooldown_key] = current_time
        
        if self.enable_audio:
            if severity == Config.SEVERITY_CRITICAL:
                self.sounds['critical'].play()
            elif severity == Config.SEVERITY_HIGH:
                self.sounds['high'].play()
            elif severity == Config.SEVERITY_MEDIUM:
                self.sounds['medium'].play()
        
        snapshot_path = None
        if frame is not None and self.enable_logging:
            snapshot_path = self.alert_images_dir / f"alert_{alert_type}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(snapshot_path), frame)
        
        alert_record = {
            'timestamp': current_time.isoformat(),
            'frame_number': frame_number,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'snapshot': str(snapshot_path) if snapshot_path else None
        }
        
        self.alert_history.append(alert_record)
        
        severity_labels = ['INFO', 'WARNING', 'HIGH', 'CRITICAL']
        print(f"\n{'='*60}")
        print(f"🚨 ALERT [{severity_labels[severity]}]: {message}")
        print(f"   Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
    
    def save_log(self):
        """Save alert log to file"""
        if not self.enable_logging:
            return
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    'total_alerts': len(self.alert_history),
                    'alerts': list(self.alert_history)
                }, f, indent=2)
            print(f"\n✓ Alert log saved: {self.log_file}")
        except Exception as e:
            print(f"✗ Error saving log: {e}")


# ==================== INTEGRATED SURVEILLANCE SYSTEM ====================
class IntegratedSurveillanceSystem:
    """Main surveillance system - Clean output version"""
    
    def __init__(self, ppe_model_path, fire_model_path, device='cpu',
                 enable_audio=True, enable_email=False, enable_logging=True):
        """Initialize the surveillance system"""
        print("\n" + "="*60)
        print("INTEGRATED SAFETY SURVEILLANCE SYSTEM")
        print("="*60)
        
        self.device = device
        self.show_dashboard = False  # Disabled by default for clean web UI output
        
        print("\n📦 Loading detection models...")
        self.ppe_model = self._load_model(ppe_model_path, "PPE Detection")
        self.fire_model = self._load_model(fire_model_path, "Fire/Smoke Detection")
        
        print("\n🔔 Initializing alert system...")
        self.alert_system = AlertSystem(enable_audio, enable_email, enable_logging)
        
        self.frame_count = 0
        self.ppe_violations = 0
        self.fire_detections = 0
        self.smoke_detections = 0
        self.fall_detections = 0
        
        self.fps_tracker = deque(maxlen=Config.STATS_WINDOW)
        self.last_frame_time = time.time()
        self.results_lock = threading.Lock()
        
        print("\n✅ System initialized successfully!")
        print("="*60 + "\n")
    
    def _load_model(self, model_path, model_name):
        """Load a YOLO model"""
        try:
            print(f"   Loading {model_name} from: {model_path}")
            model = YOLO(model_path)
            print(f"   ✓ {model_name} loaded | Classes: {list(model.names.values())}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name}: {e}")
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_distance(self, box1, box2):
        """Calculate Euclidean distance between box centers"""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        
        return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    
    def _associate_ppe_with_person(self, person_box, ppe_detections):
        """Enhanced PPE association with special handling for different PPE types"""
        associated_ppe = set()
        
        person_x_min, person_y_min, person_x_max, person_y_max = person_box
        person_center_x = (person_x_min + person_x_max) / 2
        person_center_y = (person_y_min + person_y_max) / 2
        person_width = person_x_max - person_x_min
        person_height = person_y_max - person_y_min
        
        for ppe_det in ppe_detections:
            ppe_x_min, ppe_y_min, ppe_x_max, ppe_y_max = ppe_det.bbox
            ppe_center_x = (ppe_x_min + ppe_x_max) / 2
            ppe_center_y = (ppe_y_min + ppe_y_max) / 2
            
            iou = self._calculate_iou(person_box, ppe_det.bbox)
            distance = self._calculate_distance(person_box, ppe_det.bbox)
            
            # SPECIAL CASE 1: HARDHAT (typically ABOVE person's head)
            if ppe_det.class_name == 'Hardhat':
                horizontal_distance = abs(ppe_center_x - person_center_x)
                if horizontal_distance < person_width * 0.7:
                    if ppe_center_y <= person_y_min + person_height * 0.3:
                        associated_ppe.add(ppe_det.class_name)
                        continue
            
            # SPECIAL CASE 2: BOOTS (typically BELOW person, at feet)
            if ppe_det.class_name == 'Safety_Boots':
                horizontal_distance = abs(ppe_center_x - person_center_x)
                if horizontal_distance < person_width * 0.9:
                    if ppe_center_y >= person_y_min + person_height * 0.6:
                        associated_ppe.add(ppe_det.class_name)
                        continue
                    elif ppe_y_min - person_y_max < 100:
                        associated_ppe.add(ppe_det.class_name)
                        continue
            
            # SPECIAL CASE 3: VEST (should overlap significantly with person)
            if ppe_det.class_name == 'Safety_Vest':
                if iou > 0.2 or distance < Config.PROXIMITY_THRESHOLD * 0.6:
                    associated_ppe.add(ppe_det.class_name)
                    continue
            
            # SPECIAL CASE 4: MASK (on face, near top of person)
            if ppe_det.class_name == 'Safety_Mask':
                if iou > 0.05 or (distance < Config.PROXIMITY_THRESHOLD * 0.4 and 
                                  ppe_center_y < person_center_y):
                    associated_ppe.add(ppe_det.class_name)
                    continue
            
            # SPECIAL CASE 5: GLOVES (at hands, typically overlapping)
            if ppe_det.class_name == 'Safety_Gloves':
                if iou > 0.03 or distance < Config.PROXIMITY_THRESHOLD * 0.5:
                    associated_ppe.add(ppe_det.class_name)
                    continue
            
            # DEFAULT ASSOCIATION (fallback)
            if iou > Config.IOU_THRESHOLD or distance < Config.PROXIMITY_THRESHOLD:
                associated_ppe.add(ppe_det.class_name)
        
        return associated_ppe
    
    def _check_ppe_compliance(self, detected_ppe):
        """Check PPE compliance and return status"""
        required_set = set(Config.REQUIRED_PPE)
        missing_items = list(required_set - detected_ppe)
        
        if not missing_items:
            return 'Fully_Equipped', []
        elif detected_ppe:
            return 'Partial_PPE', missing_items
        else:
            return 'No_PPE', missing_items
    
    def detect_parallel(self, frame):
        """Run parallel detection on both models"""
        results = {}
        timings = {}
        
        def detect_ppe():
            start = time.time()
            ppe_result = self.ppe_model(
                frame, 
                conf=Config.PPE_CONFIDENCE,
                iou=Config.IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )[0]
            timings['ppe'] = time.time() - start
            
            with self.results_lock:
                results['ppe'] = ppe_result
        
        def detect_fire():
            start = time.time()
            fire_result = self.fire_model(
                frame,
                conf=Config.FIRE_CONFIDENCE,
                iou=Config.IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )[0]
            timings['fire'] = time.time() - start
            
            with self.results_lock:
                results['fire'] = fire_result
        
        t1 = threading.Thread(target=detect_ppe, name="PPE-Thread", daemon=True)
        t2 = threading.Thread(target=detect_fire, name="Fire-Thread", daemon=True)
        
        overall_start = time.time()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        timings['total'] = time.time() - overall_start
        
        return results, timings
    
    def process_frame(self, frame):
        """Process a single frame with both detection models"""
        self.frame_count += 1
        annotated_frame = frame.copy()
        
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.fps_tracker.append(fps)
        self.last_frame_time = current_time
        
        results, timings = self.detect_parallel(frame)
        
        # Parse PPE detections
        persons = []
        ppe_items = []
        fall_detected = False
        
        for box in results['ppe'].boxes:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.ppe_model.names[class_id]
            
            detection = Detection(bbox, confidence, class_name, class_id)
            
            if class_name == 'Person':
                persons.append(detection)
            elif class_name in Config.REQUIRED_PPE or class_name in ['Safety_Gloves', 'Safety_Mask']:
                ppe_items.append(detection)
            elif class_name == 'Fall_Detected':
                fall_detected = True
                self.fall_detections += 1
                self._draw_detection(annotated_frame, bbox, 'FALL DETECTED ⚠️', 
                                   Config.COLOR_CRITICAL, confidence, thickness=3)
        
        # Parse Fire/Smoke detections
        hazards = []
        for box in results['fire'].boxes:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.fire_model.names[class_id]
            
            if class_name in Config.HAZARD_CLASSES:
                hazard = HazardDetection(bbox, confidence, class_name)
                hazards.append(hazard)
                
                if class_name == 'fire':
                    self.fire_detections += 1
                else:
                    self.smoke_detections += 1
        
        # Process PPE compliance for each person
        person_compliance_list = []
        for idx, person in enumerate(persons):
            detected_ppe = self._associate_ppe_with_person(person.bbox, ppe_items)
            status, missing_items = self._check_ppe_compliance(detected_ppe)
            
            person_compliance = PersonCompliance(
                person_id=idx,
                bbox=person.bbox,
                detected_ppe=detected_ppe,
                missing_ppe=missing_items,
                status=status
            )
            person_compliance_list.append(person_compliance)
            
            if status != 'Fully_Equipped':
                self.ppe_violations += 1
        
        # Draw all detections
        self._draw_all_detections(annotated_frame, person_compliance_list, ppe_items, hazards)
        
        # Handle alerts
        self._handle_alerts(annotated_frame, person_compliance_list, hazards, fall_detected)
        
        # Dashboard overlay (OPTIONAL - disabled by default for web UI)
        if self.show_dashboard:
            avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
            self._draw_dashboard(annotated_frame, len(persons), len(hazards), 
                               fall_detected, avg_fps, timings)
        
        return annotated_frame
    
    def _draw_detection(self, frame, bbox, label, color, confidence, thickness=2):
        """Draw detection with smart label positioning"""
        height, width = frame.shape[:2]
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Clamp to frame bounds
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(0, min(x_max, width - 1))
        y_max = max(0, min(y_max, height - 1))
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Prepare label
        label_text = f"{label} {confidence:.2f}"
        font_scale = 0.6
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Smart label positioning
        padding = 8
        label_height = text_height + baseline + padding * 2
        
        if y_min > label_height + 10:
            label_y_top = y_min - label_height
            label_y_bottom = y_min - 5
            text_y = label_y_bottom - baseline - 5
        else:
            label_y_top = y_min + 5
            label_y_bottom = y_min + label_height
            text_y = label_y_bottom - baseline - padding
        
        label_x_right = min(x_min + text_width + padding * 2, width - 5)
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x_min, label_y_top),
            (label_x_right, label_y_bottom),
            color,
            -1
        )
        
        cv2.rectangle(
            frame,
            (x_min, label_y_top),
            (label_x_right, label_y_bottom),
            (255, 255, 255),
            1
        )
        
        # Draw text with shadow
        cv2.putText(
            frame, label_text,
            (x_min + padding + 1, text_y + 1),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness + 1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame, label_text,
            (x_min + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
    
    def _draw_all_detections(self, frame, person_compliance_list, ppe_items, hazards):
        """Draw all detections on frame"""
        
        # Draw persons with compliance status
        for person_comp in person_compliance_list:
            if person_comp.status == 'Fully_Equipped':
                color = Config.COLOR_SAFE
                label = "✓ COMPLIANT"
            elif person_comp.status == 'Partial_PPE':
                color = Config.COLOR_WARNING
                missing_str = ", ".join(person_comp.missing_ppe[:2])
                if len(person_comp.missing_ppe) > 2:
                    missing_str += "..."
                label = f"⚠ Missing: {missing_str}"
            else:
                color = Config.COLOR_DANGER
                label = "✗ NO PPE"
            
            self._draw_detection(frame, person_comp.bbox, label, color, 1.0, thickness=3)
        
        # Draw hazards
        for idx, hazard in enumerate(hazards):
            if hazard.hazard_type == 'fire':
                color = Config.COLOR_FIRE
                label = "🔥 FIRE"
                thickness = 4
            else:
                color = (200, 200, 200)
                label = "💨 SMOKE"
                thickness = 3
            
            self._draw_detection(frame, hazard.bbox, label, color, 
                               hazard.confidence, thickness=thickness)
    
    def _handle_alerts(self, frame, person_compliance_list, hazards, fall_detected):
        """Handle all alert triggers"""
        
        for hazard in hazards:
            if hazard.hazard_type == 'fire':
                message = f"FIRE DETECTED! Area: {hazard.area:.0f} px² | Confidence: {hazard.confidence:.1%}"
                self.alert_system.trigger_alert(
                    'fire', hazard.severity, message, frame, self.frame_count
                )
            else:
                message = f"Smoke detected. Area: {hazard.area:.0f} px² | Confidence: {hazard.confidence:.1%}"
                self.alert_system.trigger_alert(
                    'smoke', hazard.severity, message, frame, self.frame_count
                )
        
        for person_comp in person_compliance_list:
            if person_comp.status == 'No_PPE':
                message = f"CRITICAL: Person #{person_comp.person_id} has NO PPE equipment!"
                self.alert_system.trigger_alert(
                    'ppe_violation', person_comp.severity, message, frame, self.frame_count
                )
            elif person_comp.status == 'Partial_PPE':
                missing_str = ", ".join(person_comp.missing_ppe)
                message = f"WARNING: Person #{person_comp.person_id} missing PPE: {missing_str}"
                self.alert_system.trigger_alert(
                    'ppe_violation', person_comp.severity, message, None, self.frame_count
                )
        
        if fall_detected:
            message = "FALL DETECTED! Immediate assistance required!"
            self.alert_system.trigger_alert(
                'fall', Config.SEVERITY_CRITICAL, message, frame, self.frame_count
            )
    
    def _draw_dashboard(self, frame, num_persons, num_hazards, fall_detected, fps, timings):
        """Draw real-time dashboard overlay (optional)"""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        dashboard_height = 180
        cv2.rectangle(overlay, (10, 10), (450, dashboard_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "SURVEILLANCE DASHBOARD", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_WHITE, 2)
        
        stats = [
            f"Frame: {self.frame_count} | FPS: {fps:.1f}",
            f"Persons: {num_persons} | Hazards: {num_hazards}",
            f"PPE Violations: {self.ppe_violations}",
            f"Fire: {self.fire_detections} | Smoke: {self.smoke_detections}",
            f"Inference: PPE {timings['ppe']*1000:.1f}ms | Fire {timings['fire']*1000:.1f}ms"
        ]
        
        y_offset = 60
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 1)
            y_offset += 22
        
        if num_hazards > 0:
            status_text = "STATUS: CRITICAL"
            status_color = Config.COLOR_DANGER
        elif self.ppe_violations > 0:
            status_text = "STATUS: WARNING"
            status_color = Config.COLOR_WARNING
        else:
            status_text = "STATUS: ALL CLEAR"
            status_color = Config.COLOR_SAFE
        
        cv2.rectangle(frame, (20, dashboard_height - 35), (430, dashboard_height - 10),
                     status_color, -1)
        cv2.putText(frame, status_text, (120, dashboard_height - 17),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_BLACK, 2)
        
        if fall_detected:
            banner_height = 60
            cv2.rectangle(frame, (0, height - banner_height), (width, height),
                         Config.COLOR_CRITICAL, -1)
            cv2.putText(frame, "⚠️  FALL DETECTED - IMMEDIATE ASSISTANCE REQUIRED  ⚠️",
                       (50, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       Config.COLOR_WHITE, 3)
    
    def process_image(self, image_path, output_path=None, display=True):
        """Process a single image"""
        print(f"\nProcessing image: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        
        result_frame = self.process_frame(frame)
        
        if output_path:
            cv2.imwrite(str(output_path), result_frame)
            print(f"Output saved: {output_path}")
        
        if display:
            cv2.imshow('Integrated Surveillance System - Press any key', result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        self.alert_system.save_log()
        
        return result_frame
    
    def process_video(self, video_path, output_path=None, display=True, skip_frames=0):
        """Process video file"""
        print(f"\n{'='*60}")
        print(f"Processing video: {video_path}")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS | {total_frames} frames")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Output: {output_path}\n")
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue
                
                result_frame = self.process_frame(frame)
                
                if writer:
                    writer.write(result_frame)
                
                if display:
                    cv2.imshow('Integrated Surveillance System - Press Q to quit', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠ Processing interrupted by user")
                        break
                
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_avg = frame_idx / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_idx) / fps_avg if fps_avg > 0 else 0
                    
                    print(f"Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {fps_avg:.1f} | ETA: {eta:.0f}s", end='\r')
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            self.alert_system.save_log()
            
            elapsed = time.time() - start_time
            print(f"\n\n{'='*60}")
            print("PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Frames processed: {frame_idx}")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Average FPS: {frame_idx/elapsed:.1f}")
            print(f"\nDetection Summary:")
            print(f"  - PPE Violations: {self.ppe_violations}")
            print(f"  - Fire detections: {self.fire_detections}")
            print(f"  - Smoke detections: {self.smoke_detections}")
            print(f"  - Fall detections: {self.fall_detections}")
            print(f"  - Total alerts: {len(self.alert_system.alert_history)}")
            print(f"{'='*60}\n")
    
    def process_realtime(self, camera_id=0):
        """Process real-time camera feed"""
        print(f"\n{'='*60}")
        print("STARTING REAL-TIME SURVEILLANCE")
        print(f"{'='*60}\n")
        print("Press 'Q' to quit")
        print("Press 'S' to save snapshot\n")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                result_frame = self.process_frame(frame)
                
                cv2.imshow('Real-time Surveillance - Q:Quit | S:Snapshot', result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopping surveillance...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(snapshot_path, result_frame)
                    print(f"Snapshot saved: {snapshot_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.alert_system.save_log()


# ==================== MAIN ====================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrated Safety Surveillance System - Clean Output Version',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--ppe-model', type=str, required=True,
                       help='Path to PPE detection model (.pt)')
    parser.add_argument('--fire-model', type=str, required=True,
                       help='Path to Fire/Smoke detection model (.pt)')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--image', type=str, help='Path to input image')
    input_group.add_argument('--camera', type=int, help='Camera device ID')
    
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cuda:0, cpu, etc.)')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Process every Nth frame')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio alerts')
    parser.add_argument('--enable-email', action='store_true',
                       help='Enable email alerts')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable logging to file')
    parser.add_argument('--show-dashboard', action='store_true',
                       help='Show dashboard overlay on output')
    
    args = parser.parse_args()
    
    if not Path(args.ppe_model).exists():
        print(f"Error: PPE model not found: {args.ppe_model}")
        sys.exit(1)
    if not Path(args.fire_model).exists():
        print(f"Error: Fire model not found: {args.fire_model}")
        sys.exit(1)
    
    try:
        system = IntegratedSurveillanceSystem(
            ppe_model_path=args.ppe_model,
            fire_model_path=args.fire_model,
            device=args.device,
            enable_audio=not args.no_audio,
            enable_email=args.enable_email,
            enable_logging=not args.no_logging
        )
        
        # Enable dashboard if requested
        if args.show_dashboard:
            system.show_dashboard = True
        
        if args.video:
            if not Path(args.video).exists():
                print(f"Error: Video not found: {args.video}")
                sys.exit(1)
            system.process_video(
                args.video,
                output_path=args.output,
                display=not args.no_display,
                skip_frames=args.skip_frames
            )
        elif args.image:
            if not Path(args.image).exists():
                print(f"Error: Image not found: {args.image}")
                sys.exit(1)
            system.process_image(
                args.image,
                output_path=args.output,
                display=not args.no_display
            )
        else:
            system.process_realtime(camera_id=args.camera)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()