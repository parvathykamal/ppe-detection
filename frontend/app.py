"""
Flask Web Dashboard for Integrated Safety Surveillance System
Provides web interface for uploading and testing images/videos
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import json
from pathlib import Path
from datetime import datetime
import threading
import base64
import numpy as np
import sys

# Import the surveillance system
# Make sure the surveillance script is in the same directory or adjust the import
try:
    from surveillance import IntegratedSurveillanceSystem, Config
    SURVEILLANCE_AVAILABLE = True
except ImportError:
    SURVEILLANCE_AVAILABLE = False
    print("WARNING: surveillance_system.py not found. Dashboard will run in demo mode.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'safety-surveillance-2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'bmp', 'webp'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    Path(folder).mkdir(exist_ok=True)

# Global surveillance system instance
surveillance_system = None
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'error': None
}


def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]


def get_file_type(filename):
    """Determine if file is image or video"""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS['image']:
        return 'image'
    elif ext in ALLOWED_EXTENSIONS['video']:
        return 'video'
    return None


def initialize_system(ppe_model_path, fire_model_path):
    """Initialize the surveillance system"""
    global surveillance_system
    
    if not SURVEILLANCE_AVAILABLE:
        return False, "Surveillance system not available"
    
    try:
        surveillance_system = IntegratedSurveillanceSystem(
            ppe_model_path=ppe_model_path,
            fire_model_path=fire_model_path,
            device='cpu',
            enable_audio=False,  # Disable audio for web
            enable_email=False,
            enable_logging=True
        )
        return True, "System initialized successfully"
    except Exception as e:
        return False, f"Initialization failed: {str(e)}"


def process_file_task(input_path, output_path, file_type):
    """Background task for processing files"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['progress'] = 10
        processing_status['message'] = f'Processing {file_type}...'
        processing_status['error'] = None
        
        if file_type == 'image':
            surveillance_system.process_image(
                input_path,
                output_path=output_path,
                display=False
            )
            processing_status['progress'] = 100
        
        elif file_type == 'video':
            # For video, we'd need to track progress
            # This is a simplified version
            surveillance_system.process_video(
                input_path,
                output_path=output_path,
                display=False,
                skip_frames=0
            )
            processing_status['progress'] = 100
        
        processing_status['message'] = 'Processing complete!'
        
    except Exception as e:
        processing_status['error'] = str(e)
        processing_status['message'] = 'Processing failed'
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        processing_status['is_processing'] = False


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'system_ready': surveillance_system is not None,
        'allowed_image_formats': list(ALLOWED_EXTENSIONS['image']),
        'allowed_video_formats': list(ALLOWED_EXTENSIONS['video']),
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    })


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the surveillance system with model paths"""
    data = request.json
    ppe_model = data.get('ppe_model')
    fire_model = data.get('fire_model')
    
    if not ppe_model or not fire_model:
        return jsonify({'success': False, 'error': 'Model paths required'}), 400
    
    success, message = initialize_system(ppe_model, fire_model)
    
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Determine file type
    file_type = get_file_type(file.filename)
    if not file_type:
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    if not allowed_file(file.filename, file_type):
        return jsonify({'success': False, 'error': f'File type not allowed for {file_type}'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'filename': unique_filename,
        'file_type': file_type,
        'file_path': filepath
    })


@app.route('/api/process', methods=['POST'])
def process():
    """Process uploaded file"""
    global processing_status
    
    if surveillance_system is None:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400
    
    if processing_status['is_processing']:
        return jsonify({'success': False, 'error': 'Another file is being processed'}), 400
    
    data = request.json
    filename = data.get('filename')
    file_type = data.get('file_type')
    
    if not filename or not file_type:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(input_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    # Generate output filename
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_file_task,
        args=(input_path, output_path, file_type)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Processing started',
        'output_filename': output_filename
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processing status"""
    return jsonify(processing_status)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if surveillance_system is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    return jsonify({
        'frames_processed': surveillance_system.frame_count,
        'ppe_violations': surveillance_system.ppe_violations,
        'fire_detections': surveillance_system.fire_detections,
        'smoke_detections': surveillance_system.smoke_detections,
        'fall_detections': surveillance_system.fall_detections,
        'total_alerts': len(surveillance_system.alert_system.alert_history)
    })


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    if surveillance_system is None:
        return jsonify({'error': 'System not initialized'}), 400
    
    # Get last 50 alerts
    recent_alerts = list(surveillance_system.alert_system.alert_history)[-50:]
    
    return jsonify({
        'alerts': recent_alerts,
        'count': len(recent_alerts)
    })


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download processed file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True)


@app.route('/api/preview/<filename>')
def preview_file(filename):
    """Get preview/thumbnail of processed file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # For images, return directly
    file_type = get_file_type(filename)
    
    if file_type == 'image':
        return send_file(filepath)
    
    elif file_type == 'video':
        # Extract first frame as thumbnail
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
        else:
            return jsonify({'error': 'Could not extract frame'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/clear', methods=['POST'])
def clear_files():
    """Clear uploaded and processed files"""
    try:
        # Clear uploads
        for file in Path(app.config['UPLOAD_FOLDER']).glob('*'):
            file.unlink()
        
        # Clear outputs
        for file in Path(app.config['OUTPUT_FOLDER']).glob('*'):
            file.unlink()
        
        return jsonify({'success': True, 'message': 'Files cleared'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Safety Surveillance Web Dashboard')
    parser.add_argument('--ppe-model', type=str, help='Path to PPE detection model')
    parser.add_argument('--fire-model', type=str, help='Path to Fire detection model')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Auto-initialize if model paths provided
    if args.ppe_model and args.fire_model:
        print("\nInitializing surveillance system...")
        success, message = initialize_system(args.ppe_model, args.fire_model)
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            print("You can initialize later through the web interface.")
    else:
        print("\nModel paths not provided. Initialize through web interface.")
    
    print(f"\n{'='*60}")
    print("🚀 Starting Safety Surveillance Dashboard")
    print(f"{'='*60}")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Press CTRL+C to stop")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
