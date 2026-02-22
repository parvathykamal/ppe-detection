# 🛡️ Safety Surveillance Command Center

A professional web-based dashboard for the Integrated Safety Surveillance System. Upload images or videos to detect PPE compliance, fire hazards, smoke, and safety violations in real-time.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Detection**: PPE compliance, fire, smoke, and fall detection
- **Multi-format Support**: Process images (PNG, JPG, BMP, WEBP) and videos (MP4, AVI, MOV, MKV)
- **Live Statistics**: Track violations, detections, and alerts in real-time
- **Alert System**: Comprehensive logging of all safety incidents
- **Visual Preview**: View processed results directly in the browser
- **Batch Processing**: Handle large video files with progress tracking

### 🎨 Modern Interface
- **Industrial Cyberpunk Design**: Striking visual aesthetic with animated backgrounds
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Real-time Updates**: Live statistics and processing status
- **Drag-and-Drop Upload**: Intuitive file handling
- **Smooth Animations**: Professional transitions and effects

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- YOLO detection models (PPE and Fire/Smoke)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Install dependencies**
```bash
pip install flask opencv-python numpy ultralytics werkzeug
```

2. **Organize your files**
```
project/
├── app.py                    # Flask server
├── surveillance_system.py    # Your detection system
├── templates/
│   └── index.html           # Dashboard HTML
├── static/
│   ├── css/
│   │   └── style.css        # Styles
│   └── js/
│       └── app.js           # Frontend logic
├── uploads/                 # Auto-created
└── outputs/                 # Auto-created
```

3. **Prepare your models**
- Place your trained YOLO models (.pt files) in an accessible location
- Note the paths to both PPE and Fire detection models

## 🎮 Usage

### Starting the Server

**Option 1: With pre-initialized models**
```bash
python app.py --ppe-model path/to/ppe_model.pt --fire-model path/to/fire_model.pt
```

**Option 2: Initialize later via web interface**
```bash
python app.py
```

**Additional options:**
```bash
python app.py --host 0.0.0.0 --port 8080 --debug
```

### Using the Dashboard

1. **Access**: Open browser to `http://localhost:5000`
2. **Initialize**: Enter model paths and click "INITIALIZE SYSTEM"
3. **Upload**: Drag and drop or browse to select file
4. **Process**: Click "START ANALYSIS" and watch progress
5. **Results**: View in Preview panel, check statistics, download

## 📊 Dashboard Components

- **System Control**: Model configuration and file upload
- **Live Statistics**: Real-time detection counts
- **Analysis Preview**: Visual output display
- **Recent Alerts**: Severity-coded safety alerts

## 🔧 Configuration

### Server Settings (app.py)

```python
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
```

### Detection Settings (surveillance_system.py)

```python
PPE_CONFIDENCE = 0.5
FIRE_CONFIDENCE = 0.4
PROXIMITY_THRESHOLD = 300
```

## 🎨 Customization

### Colors (static/css/style.css)

```css
:root {
    --primary: #00d9ff;
    --secondary: #ff3366;
    --accent: #ffed4e;
    --bg-dark: #0a0e27;
}
```

## 🐛 Troubleshooting

- **System not initialized**: Check model paths exist
- **Upload failed**: Verify file size and format
- **Processing stalled**: Check server console for errors
- **Models not loading**: Install ultralytics: `pip install ultralytics`

## 📝 API Endpoints

- `GET /` - Dashboard
- `POST /api/initialize` - Initialize models
- `POST /api/upload` - Upload file
- `POST /api/process` - Process file
- `GET /api/status` - Processing status
- `GET /api/stats` - Statistics
- `GET /api/alerts` - Recent alerts
- `GET /api/download/<filename>` - Download result

## 🔒 Security (Production)

1. Change secret key in app.py
2. Add authentication
3. Use HTTPS
4. Set file size limits
5. Implement rate limiting

## 📈 Performance Tips

- Use GPU: `--device cuda:0`
- Skip frames for videos
- Use lighter YOLO models
- Enable production WSGI server

---

**Built for Safety Professionals** 🛡️
