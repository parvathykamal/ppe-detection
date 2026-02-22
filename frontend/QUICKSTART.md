# 🚀 QUICK START GUIDE
## Safety Surveillance Command Center Dashboard

### Step 1: Setup (2 minutes)

1. **Extract files**
   ```bash
   tar -xzf surveillance-dashboard.tar.gz
   cd surveillance-dashboard/
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your surveillance_system.py**
   - Copy your existing surveillance_system.py file to this directory
   - Make sure it has the IntegratedSurveillanceSystem class

### Step 2: Launch (30 seconds)

**Quick launch:**
```bash
python app.py
```

**With pre-loaded models:**
```bash
python app.py --ppe-model /path/to/ppe_best.pt --fire-model /path/to/fire_best.pt
```

**Custom port:**
```bash
python app.py --port 8080
```

### Step 3: Use (1 minute)

1. Open browser: `http://localhost:5000`
2. Enter model paths (if not pre-loaded)
3. Click "INITIALIZE SYSTEM"
4. Drag & drop an image or video
5. Click "START ANALYSIS"
6. View results!

### What You'll See

```
┌─────────────────────────────────────────────────┐
│  🛡️ SURVEILLANCE COMMAND CENTER                │
├─────────────────────────────────────────────────┤
│                                                 │
│  System Control          Live Statistics        │
│  ┌──────────────┐       ┌─────────────────┐   │
│  │ Upload Files │       │ PPE Violations  │   │
│  │ Process      │       │ Fire Detected   │   │
│  └──────────────┘       │ Smoke Detected  │   │
│                         └─────────────────┘   │
│                                                 │
│  Analysis Preview        Recent Alerts          │
│  ┌──────────────┐       ┌─────────────────┐   │
│  │  [IMAGE]     │       │ ⚠️ Alert 1      │   │
│  │  or VIDEO    │       │ 🚨 Alert 2      │   │
│  └──────────────┘       └─────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Features at a Glance

✅ **Drag & Drop Upload** - Just drop files onto the upload zone
✅ **Real-time Processing** - Watch progress bars in action
✅ **Live Statistics** - See detections count up automatically
✅ **Visual Results** - Preview processed images/videos instantly
✅ **Download Results** - One-click download of processed files
✅ **Alert History** - Track all safety violations
✅ **Responsive Design** - Works on desktop, tablet, mobile

### Supported Files

**Images:** PNG, JPG, JPEG, BMP, WEBP
**Videos:** MP4, AVI, MOV, MKV, WMV
**Max Size:** 500MB (configurable)

### Troubleshooting

❌ **"System not initialized"**
   → Enter correct model paths and click Initialize

❌ **"File upload failed"**
   → Check file size is under 500MB
   → Verify file format is supported

❌ **"Processing stalled"**
   → Check console for errors
   → Large videos take time - be patient

❌ **Models not loading**
   → Install ultralytics: `pip install ultralytics`
   → Verify .pt model files exist

### Performance Tips

🚀 For faster processing:
- Use GPU: Add `--device cuda:0` flag in surveillance_system.py
- Process videos at lower resolution
- Use lighter YOLO models (YOLOv8n instead of YOLOv8x)
- Skip frames: Modify `skip_frames` parameter

### File Structure

```
surveillance-dashboard/
├── app.py                 # Flask server (don't modify)
├── surveillance_system.py # Your detection code (add this!)
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── templates/
│   └── index.html        # Dashboard HTML
└── static/
    ├── css/
    │   └── style.css     # Styles
    └── js/
        └── app.js        # Frontend logic
```

### Next Steps

1. ✅ Get dashboard running
2. 📊 Process your first file
3. 🎨 Customize colors in static/css/style.css
4. 🔧 Adjust detection thresholds in surveillance_system.py
5. 🚀 Deploy to production server

### Need Help?

1. Check README.md for full documentation
2. Review server console for error messages
3. Verify all dependencies are installed
4. Test with small files first

---

**Ready to protect workers and save lives!** 🛡️

*For detailed documentation, see README.md*
