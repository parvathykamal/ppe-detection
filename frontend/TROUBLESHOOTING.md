# 🚨 TROUBLESHOOTING: Template Not Found Error

## Problem
You're seeing this error:
```
jinja2.exceptions.TemplateNotFound: index.html
```

## Root Cause
Flask can't find the `index.html` file because the `templates/` folder is not in the correct location relative to `app.py`.

## ✅ SOLUTION (Step-by-Step)

### Step 1: Extract All Files to Same Directory

**IMPORTANT**: All files must be in the same directory!

```
your-project-folder/
├── app.py                    ← Flask server
├── surveillance_system.py    ← Your detection code
├── requirements.txt
├── README.md
├── templates/                ← This folder MUST be here!
│   └── index.html
└── static/                   ← This folder MUST be here!
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

### Step 2: Navigate to the Correct Directory

Open your terminal/command prompt and navigate to where you extracted the files:

**Windows:**
```bash
cd C:\Users\nanda\OneDrive\Desktop\PPE\ppe_finl\ui
```

**Verify you're in the right place:**
```bash
# Windows
dir

# You should see:
# app.py
# templates/
# static/
# etc.
```

### Step 3: Verify File Structure

Run the verification script:
```bash
python verify_setup.py
```

This will check if all files are in the correct locations.

### Step 4: Run the Dashboard

```bash
python app.py --ppe-model C:\path\to\ppe_detection.pt --fire-model C:\path\to\fire_smoke_detection.pt
```

**Note:** On Windows, use forward slashes OR double backslashes in paths:
```bash
# Option 1: Forward slashes
python app.py --ppe-model C:/models/ppe_detection.pt --fire-model C:/models/fire_smoke_detection.pt

# Option 2: Double backslashes
python app.py --ppe-model C:\\models\\ppe_detection.pt --fire-model C:\\models\\fire_smoke_detection.pt
```

## 🔍 Quick Diagnostic

### Check 1: Are you in the right directory?
```bash
# Windows
dir app.py
dir templates
dir static

# If any of these fail, you're in the wrong directory!
```

### Check 2: Does templates/index.html exist?
```bash
# Windows
type templates\index.html

# If this fails, the file is missing
```

### Check 3: Does static/css/style.css exist?
```bash
# Windows
type static\css\style.css

# If this fails, the file is missing
```

## 📦 Re-Extract from Archive (If Needed)

If files are missing or in wrong locations:

1. **Create a new folder:**
   ```bash
   mkdir C:\surveillance-dashboard
   cd C:\surveillance-dashboard
   ```

2. **Extract the archive HERE:**
   ```bash
   tar -xzf surveillance-dashboard-complete.tar.gz
   ```
   
   Or on Windows with 7-Zip/WinRAR, extract to this folder.

3. **Verify structure:**
   ```bash
   dir
   # You should see:
   # app.py
   # templates/
   # static/
   # README.md
   # requirements.txt
   ```

4. **Copy your surveillance_system.py here:**
   ```bash
   copy C:\path\to\surveillance_system.py .
   ```

## 🎯 Complete Working Example

```bash
# 1. Create and navigate to project folder
mkdir C:\Projects\surveillance-dashboard
cd C:\Projects\surveillance-dashboard

# 2. Extract all files here
tar -xzf surveillance-dashboard-complete.tar.gz

# 3. Copy your detection system
copy C:\path\to\surveillance_system.py .

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify setup
python verify_setup.py

# 6. Run dashboard
python app.py --ppe-model C:/models/ppe.pt --fire-model C:/models/fire.pt

# 7. Open browser to http://localhost:5000
```

## 🐛 Still Not Working?

### Error: "Module not found: surveillance_system"
**Solution:** Copy your `surveillance_system.py` file to the same directory as `app.py`

### Error: "Template not found"
**Solution:** Make sure `templates/` folder is in the SAME directory as `app.py`

### Error: "Static files not loading"
**Solution:** Make sure `static/` folder is in the SAME directory as `app.py`

### Error: "ModuleNotFoundError: No module named 'flask'"
**Solution:** Install dependencies: `pip install -r requirements.txt`

## 📞 Need More Help?

1. Run `python verify_setup.py` to diagnose issues
2. Check that you're running `app.py` from the correct directory
3. Verify all files were extracted from the archive
4. Make sure templates/ and static/ folders are present

## ✅ Success Checklist

- [ ] All files extracted to same folder
- [ ] Running terminal from correct directory (where app.py is)
- [ ] templates/ folder exists with index.html inside
- [ ] static/ folder exists with css/ and js/ subfolders
- [ ] surveillance_system.py is in the same folder as app.py
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Model paths are correct and files exist

Once all boxes are checked, the dashboard should work! 🎉
