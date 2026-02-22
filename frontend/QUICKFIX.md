# 🚀 QUICK FIX - Template Not Found Error

## THE PROBLEM
Your error: `jinja2.exceptions.TemplateNotFound: index.html`

**Root cause:** The `templates` and `static` folders are not in the same directory as `app.py`

## THE FIX (3 Simple Steps)

### Step 1: Create the folder structure
In the SAME folder as your `app.py`, create these folders:

```
📁 Where app.py is located/
   ├── 📄 app.py
   ├── 📄 surveillance_system.py (your file)
   ├── 📁 templates/
   └── 📁 static/
       ├── 📁 css/
       └── 📁 js/
```

**Windows commands:**
```cmd
cd C:\Users\nanda\OneDrive\Desktop\PPE\ppe_finl\ui
mkdir templates
mkdir static\css
mkdir static\js
```

### Step 2: Put the files in the right places

Download these files and place them:

1. **index.html** → Put in `templates/` folder
2. **style.css** → Put in `static/css/` folder  
3. **app.js** → Put in `static/js/` folder

**Result should be:**
```
📁 ui/
   ├── 📄 app.py
   ├── 📄 surveillance_system.py
   ├── 📁 templates/
   │   └── 📄 index.html          ← HERE!
   └── 📁 static/
       ├── 📁 css/
       │   └── 📄 style.css        ← HERE!
       └── 📁 js/
           └── 📄 app.js           ← HERE!
```

### Step 3: Run from the correct directory

**IMPORTANT:** Run the command from where `app.py` is located!

```cmd
cd C:\Users\nanda\OneDrive\Desktop\PPE\ppe_finl\ui
python app.py --ppe-model C:/models/ppe_detection.pt --fire-model C:/models/fire_smoke_detection.pt
```

## ✅ VERIFY IT'S FIXED

Run this in your terminal:
```cmd
# Should show the file exists
type templates\index.html

# Should show the file exists  
type static\css\style.css

# Should show the file exists
type static\js\app.js
```

If all three commands work, you're good to go!

## 🎯 Alternative: Use the Complete Archive

Or simply extract everything fresh:

1. Download: `dashboard-FIXED-complete.tar.gz`
2. Extract to a NEW folder (e.g., `C:\surveillance-dashboard\`)
3. Copy your `surveillance_system.py` to that folder
4. Run from that folder

```cmd
cd C:\surveillance-dashboard
python app.py --ppe-model C:/models/ppe.pt --fire-model C:/models/fire.pt
```

## 💡 Pro Tip: Use the Verification Script

Run `verify_setup.py` to automatically check if everything is in the right place:

```cmd
python verify_setup.py
```

It will tell you exactly what's missing!

---

**Once fixed, open:** http://localhost:5000 🎉
