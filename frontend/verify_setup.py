#!/usr/bin/env python3
"""
Setup Verification Script for Safety Surveillance Dashboard
Checks that all files are in the correct location
"""

import os
import sys
from pathlib import Path

def check_file_structure():
    """Verify all required files and directories exist"""
    
    print("🔍 Checking Safety Surveillance Dashboard Setup...\n")
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}\n")
    
    required_files = {
        'app.py': 'Flask server',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Documentation',
    }
    
    required_dirs = {
        'templates': 'HTML templates',
        'static': 'CSS/JS files',
        'static/css': 'Stylesheets',
        'static/js': 'JavaScript',
    }
    
    required_in_templates = {
        'templates/index.html': 'Main dashboard HTML'
    }
    
    required_in_static = {
        'static/css/style.css': 'Dashboard styles',
        'static/js/app.js': 'Dashboard JavaScript'
    }
    
    all_good = True
    
    # Check files
    print("📄 Checking required files:")
    for file, description in required_files.items():
        file_path = current_dir / file
        if file_path.exists():
            print(f"   ✅ {file} ({description})")
        else:
            print(f"   ❌ {file} ({description}) - MISSING!")
            all_good = False
    
    print()
    
    # Check directories
    print("📁 Checking required directories:")
    for dir_name, description in required_dirs.items():
        dir_path = current_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"   ✅ {dir_name}/ ({description})")
        else:
            print(f"   ❌ {dir_name}/ ({description}) - MISSING!")
            all_good = False
    
    print()
    
    # Check template files
    print("📝 Checking template files:")
    for file, description in required_in_templates.items():
        file_path = current_dir / file
        if file_path.exists():
            print(f"   ✅ {file} ({description})")
        else:
            print(f"   ❌ {file} ({description}) - MISSING!")
            all_good = False
    
    print()
    
    # Check static files
    print("🎨 Checking static files:")
    for file, description in required_in_static.items():
        file_path = current_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   ✅ {file} ({description}) - {size} bytes")
        else:
            print(f"   ❌ {file} ({description}) - MISSING!")
            all_good = False
    
    print()
    
    # Check optional surveillance_system.py
    print("🔧 Checking optional files:")
    surveillance_file = current_dir / 'surveillance_system.py'
    if surveillance_file.exists():
        print(f"   ✅ surveillance_system.py (Detection system) - Found")
    else:
        print(f"   ⚠️  surveillance_system.py (Detection system) - Not found")
        print(f"      Dashboard will run in demo mode without this file")
    
    print()
    print("=" * 60)
    
    if all_good:
        print("✅ Setup verification PASSED!")
        print("\nYou can now run:")
        print("   python app.py")
        print("\nOr with model paths:")
        print("   python app.py --ppe-model /path/to/ppe.pt --fire-model /path/to/fire.pt")
    else:
        print("❌ Setup verification FAILED!")
        print("\nPlease ensure all files are extracted to the same directory.")
        print("Run this script from the directory containing app.py")
        print("\nExpected structure:")
        print("   surveillance-dashboard/")
        print("   ├── app.py")
        print("   ├── surveillance_system.py (your file)")
        print("   ├── requirements.txt")
        print("   ├── README.md")
        print("   ├── templates/")
        print("   │   └── index.html")
        print("   └── static/")
        print("       ├── css/")
        print("       │   └── style.css")
        print("       └── js/")
        print("           └── app.js")
    
    print("=" * 60)
    
    return all_good

def create_missing_directories():
    """Create missing directories"""
    current_dir = Path.cwd()
    
    dirs_to_create = ['templates', 'static', 'static/css', 'static/js', 'uploads', 'outputs']
    
    print("\n📁 Creating missing directories...")
    for dir_name in dirs_to_create:
        dir_path = current_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created {dir_name}/")
        else:
            print(f"   ⏭️  {dir_name}/ already exists")

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("   SAFETY SURVEILLANCE DASHBOARD")
    print("   Setup Verification Tool")
    print("=" * 60)
    print()
    
    result = check_file_structure()
    
    if not result:
        print("\n💡 Would you like to create missing directories? (y/n): ", end='')
        try:
            response = input().strip().lower()
            if response == 'y':
                create_missing_directories()
                print("\n✅ Directories created!")
                print("⚠️  You still need to copy the template and static files!")
        except KeyboardInterrupt:
            print("\n\nAborted.")
            sys.exit(1)
    
    print()
