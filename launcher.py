#!/usr/bin/env python3
"""
Unified Launcher for Quantum Emotion Pipeline
Handles initialization and launches the appropriate interface
"""

import sys
import os
from pathlib import Path

# Fix Python path for PyInstaller (must be first)
if getattr(sys, 'frozen', False):
    if hasattr(sys, '_MEIPASS'):
        sys.path.insert(0, sys._MEIPASS)
    elif sys.platform == 'darwin' and '.app/Contents/MacOS' in sys.executable:
        # macOS app bundle - Resources has all modules
        app_bundle = Path(sys.executable).parent.parent
        resources = app_bundle / 'Resources'
        if resources.exists() and str(resources) not in sys.path:
            sys.path.insert(0, str(resources))

# Add project root to path
if getattr(sys, 'frozen', False):
    if sys.platform == 'darwin' and '.app' in sys.executable:
        project_root = Path(sys.executable).parent.parent / 'Resources'
    else:
        project_root = Path(sys.executable).parent
else:
    project_root = Path(__file__).parent

sys.path.insert(0, str(project_root))

def main():
    """Main launcher function"""
    print("🌌 Quantum Emotion Pipeline - Starting...")
    print("=" * 60)
    
    # Check for web-based app (preferred - no dependencies)
    try:
        from desktop_app_web import main as web_main
        print("✓ Using web-based interface")
        web_main()
        return
    except ImportError as e:
        print(f"⚠️ Web app not available: {e}")
    
    # Fallback to GUI app
    try:
        import tkinter
        from desktop_app import main as gui_main
        print("✓ Using GUI interface")
        gui_main()
        return
    except ImportError:
        print("⚠️ GUI not available (tkinter missing)")
    
    # Fallback to API only
    try:
        print("✓ Starting API server only")
        os.chdir(project_root / "api")
        from main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return
    except Exception as e:
        print(f"❌ Failed to start any interface: {e}")
        print("\nPlease ensure dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()

