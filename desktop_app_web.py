#!/usr/bin/env python3
"""
Web-based Desktop Application Launcher
Opens the web dashboard in default browser and starts the API server
Works on all platforms without tkinter dependency
"""

import webbrowser
import threading
import time
import sys
from pathlib import Path
import subprocess
import signal
import os

# Fix PyInstaller path for macOS app bundles (must be before other imports)
if getattr(sys, 'frozen', False):
    if hasattr(sys, '_MEIPASS'):
        sys.path.insert(0, sys._MEIPASS)
    elif sys.platform == 'darwin' and '.app/Contents/MacOS' in sys.executable:
        app_bundle = Path(sys.executable).parent.parent
        resources = app_bundle / 'Resources'
        if resources.exists() and str(resources) not in sys.path:
            sys.path.insert(0, str(resources))

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Explicit imports for PyInstaller to detect
try:
    import fastapi
    import uvicorn
    import pydantic
    import starlette
except ImportError:
    pass  # Will be handled in start_api_server

def start_api_server():
    """Start the FastAPI server"""
    try:
        # Import fastapi explicitly at module level for PyInstaller
        import fastapi
        import uvicorn
        from api.main import app
        
        print("🚀 Starting API server...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Python path: {sys.path}")
        print("\n💡 Make sure you have installed:")
        print("   pip install fastapi uvicorn python-multipart")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def wait_for_server(url, timeout=60):
    """Wait for server to be responsive with increased timeout"""
    import urllib.request
    import urllib.error
    
    print(f"⏳ Waiting for server at {url}...")
    start_time = time.time()
    last_check_time = start_time
    
    while time.time() - start_time < timeout:
        try:
            # Add progress indicator
            if time.time() - last_check_time > 5:
                elapsed = int(time.time() - start_time)
                print(f"   Still waiting... ({elapsed}s elapsed)")
                last_check_time = time.time()
            
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    print("✅ Server is ready!")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)
        except Exception as e:
            print(f"   Check error: {e}")
            time.sleep(1)
    
    print(f"❌ Server timed out after {timeout} seconds")
    return False

def open_browser():
    """Open browser in app mode if possible"""
    url = "http://127.0.0.1:8000"
    
    # Wait for server to start
    if not wait_for_server(url):
        print("⚠️ Proceeding to open browser anyway...")
    
    print(f"🌐 Opening Quantum Interface at {url}")
    
    # Try to open in app mode (Chrome/Edge)
    # This gives a "native app" feel without address bar
    try:
        if sys.platform == 'win32':
             # Try Edge (default on Windows)
             subprocess.Popen(['start', 'msedge', '--app=' + url], shell=True)
             return
        elif sys.platform == 'darwin':
             # Try Chrome on macOS
             subprocess.Popen(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--app=' + url])
             return
    except Exception:
        pass # Fallback to default browser
        
    webbrowser.open(url)

def main():
    """Main entry point"""
    print("=" * 60)
    print("🌌 Quantum Emotion Pipeline - Desktop App")
    print("=" * 60)
    print("\nStarting web-based interface...")
    print("The application will open in your default browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Start browser in separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start API server (this will block)
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()

