#!/usr/bin/env python3
"""
Main launcher script for Quantum Emotion Pipeline
Automatically chooses the best available interface
"""

import sys
from pathlib import Path

def check_tkinter():
    """Check if tkinter is available"""
    try:
        import tkinter
        return True
    except ImportError:
        return False

def main():
    """Main launcher"""
    print("🌌 Quantum Emotion Pipeline")
    print("=" * 50)
    
    # Check for tkinter
    has_tkinter = check_tkinter()
    
    if has_tkinter:
        print("✓ Tkinter available - using GUI desktop app")
        print("Starting desktop application...\n")
        try:
            from desktop_app import main as desktop_main
            desktop_main()
        except Exception as e:
            print(f"⚠️ Error starting GUI app: {e}")
            print("Falling back to web-based interface...\n")
            has_tkinter = False
    
    if not has_tkinter:
        print("ℹ️  Using web-based interface (no tkinter needed)")
        print("Starting web server...\n")
        try:
            from desktop_app_web import main as web_main
            web_main()
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n💡 Make sure you have installed:")
            print("   pip install fastapi uvicorn")
            sys.exit(1)

if __name__ == "__main__":
    main()

