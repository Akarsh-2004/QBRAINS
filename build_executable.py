#!/usr/bin/env python3
"""
Build script for creating executable
Uses PyInstaller to create standalone executable
"""

import subprocess
import sys
import os
from pathlib import Path

def build_executable():
    """Build executable using PyInstaller"""
    import platform
    
    # Detect platform
    is_macos = platform.system() == 'Darwin'
    is_windows = platform.system() == 'Windows'
    
    # Base command
    cmd = [
        'pyinstaller',
        '--name=QuantumEmotionPipeline',
        '--additional-hooks-dir=.',  # Use local hooks
    ]
    
    # macOS: Use onedir instead of onefile for windowed mode
    if is_macos:
        cmd.extend([
            '--onedir',  # Use directory mode on macOS
            '--windowed',  # No console window
        ])
    elif is_windows:
        cmd.extend([
            '--onefile',
            '--windowed',  # No console window on Windows
        ])
    else:
        # Linux
        cmd.extend([
            '--onefile',
            # No --windowed on Linux (keeps console)
        ])
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Common options
    cmd.extend([
        f'--add-data={project_root / "model"}{os.pathsep}model',  # Include model directory
        f'--add-data={project_root / "src"}{os.pathsep}src',  # Include src directory
        f'--add-data={project_root / "api"}{os.pathsep}api',  # Include API directory
        f'--add-data={project_root / "config.json"}{os.pathsep}.',  # Include config
        '--hidden-import=PIL',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=tensorflow',
        '--hidden-import=transformers',
        '--hidden-import=torch',
        '--hidden-import=sklearn',
        '--hidden-import=librosa',
        '--hidden-import=fastapi',
        '--hidden-import=fastapi.applications',
        '--hidden-import=fastapi.middleware',
        '--hidden-import=fastapi.middleware.cors',
        '--hidden-import=fastapi.responses',
        '--hidden-import=fastapi.staticfiles',
        '--hidden-import=fastapi.routing',
        '--hidden-import=uvicorn',
        '--hidden-import=uvicorn.lifespan',
        '--hidden-import=uvicorn.lifespan.on',
        '--hidden-import=uvicorn.protocols',
        '--hidden-import=uvicorn.protocols.http',
        '--hidden-import=uvicorn.protocols.websockets',
        '--hidden-import=uvicorn.loops',
        '--hidden-import=uvicorn.loops.auto',
        '--hidden-import=uvicorn.loops.selector',
        '--hidden-import=pydantic',
        '--hidden-import=pydantic.fields',
        '--hidden-import=pydantic.main',
        '--hidden-import=starlette',
        '--hidden-import=starlette.applications',
        '--hidden-import=starlette.middleware',
        '--hidden-import=starlette.routing',
        '--collect-all=fastapi',
        '--collect-all=uvicorn',
        '--collect-all=pydantic',
        '--hidden-import=src.utils',
        '--hidden-import=src.audio_features',
        '--collect-all=tensorflow',
        '--collect-all=transformers',
        '--collect-all=librosa',
        '--noconfirm',  # Overwrite without asking
    ])
    
    # Use unified launcher
    app_file = 'launcher.py'
    cmd.append(app_file)
    
    # Add test script for debugging (optional)
    # Uncomment to test paths:
    # test_file = 'test_executable_paths.py'
    # cmd.extend(['--add-data', f'{project_root / test_file}:.'])
    
    print("Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Executable built successfully!")
        print("Location: dist/QuantumEmotionPipeline")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ PyInstaller not found. Install it with: pip install pyinstaller")
        sys.exit(1)


if __name__ == "__main__":
    build_executable()

