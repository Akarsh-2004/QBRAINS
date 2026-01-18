# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for QBRAINS Desktop Application
Builds a standalone executable with all dependencies bundled
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

# Get absolute path to project directory
block_cipher = None
project_dir = os.path.abspath(os.path.dirname(SPEC))

# Collect all data files and hidden imports for major packages
datas = []
binaries = []
hiddenimports = []

# Bundle application files
datas += [
    (os.path.join(project_dir, 'api'), 'api'),
    (os.path.join(project_dir, 'src'), 'src'),
    (os.path.join(project_dir, 'config.json'), '.'),
]

# Bundle model files if they exist (optional - can be downloaded on first run)
model_dir = os.path.join(project_dir, 'model')
if os.path.exists(model_dir):
    datas += [(model_dir, 'model')]

# Essential hidden imports for the application
hiddenimports += [
    # Core application modules
    'src.utils',
    'src.audio_features',
    'src.custom_layers',
    'src.safe_model_loader',
    'src.quantum_emotion_engine',
    'src.audio_text_processor',
    'src.video_processor',
    'src.eeg_processor',
    'src.fallback_emotion_detector',
    
    # FastAPI and related
    'fastapi',
    'fastapi.applications',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'fastapi.responses',
    'fastapi.staticfiles', 
    'fastapi.routing',
    'fastapi.encoders',
    
    # Uvicorn server
    'uvicorn',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.logging',
    
    # Pydantic
    'pydantic',
    'pydantic.fields',
    'pydantic.main',
    'pydantic_core',
    
    # Starlette
    'starlette',
    'starlette.applications',
    'starlette.middleware',
    'starlette.routing',
    'starlette.responses',
    
    # ML/AI libraries
    'tensorflow',
    'tensorflow.keras',
    'transformers',
    'torch',
    'sklearn',
    'sklearn.preprocessing',
    'librosa',
    'cv2',
    'PIL',
    'numpy',
    'scipy',
    
    # Audio processing
    'soundfile',
    'audioread',
    'resampy',
    
    # Utilities
    'joblib',
    'requests',
    'urllib3',
]

# Collect all dependencies for major packages
packages_to_collect = [
    'fastapi',
    'uvicorn', 
    'pydantic',
    'starlette',
    'transformers',
    'librosa',
]

for package in packages_to_collect:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1] 
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect {package}: {e}")

a = Analysis(
    [os.path.join(project_dir, 'desktop_app_web.py')],  # Use desktop app as entry point
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Exclude if not needed
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder build (recommended for easier debugging and faster startup)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QBRAINS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console for debugging - set to False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='QBRAINS',
)

