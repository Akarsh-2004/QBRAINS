# Build Instructions for Quantum Emotion Pipeline

## Building the Desktop Executable

### Prerequisites

1. Install Python 3.8 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
pip install pyinstaller
```

### Build Steps

1. **Build the executable:**
```bash
python build_executable.py
```

The build script automatically detects your platform and uses appropriate settings:
- **macOS**: Uses `--onedir` mode (required for windowed apps)
- **Windows**: Uses `--onefile` mode
- **Linux**: Uses `--onefile` mode

2. **Find the executable:**
   - **macOS**: `dist/QuantumEmotionPipeline.app` (double-click to run)
   - **Windows**: `dist/QuantumEmotionPipeline.exe`
   - **Linux**: `dist/QuantumEmotionPipeline`

### Running the Application

#### Web-based Desktop App (Recommended - No tkinter needed!)
```bash
python desktop_app_web.py
```
This opens the web dashboard in your browser automatically.

#### Tkinter Desktop App (Requires tkinter)
```bash
# On macOS, you may need system Python or tkinter installation
# See INSTALL_TKINTER.md for instructions
python desktop_app.py
```

#### Using the Executable
- **macOS**: Double-click `dist/QuantumEmotionPipeline.app`
- **Windows/Linux**: Run `dist/QuantumEmotionPipeline` or `dist/QuantumEmotionPipeline.exe`

#### Web API
```bash
cd api
python main.py
```

Then open `http://localhost:8000` in your browser.

#### Web Dashboard
The dashboard is available at `http://localhost:8000` when the API is running.

### Testing

Run tests:
```bash
python -m pytest tests/
```

### Package Installation

Install as a package:
```bash
pip install -e .
```

### Notes

- The executable includes all models and dependencies
- First run may be slow as models are loaded
- Ensure you have sufficient disk space (models are large)
- GPU support requires CUDA-compatible TensorFlow

