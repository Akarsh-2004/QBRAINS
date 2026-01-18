# Building Standalone Executable

This guide explains how to build a standalone executable that includes all dependencies and models.

## Prerequisites

1. **Python 3.8+** installed
2. **All dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **PyInstaller** installed:
   ```bash
   pip install pyinstaller
   ```
4. **All models trained** and present in `model/` directory:
   - `improved_expression_model.keras` (face emotion model)
   - `sound_emotion_detector.keras` (audio emotion model)
   - `sound_emotion_scaler.pkl`
   - `sound_emotion_label_encoder.pkl`
   - `emotion_llm_final/` (emotion LLM model)
   - `emotion_label_encoder.pkl`

## Building the Executable

### Quick Build

```bash
python build_executable.py
```

This will:
- Detect your platform (macOS/Windows/Linux)
- Create a standalone executable
- Bundle all models and dependencies
- Output to `dist/QuantumEmotionPipeline`

### Platform-Specific Outputs

- **macOS**: `dist/QuantumEmotionPipeline.app` (application bundle)
- **Windows**: `dist/QuantumEmotionPipeline.exe` (single executable)
- **Linux**: `dist/QuantumEmotionPipeline` (single executable)

## What's Included

The executable includes:
- ✅ All Python dependencies
- ✅ TensorFlow models
- ✅ Configuration files
- ✅ Source code
- ✅ Model files
- ✅ API server

## Running the Executable

### macOS
```bash
open dist/QuantumEmotionPipeline.app
# Or
./dist/QuantumEmotionPipeline.app/Contents/MacOS/QuantumEmotionPipeline
```

### Windows
```bash
dist\QuantumEmotionPipeline.exe
```

### Linux
```bash
./dist/QuantumEmotionPipeline
```

## Troubleshooting

### Build Fails

1. **Missing PyInstaller**:
   ```bash
   pip install pyinstaller
   ```

2. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Models not found**:
   - Ensure all model files are in `model/` directory
   - Train models if missing (see notebooks)

### Executable Doesn't Run

1. **Check console output** (if available)
2. **Verify models are bundled**: Check `dist/QuantumEmotionPipeline/model/`
3. **Check permissions**: Make executable on Linux:
   ```bash
   chmod +x dist/QuantumEmotionPipeline
   ```

### Large Executable Size

The executable is large (~500MB-2GB) because it includes:
- TensorFlow runtime
- PyTorch (if using transformers)
- All model files
- All dependencies

This is normal for ML applications.

## Distribution

To distribute the executable:

1. **macOS**: 
   - Zip the `.app` bundle
   - Users can double-click to run

2. **Windows**:
   - Distribute the `.exe` file
   - May need to install Visual C++ Redistributable

3. **Linux**:
   - Distribute the executable
   - May need to install system dependencies

## Notes

- The executable is **self-contained** - no Python installation needed
- Models are **bundled** - no separate model files needed
- Configuration is **included** - can be modified in `config.json`
- Ollama is **optional** - works without it (with reduced features)

## Advanced Options

To customize the build, edit `build_executable.py`:

- Change app name: `--name=YourAppName`
- Add more data files: `--add-data=path:dest`
- Include/exclude modules: `--hidden-import` or `--exclude-module`

