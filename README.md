# 🌌 Quantum Emotion Pipeline

A comprehensive, quantum-inspired multi-modal emotion detection and processing system.

## 🚀 Quick Start

### Easiest Way (Web-based - No Dependencies!)

```bash
python desktop_app_web.py
```

This will:
- Start the API server
- Open the web dashboard in your browser
- Provide full functionality

### Alternative: Smart Launcher

```bash
python run.py
```

Automatically chooses the best available interface (GUI if tkinter available, otherwise web-based).

## 📦 Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 🎯 Features

- ✅ **Multi-Modal Emotion Detection**: Face, Voice, Text, EEG
- ✅ **Quantum-Inspired Processing**: Superposition, interference, collapse
- ✅ **Real-time Processing**: Live video, audio, and EEG streams
- ✅ **Multi-Person Tracking**: Track and analyze multiple people
- ✅ **Memory & Learning**: Long-term emotional pattern tracking
- ✅ **LLM Integration**: Emotion-aware text generation
- ✅ **Web Dashboard**: Beautiful browser-based interface
- ✅ **REST API**: Complete API for all features
- ✅ **Desktop App**: Standalone application

## 🖥️ Running the Application

### Option 1: Web-based Desktop App (Recommended)

```bash
python desktop_app_web.py
```

**Advantages:**
- No tkinter dependency
- Works on all platforms
- Beautiful modern UI
- Same functionality as GUI app

### Option 2: Tkinter Desktop App

```bash
python desktop_app.py
```

**Note:** On macOS, you may need to install tkinter. See `INSTALL_TKINTER.md`.

### Option 3: Web API Only

```bash
cd api
python main.py
```

Then open `http://localhost:8000` in your browser.

### Option 4: Command Line

```bash
python -m src.quantum_pipeline_integrated --text "Your text here"
```

## 🔧 Building Executable

```bash
# Build standalone executable
python build_executable.py
```

**Output:**
- **macOS**: `dist/QuantumEmotionPipeline.app`
- **Windows**: `dist/QuantumEmotionPipeline.exe`
- **Linux**: `dist/QuantumEmotionPipeline`

## 📚 Documentation

- `QUICK_START.md` - Quick start guide
- `BUILD_INSTRUCTIONS.md` - Building executables
- `COMPLETION_SUMMARY.md` - Feature overview
- `PROJECT_STATUS.md` - Project status
- `INSTALL_TKINTER.md` - Tkinter installation guide

## 🐛 Troubleshooting

### Tkinter Not Available?

**Solution:** Use the web-based app:
```bash
python desktop_app_web.py
```

### Port 8000 Already in Use?

Change port in `api/main.py` or kill the process:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill

# Windows  
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Models Not Found?

Train models using the notebooks:
- `notebooks/face_emotion_reader_image.ipynb`
- `notebooks/sound_emotion_detector.ipynb`
- `notebooks/llm_emotion_training.ipynb`

## 🎓 Usage Examples

### Process Text
```python
from src.quantum_pipeline_integrated import IntegratedQuantumPipeline

pipeline = IntegratedQuantumPipeline()
result = pipeline.process(text="I'm feeling great!")
print(result['final_output']['formatted_text'])
```

### Process with EEG
```python
result = pipeline.process(
    text="Test",
    eeg_path="data/archive/s00.csv"
)
```

### Real-time Processing
```python
from src.realtime_processor import RealTimeProcessor

processor = RealTimeProcessor(pipeline=pipeline)
processor.start_video_stream(camera_id=0)
# Results available via processor.get_latest_result()
```

## 📊 System Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Models trained (see notebooks)
- Optional: GPU for faster processing

## 🔗 API Endpoints

- `POST /process/text` - Process text
- `POST /process/audio` - Process audio file
- `POST /process/video` - Process video file
- `POST /process/eeg` - Process EEG data
- `POST /chat` - Chat interface
- `GET /memory` - Get memory summary
- `POST /realtime/start` - Start real-time processing

See `api/main.py` for complete API documentation.

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! See CONTRIBUTING.md (if available).

---

**Made with ❤️ for quantum-inspired emotion processing**

