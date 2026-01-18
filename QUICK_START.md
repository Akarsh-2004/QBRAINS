# Quick Start Guide

## 🚀 Fastest Way to Run

### Option 1: Web-based Desktop App (Recommended)

```bash
python desktop_app_web.py
```

This will:
1. Start the API server
2. Open the web dashboard in your browser
3. Provide full functionality without any GUI dependencies

### Option 2: Web API Only

```bash
cd api
python main.py
```

Then open `http://localhost:8000` in your browser.

### Option 3: Command Line

```bash
python -m src.quantum_pipeline_integrated
```

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# That's it! No additional setup needed.
```

## 🎯 What You Can Do

1. **Process Text**: Analyze emotions from text input
2. **Process Audio**: Upload audio files for emotion detection
3. **Process Video**: Upload video files for multi-modal analysis
4. **Process EEG**: Upload EEG CSV files for brain signal analysis
5. **Real-time Processing**: Use webcam for live emotion detection
6. **Multi-Person Tracking**: Track multiple people simultaneously
7. **Chat Interface**: Interactive emotion-aware chat

## 🐛 Troubleshooting

### Tkinter Error?
Use the web-based app instead:
```bash
python desktop_app_web.py
```

### Port Already in Use?
Change the port in `api/main.py` or kill the process:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Models Not Found?
Make sure you've trained the models using the notebooks:
- `notebooks/face_emotion_reader_image.ipynb`
- `notebooks/sound_emotion_detector.ipynb`
- `notebooks/llm_emotion_training.ipynb`

## 📚 More Information

- See `BUILD_INSTRUCTIONS.md` for building executables
- See `COMPLETION_SUMMARY.md` for feature overview
- See `PROJECT_STATUS.md` for project status

