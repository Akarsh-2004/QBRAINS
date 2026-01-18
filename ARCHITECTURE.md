# Quantum Emotion Pipeline - Architecture Overview

## 🏗️ System Architecture

The Quantum Emotion Pipeline is designed as a modular, extensible system with clear separation of concerns and quantum-inspired processing principles.

## 📦 Core Components

### 1. Quantum Emotion AI (`quantum_emotion_ai.py`)
**Purpose**: Core quantum-inspired emotion processing engine

**Key Features**:
- Quantum superposition creation from multiple modalities
- Quantum interference patterns between emotions
- Sarcasm detection through modality mismatches
- Authenticity scoring based on consistency
- Quantum state collapse to primary emotion

**Input**: Face emotions, voice emotions, text emotions, context
**Output**: QuantumEmotionState with superposition, sarcasm, authenticity

### 2. Video Processor (`src/video_processor.py`)
**Purpose**: Extract emotions from video files

**Processing Flow**:
```
Video File
  ↓
Extract Frames (every Nth frame)
  ↓
Face Detection (Haar Cascade)
  ↓
Face Emotion Detection (CNN Model)
  ↓
Audio Extraction (FFmpeg)
  ↓
Audio Feature Extraction (MFCC, Chroma, etc.)
  ↓
Audio Emotion Detection (CNN+LSTM Model)
  ↓
Aggregate Results
```

**Output**: Face emotions, audio emotions, tone analysis, context

### 3. Audio/Text Processor (`src/audio_text_processor.py`)
**Purpose**: Process audio files and/or text input

**Processing Flow**:
```
Audio File / Text Input
  ↓
[If Audio] Feature Extraction → Model Prediction
[If Text] Keyword Analysis → Sentiment Analysis
  ↓
Combine Results (weighted fusion)
  ↓
Emotion Distribution
```

**Output**: Audio emotions, text emotions, combined emotions, sentiment

### 4. Ollama LLM Integration (`src/ollama_llm.py`)
**Purpose**: LLM-based prediction and reframing

**Features**:
- **Next Output Prediction**: Predicts what type of response will come next
- **Text Reframing**: Improves emotional clarity and appropriateness
- **Emotion Context Analysis**: Deep understanding of emotional content

**API Integration**:
- Uses Ollama REST API (`http://localhost:11434`)
- Supports multiple models (llama2, mistral, etc.)
- Handles connection errors gracefully

### 5. Main Pipeline (`src/quantum_pipeline.py`)
**Purpose**: Orchestrates all components

**Processing Pipeline**:
```
Input (Video/Audio/Text)
  ↓
[Mode Selection]
  ├─ Video Mode → VideoProcessor
  └─ Audio/Text Mode → AudioTextProcessor
  ↓
Raw Emotion Extraction
  ↓
Quantum Superposition Creation
  ↓
LLM Prediction (Next Output Type)
  ↓
LLM Reframing (Improve Output)
  ↓
Final Output Generation
```

## 🔄 Data Flow

### Video Mode Flow

```
Video File
  ↓
┌─────────────────────────────────┐
│   Video Processor              │
│   ├─ Face Detection            │
│   ├─ Expression Analysis       │
│   ├─ Audio Extraction          │
│   └─ Tone Analysis             │
└─────────────────────────────────┘
  ↓
Face Emotions + Audio Emotions + Context
  ↓
┌─────────────────────────────────┐
│   Quantum AI                   │
│   ├─ Superposition Creation   │
│   ├─ Interference Patterns     │
│   ├─ Sarcasm Detection         │
│   └─ Authenticity Scoring      │
└─────────────────────────────────┘
  ↓
Quantum State
  ↓
┌─────────────────────────────────┐
│   LLM (Ollama)                 │
│   ├─ Predict Next Output       │
│   └─ Reframe Text              │
└─────────────────────────────────┘
  ↓
Final Output
```

### Audio/Text Mode Flow

```
Audio File / Text Input
  ↓
┌─────────────────────────────────┐
│   Audio/Text Processor         │
│   ├─ Audio Feature Extract     │
│   ├─ Text Keyword Analysis     │
│   └─ Sentiment Analysis        │
└─────────────────────────────────┘
  ↓
Audio Emotions + Text Emotions
  ↓
┌─────────────────────────────────┐
│   Quantum AI                   │
│   ├─ Superposition Creation   │
│   ├─ Interference Patterns     │
│   ├─ Sarcasm Detection         │
│   └─ Authenticity Scoring      │
└─────────────────────────────────┘
  ↓
Quantum State
  ↓
┌─────────────────────────────────┐
│   LLM (Ollama)                 │
│   ├─ Predict Next Output       │
│   └─ Reframe Text              │
└─────────────────────────────────┘
  ↓
Final Output
```

## 🧠 Quantum-Inspired Concepts

### Superposition
Multiple emotional states exist simultaneously until "collapsed" to a primary emotion. This allows:
- Handling ambiguous emotional states
- Capturing complex, mixed emotions
- Modeling emotional uncertainty

### Interference
Emotions can interfere with each other:
- **Constructive Interference**: Emotions amplify (e.g., angry + fear)
- **Destructive Interference**: Emotions dampen (e.g., happy + sad)

### Collapse
Quantum state collapses to primary emotion based on:
- Probability distribution
- Uncertainty level
- Contextual modifiers

### Uncertainty (Heisenberg Principle)
Measured using Shannon entropy:
- High uncertainty = multiple emotions equally likely
- Low uncertainty = one emotion dominates

## 📊 Output Structure

```python
{
    'timestamp': 'ISO timestamp',
    'mode': 'video' | 'audio_text',
    'raw_emotions': {
        'face': {...},        # Video mode
        'audio': {...},
        'text': {...},        # Audio/text mode
        'tone': {...}         # Video mode
    },
    'quantum_state': {
        'primary_emotion': str,
        'emotional_superposition': Dict[str, float],
        'sarcasm_probability': float,
        'authenticity_score': float,
        'quantum_uncertainty': float
    },
    'llm_prediction': {
        'prediction_text': str,
        'confidence': float,
        'parsed': {...}       # If JSON parseable
    },
    'reframed_output': {
        'original': str,
        'reframed': str,
        'confidence': float
    },
    'final_output': {
        'summary': {...},
        'llm_prediction': {...},
        'reframed_text': str,
        'recommendations': List[str]
    }
}
```

## 🔌 Integration Points

### Model Integration
- **Face Model**: `model/improved_expression_model.keras`
- **Audio Model**: `model/sound_emotion_detector.keras`
- **Scaler**: `model/sound_emotion_scaler.pkl`
- **Encoder**: `model/sound_emotion_label_encoder.pkl`

### External Services
- **Ollama**: Local LLM service (default: `http://localhost:11434`)
- **FFmpeg**: Video/audio processing (for video mode)

## 🎯 Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new processors or models
3. **Quantum-Inspired**: Uses quantum principles for emotion processing
4. **Multi-Modal**: Supports face, voice, and text inputs
5. **LLM-Enhanced**: Uses LLM for prediction and reframing
6. **Context-Aware**: Considers temporal and situational context

## 🔄 Extension Points

### Adding New Processors
1. Create new processor class (e.g., `eeg_processor.py`)
2. Implement `process()` method
3. Integrate into `quantum_pipeline.py`

### Adding New Models
1. Train model using notebooks
2. Save to `model/` directory
3. Update processor to load new model

### Customizing Quantum Processing
Modify `quantum_emotion_ai.py`:
- Add new interference patterns
- Adjust superposition weights
- Customize collapse logic

### Adding LLM Features
Extend `ollama_llm.py`:
- Add new LLM methods
- Customize prompts
- Integrate with pipeline

## 📈 Performance Considerations

### Video Processing
- Frame sampling interval affects speed vs. accuracy
- Face detection can be optimized with GPU
- Audio extraction requires FFmpeg

### Audio Processing
- Feature extraction is CPU-intensive
- Model inference can be GPU-accelerated
- Batch processing improves throughput

### LLM Processing
- Ollama runs locally (no API costs)
- Response time depends on model size
- Can be parallelized for batch processing

## 🔒 Error Handling

Each component handles errors gracefully:
- **Model Loading**: Falls back to demo mode if models missing
- **Ollama**: Returns error messages if not running
- **FFmpeg**: Skips audio if not available
- **Processing**: Continues with available modalities

## 🧪 Testing

Test individual components:
```python
# Test video processor
from src.video_processor import VideoProcessor
processor = VideoProcessor()
results = processor.process_video('video.mp4')

# Test audio/text processor
from src.audio_text_processor import AudioTextProcessor
processor = AudioTextProcessor()
results = processor.process_text("Test text")

# Test LLM
from src.ollama_llm import OllamaLLM
llm = OllamaLLM()
response = llm.generate("Test prompt")
```

## 📚 Related Documentation

- `QUANTUM_PIPELINE_README.md`: Full documentation
- `QUICK_START.md`: Quick setup guide
- `examples/example_usage.py`: Usage examples
- Notebooks: Model training and evaluation

## 🔮 Future Enhancements

Potential improvements:
1. Real-time video processing (webcam)
2. Multi-person emotion detection
3. Emotion timeline visualization
4. Custom quantum interference patterns
5. Emotion memory and learning
6. Integration with more LLM providers
7. Web API interface
8. Mobile app integration

