# Quantum Emotion Engine - Complete Guide

## 🌌 Overview

The Quantum Emotion Engine is a sophisticated system that processes multiple emotion sources using quantum principles, then generates optimally formatted outputs through an emotion LLM and Ollama.

## 🔄 Complete Pipeline Flow

```
INPUT
  ↓
┌─────────────────────────────────────────────────────────┐
│  MULTI-SOURCE COLLECTION                               │
│  ├─ Tone/Sentiment Analysis (text/audio)               │
│  ├─ Long-term Memory (emotional patterns)              │
│  ├─ Conversation History (recent context)              │
│  ├─ Expression Analysis (face emotions)                │
│  └─ Context Analysis (situation, time, etc.)           │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│  QUANTUM SUPERPOSITION CREATION                        │
│  ├─ Collect all possibilities from sources             │
│  ├─ Apply quantum interference patterns                │
│  ├─ Calculate quantum amplitudes                       │
│  └─ Measure uncertainty (Shannon entropy)               │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│  QUANTUM STATE COLLAPSE                                │
│  ├─ Low uncertainty → Highest probability              │
│  └─ High uncertainty → Weighted random selection       │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│  EMOTION LLM PROCESSING                                │
│  ├─ Predict emotion from text                          │
│  ├─ Generate emotion-aware output                      │
│  └─ Provide emotion context                            │
└─────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────┐
│  OLLAMA FORMATTING & MODULATION                        │
│  ├─ Reframe text for emotional clarity                 │
│  ├─ Adjust tone and intensity                          │
│  └─ Ensure contextual appropriateness                  │
└─────────────────────────────────────────────────────────┘
  ↓
FINAL OUTPUT (emotionally modulated and formatted)
```

## 🧠 Key Components

### 1. Quantum Emotion Engine (`quantum_emotion_engine.py`)

**Core Features:**
- **Multi-source possibility collection**: Gathers emotions from tone, memory, history, expression, context
- **Quantum superposition**: Creates quantum state from all possibilities
- **Quantum interference**: Emotions interact and modify each other
- **State collapse**: Determines primary emotion based on uncertainty
- **Integration**: Connects to Emotion LLM and Ollama

### 2. Long-term Memory (`LongTermMemory`)

**Features:**
- Stores interaction history
- Tracks emotional patterns over time
- Maintains emotional baselines per user
- Identifies contextual preferences
- Learns from conversation flow

### 3. Conversation History (`ConversationHistory`)

**Features:**
- Maintains recent conversation context
- Tracks emotion flow
- Provides context window for analysis
- Supports multi-turn conversations

### 4. Tone Analyzer (`ToneAnalyzer`)

**Features:**
- Analyzes text sentiment
- Processes audio tone
- Combines text + audio analysis
- Provides emotion distribution

### 5. Integrated Pipeline (`quantum_pipeline_integrated.py`)

**Features:**
- Simple interface for complete pipeline
- Chat mode support
- Memory management
- Result formatting

## 📊 Quantum Principles Used

### Superposition
Multiple emotional states exist simultaneously until collapsed:
- All source emotions contribute to quantum state
- Each possibility has amplitude and probability
- System explores all combinations

### Interference
Emotions interfere with each other:
- **Constructive**: Emotions amplify (angry + fear)
- **Destructive**: Emotions dampen (happy + sad)
- Patterns emerge from interference

### Uncertainty (Heisenberg Principle)
Measured using Shannon entropy:
- **Low uncertainty** (< 0.3): Clear emotion, confident collapse
- **High uncertainty** (> 0.6): Ambiguous, probabilistic collapse
- Guides collapse strategy

### Collapse
Quantum state collapses to primary emotion:
- Based on probability distribution
- Weighted by uncertainty level
- Considers interference patterns

## 🚀 Usage Examples

### Basic Usage

```python
from src.quantum_pipeline_integrated import IntegratedQuantumPipeline

# Initialize
pipeline = IntegratedQuantumPipeline()

# Process text
result = pipeline.process(
    text="I'm feeling really excited!",
    context={'situation': 'work_context'}
)

print(result['final_output']['formatted_text'])
```

### With Memory

```python
# First interaction
result1 = pipeline.process(text="I'm sad today")
# Memory learns: user tends to be sad

# Second interaction
result2 = pipeline.process(text="I'm okay")
# Memory considers: recent pattern shows sadness, but current is neutral
# Quantum engine balances both
```

### Multimodal Input

```python
result = pipeline.process(
    text="I'm fine",  # Says fine
    face_emotions={'sad': 0.8, 'neutral': 0.2},  # But looks sad
    context={'situation': 'personal_context'}
)
# Quantum engine detects mismatch and considers both
```

### Chat Mode

```python
# Interactive chat
while True:
    user_input = input("You: ")
    response = pipeline.chat(user_input)
    print(f"AI: {response}")
```

## 🔧 Configuration

### Initialize with Custom Paths

```python
pipeline = IntegratedQuantumPipeline(
    emotion_llm_path="model/emotion_llm_final",
    label_encoder_path="model/emotion_label_encoder.pkl",
    ollama_model="llama2",  # or "mistral", "codellama", etc.
    ollama_url="http://localhost:11434"
)
```

### Memory Configuration

```python
# Access memory
memory = pipeline.engine.memory

# Get baseline
baseline = memory.get_emotional_baseline(user_id="user123")

# Get recent pattern
pattern = memory.get_recent_pattern(window=20)

# Clear memory (if needed)
pipeline.clear_memory()
```

## 📈 Output Structure

```python
{
    'input': {
        'text': '...',
        'audio_path': '...',
        'video_path': '...',
        'context': {...}
    },
    'quantum_superposition': {
        'possibilities': [
            {
                'emotion': 'happy',
                'probability': 0.65,
                'source': 'tone',
                'confidence': 0.8,
                'metadata': {...}
            },
            ...
        ],
        'collapsed_emotion': 'happy',
        'uncertainty': 0.25,
        'interference_patterns': {...}
    },
    'emotion_llm_output': {
        'text': '...',
        'emotion': 'happy',
        'confidence': 0.85,
        ...
    },
    'final_output': {
        'original_text': '...',
        'formatted_text': '...',  # Ollama-formatted
        'target_emotion': 'happy',
        'confidence': 0.9,
        'quantum_uncertainty': 0.25,
        'modulation': {
            'emotion_intensity': 0.75,
            'tone_adjustment': {
                'warmth': 0.8,
                'energy': 0.7,
                'formality': -0.3
            }
        }
    },
    'timestamp': '2024-01-01T12:00:00'
}
```

## 🎯 Key Features

### 1. Multi-Source Integration
- Combines tone, memory, history, expression, context
- Each source weighted by confidence
- Quantum interference between sources

### 2. Quantum Processing
- Superposition of all possibilities
- Interference patterns
- Uncertainty measurement
- Probabilistic collapse

### 3. Memory Learning
- Learns emotional patterns
- Tracks baselines
- Identifies triggers
- Contextual preferences

### 4. LLM Integration
- Emotion prediction from text
- Emotion-aware generation
- Context provision

### 5. Ollama Formatting
- Natural language reframing
- Emotional modulation
- Tone adjustment
- Contextual appropriateness

## 🔬 Quantum Interference Patterns

The engine uses predefined interference patterns:

| Emotion Pair | Interference | Effect |
|-------------|--------------|--------|
| Happy + Sad | -0.3 | Strong negative (mutually exclusive) |
| Angry + Fear | +0.4 | Positive (amplify each other) |
| Surprise + Neutral | -0.2 | Breaks neutrality |
| Disgust + Happy | -0.4 | Strong negative |
| Fear + Sad | +0.3 | Co-occurrence |

## 📊 Probability Calculation

For each emotion:
```
quantum_amplitude = (
    base_amplitude_from_sources +
    interference_from_other_emotions * 0.1
)

probability = normalize(quantum_amplitude)
```

## 🎛️ Uncertainty-Based Collapse

- **Low Uncertainty (< 0.3)**: 
  - Clear emotion detected
  - Use highest probability
  
- **High Uncertainty (> 0.6)**:
  - Ambiguous state
  - Weighted random selection
  - Consider multiple possibilities

## 💡 Use Cases

1. **Conversational AI**: Natural, emotion-aware responses
2. **Emotion Analysis**: Multi-modal emotion detection
3. **Personalization**: Learning user emotional patterns
4. **Context Awareness**: Situation-appropriate responses
5. **Emotion Modulation**: Adjusting output tone and intensity

## 🔄 Memory Management

### Automatic Learning
- Tracks every interaction
- Updates patterns automatically
- Maintains baselines
- Identifies preferences

### Manual Control
```python
# Get memory summary
summary = pipeline.get_memory_summary()

# Clear memory
pipeline.clear_memory()

# Access memory directly
memory = pipeline.engine.memory
memory.add_interaction(emotion='happy', context={...})
```

## 🚀 Quick Start

```python
from src.quantum_pipeline_integrated import IntegratedQuantumPipeline

# Initialize
pipeline = IntegratedQuantumPipeline()

# Process
result = pipeline.process(
    text="I'm feeling great!",
    context={'situation': 'social_context'}
)

# Get formatted output
print(result['final_output']['formatted_text'])

# Chat mode
response = pipeline.chat("How are you?")
print(response)
```

## 📝 Notes

- **Memory File**: Stored in `data/emotion_memory.json`
- **Ollama Required**: Make sure Ollama is running for formatting
- **Emotion LLM**: Optional, falls back if not available
- **Quantum Processing**: All calculations use numpy for efficiency

## 🔮 Advanced Usage

### Custom Interference Patterns

Modify `_create_interference_matrix()` in `QuantumEmotionEngine` to add custom patterns.

### Custom Weighting

Adjust source weights in `process_input()`:
- Tone: 0.6
- Memory baseline: 0.3
- Memory pattern: 0.5
- History: 0.4
- Expression: 0.6
- Context: 0.3

### Custom Collapse Strategy

Modify `_collapse_quantum_state()` to implement custom collapse logic.

---

**The Quantum Emotion Engine provides a complete, quantum-inspired solution for emotion-aware AI interactions!**

