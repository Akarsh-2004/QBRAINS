# Personal Emotion Memory System

A comprehensive emotion recognition system with **personal memory and contextual intelligence** that learns individual patterns and provides personalized insights.

## 🌟 Features

### Core Functionality
- **Real-time Emotion Detection**: Using advanced CNN with transfer learning
- **Personal Memory System**: Local storage of emotion history and patterns
- **Contextual Intelligence**: Learns your unique emotional patterns
- **Privacy-First**: All data stored locally on your device
- **Personal Insights**: AI-generated observations about your emotional patterns

### Personalization Features
- **Emotional Baselines**: Tracks your normal emotional levels
- **Pattern Recognition**: Identifies triggers and emotional cycles
- **Context Awareness**: Remembers emotions in different situations
- **Progress Tracking**: Weekly/monthly emotion trends
- **Personal Insights**: "You seem happier on weekends" type observations

## 📁 File Structure

```
QBRAINS/
├── notebooks/
│   └── face_emotion_reader.ipynb          # Improved training notebook
├── personal_emotion_memory.py             # Core memory system
├── emotion_memory_gui.py                  # GUI interface
├── emotion_integration.py                  # Integration script
├── data/expressions/                       # Training dataset
├── model/                                  # Trained models
└── README_PERSONAL_MEMORY.md              # This file
```

## 🚀 Quick Start

### 1. Train the Model
First, run the improved emotion recognition model:

```bash
# Open the improved notebook
jupyter notebook notebooks/face_emotion_reader.ipynb

# Run all cells to train the model
# The model will be saved as: model/improved_expression_model.keras
```

### 2. Run the Personal System

#### Option A: Interactive Mode (Recommended for beginners)
```bash
python emotion_integration.py
```

#### Option B: GUI Mode
```bash
python emotion_memory_gui.py
```

#### Option C: Camera Session
```python
from emotion_integration import EmotionReaderWithMemory

reader = EmotionReaderWithMemory()
reader.start_camera_session()  # Press 'q' to quit
```

## 🎯 Usage Examples

### Camera Session with Memory
```python
from emotion_integration import EmotionReaderWithMemory

# Initialize system
reader = EmotionReaderWithMemory()

# Start real-time emotion tracking
reader.start_camera_session(context="work_session")
# Emotions automatically saved every 5 seconds
```

### Analyze Image Files
```python
# Analyze specific images
reader.analyze_image_file("photo.jpg", context="family_gathering")

# Get personal summary
summary = reader.get_personal_summary()
print(f"Total sessions: {summary['total_sessions']}")
print(f"Baseline happiness: {summary['baseline_emotions']['Happy']:.2f}")
```

### Manual Memory Management
```python
from personal_emotion_memory import PersonalEmotionMemory

memory = PersonalEmotionMemory()

# Create/load user
user_id = memory.create_user_profile("your_name")

# Store emotions manually
emotions = {'Happy': 0.8, 'Neutral': 0.2, 'Angry': 0.0, 
           'Disgust': 0.0, 'Fear': 0.0, 'Sad': 0.0, 'Surprise': 0.0}
memory.store_emotion_session(emotions, context="morning_routine")

# Get insights
insights = memory.get_personal_insights(days=7)
patterns = memory.get_emotion_patterns()
```

## 📊 Personal Insights Examples

The system generates personalized insights like:

- **Emotion Spikes**: "Your anxiety level (0.8) is higher than your usual baseline (0.3)"
- **Pattern Recognition**: "You tend to be happier on weekends (0.7 vs 0.4 on weekdays)"
- **Context Patterns**: "You're more focused during work sessions (neutral: 0.6)"
- **Temporal Trends**: "Your happiness has increased by 15% this month"

## 🔧 Technical Architecture

### Data Storage
- **SQLite Database**: Local, encrypted storage for user data
- **Tables**: user_profiles, emotion_sessions, personal_insights
- **Privacy**: All data stored locally, never transmitted

### Learning Algorithm
- **Baseline Tracking**: Exponential moving average (α=0.1)
- **Pattern Detection**: Statistical analysis of emotion history
- **Context Learning**: Emotion patterns by situation/time
- **Insight Generation**: Rule-based with confidence scoring

### Model Integration
- **Transfer Learning**: EfficientNetB0 backbone
- **Real-time Processing**: Optimized for live camera feed
- **Face Detection**: OpenCV Haar cascades
- **Emotion Classification**: 7-class softmax output

## 🛡️ Privacy & Security

### Data Protection
- **Local-First**: All data stored on your device only
- **No Cloud**: No data transmission to external servers
- **User Control**: Full data export and deletion capabilities
- **Encryption**: SQLite database encryption available

### Data Management
```python
# Export all your data
memory.export_user_data("my_emotion_data.json")

# Delete all data (privacy compliance)
memory.delete_user_data()

# View data retention settings
profile = memory.load_user_profile(user_id)
print(profile.preferences['retention'])  # "90 days", "1 year", etc.
```

## 📈 Performance Improvements

### Model Accuracy
- **Original**: ~54% accuracy
- **Improved**: 70-80% accuracy target
- **Techniques**: Transfer learning, data augmentation, class balancing

### Personalization Benefits
- **Individual Baselines**: Accounts for personality differences
- **Context Awareness**: Reduces false positives
- **Pattern Learning**: Improves prediction over time
- **Personal Insights**: Adds value beyond raw predictions

## 🎨 GUI Features

### Dashboard
- **Current Emotions**: Real-time emotion display
- **Personal Baseline**: Your normal emotional levels
- **Quick Input**: Manual emotion entry

### History Tab
- **Timeline View**: Emotion trends over time
- **Context Filtering**: View by situation/context
- **Time Range**: 1 day to 90 days

### Insights Tab
- **Personal Observations**: AI-generated insights
- **Pattern Analysis**: Emotional patterns and triggers
- **Confidence Scores**: Reliability of insights

### Settings Tab
- **Privacy Controls**: Data retention settings
- **Insight Frequency**: Real-time, daily, weekly
- **Export/Import**: Data portability options

## 🔍 Advanced Usage

### Custom Contexts
```python
# Define custom contexts for better pattern learning
contexts = ["work", "home", "exercise", "social", "study"]
for context in contexts:
    reader.start_camera_session(context=context)
```

### Integration with Other Apps
```python
# Use as a library in other applications
from emotion_integration import EmotionReaderWithMemory

class WellnessApp:
    def __init__(self):
        self.emotion_reader = EmotionReaderWithMemory()
    
    def track_mood(self):
        emotions = self.emotion_reader.predict_from_image("selfie.jpg")
        return self.emotion_reader.get_personal_summary()
```

### Batch Analysis
```python
# Analyze multiple images
import glob
image_files = glob.glob("photos/*.jpg")

for image_path in image_files:
    emotions = reader.analyze_image_file(image_path, context="batch_analysis")
    print(f"{image_path}: {max(emotions, key=emotions.get)}")
```

## 🐛 Troubleshooting

### Common Issues

**Model Not Found**
```
Error: No trained model found
Solution: Run the improved notebook first to train the model
```

**Camera Not Working**
```
Error: Could not open camera
Solution: Check camera permissions and ensure no other app is using it
```

**No Faces Detected**
```
Error: No faces detected in the image
Solution: Ensure good lighting and face is clearly visible
```

### Performance Tips
- **Good Lighting**: Ensure adequate lighting for better face detection
- **Face Position**: Keep face centered and clearly visible
- **Regular Use**: More data = better personalization
- **Context Labels**: Use meaningful context names for better patterns

## 🚀 Future Enhancements

### Planned Features
- **Voice Emotion Analysis**: Integrate speech emotion detection
- **Physiological Sensors**: Heart rate, stress indicators
- **Social Patterns**: Group emotion dynamics
- **Mobile App**: iOS/Android applications
- **Web Dashboard**: Browser-based interface

### Research Directions
- **Multi-modal Fusion**: Combine face, voice, and text
- **Long-term Patterns**: Monthly/seasonal emotion cycles
- **Intervention Suggestions**: Personalized wellness recommendations
- **Social Comparison**: Anonymous, aggregated insights

## 📞 Support

### Getting Help
1. **Check README**: Review this documentation first
2. **Test Model**: Ensure training notebook completed successfully
3. **Verify Data**: Check database file creation
4. **Review Logs**: Check console output for error messages

### Contributing
Welcome contributions for:
- **Model Improvements**: Better accuracy, faster inference
- **UI/UX Enhancements**: Better interface design
- **New Features**: Additional emotion tracking capabilities
- **Documentation**: Improved guides and examples

---

**Made with ❤️ for personal emotional intelligence and privacy**

*Your emotions, your data, your insights.*
