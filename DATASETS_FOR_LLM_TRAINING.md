# Datasets for Training Emotion-Aware LLM

Comprehensive list of datasets for training your quantum-inspired emotion pipeline LLM, organized by use case and modality.

## 🎯 Recommended Dataset Combinations

### For Video Mode (Face + Tone + Context)
**Primary Recommendations:**
1. **IEMOCAP** - Multimodal, conversational
2. **MELD** - TV dialogues with context
3. **CREMA-D** - Diverse actors, multimodal
4. **MEAD** - Intensity levels, multiple angles

### For Audio/Text Mode
**Primary Recommendations:**
1. **MELD** - Text + audio + context
2. **EmotionTalk** - Chinese dialogues
3. **IEMOCAP** - Rich transcripts
4. **CH-SIMS v2.0** - Non-verbal emphasis

### For LLM Text Training
**Primary Recommendations:**
1. **GoEmotions** - Large-scale text emotions
2. **EmoBank** - Valence/Arousal text
3. **EmotionLines** - Conversational emotions
4. **SemEval Emotion** - Standard benchmarks

---

## 📹 Multimodal Datasets (Video + Audio + Text)

### 1. **IEMOCAP** (Interactive Emotional Dyadic Motion Capture)
- **Modalities**: Video, Audio, Motion-capture, Transcripts
- **Size**: ~12 hours, dyadic sessions
- **Emotions**: 9 discrete labels (angry, happy, sad, excited, frustrated, etc.) + dimensional (valence/arousal/dominance)
- **Why**: Excellent for multimodal training, natural conversations, rich annotations
- **Link**: Papers with Code, USC website
- **Best For**: Video mode, context-aware processing

### 2. **MELD** (Multimodal EmotionLines Dataset)
- **Modalities**: Video, Audio, Text (TV series dialogues)
- **Size**: ~13,000 utterances from 1,433 dialogues
- **Emotions**: 7 emotions (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise) + sentiment
- **Why**: Conversation context, multi-party dialogues, TV series naturalness
- **Link**: GitHub, ArXiv
- **Best For**: Both video and audio/text modes, conversational context

### 3. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors)
- **Modalities**: Audio + Video
- **Size**: ~7,400 clips from 91 actors (diverse ethnicities)
- **Emotions**: 6 basic + neutral (happy, sad, anger, fear, disgust, neutral)
- **Why**: Diverse actors, good baseline, cross-cultural
- **Link**: TensorFlow Datasets
- **Best For**: Face + tone detection baseline

### 4. **MEAD** (Multimodal Emotional Audio-Visual Dataset)
- **Modalities**: Audio + Visual (talking-face videos)
- **Size**: ~60 actors, 8 emotions, 3 intensity levels, 7 camera angles
- **Emotions**: Angry, sad, happy, fear, surprise, disgust, etc. with intensities
- **Why**: Fine-grained intensity control, multiple viewpoints
- **Link**: MEAD project website
- **Best For**: Facial expression + tone with intensity

### 5. **CH-SIMS v2.0**
- **Modalities**: Visual + Acoustic + Text
- **Size**: ~2,121 refined segments + 10,161 unlabelled
- **Emotions**: Sentiment/emotion categories with non-verbal emphasis
- **Why**: Rich non-verbal cues, de-emphasizes text dominance
- **Link**: ArXiv
- **Best For**: Non-verbal emotion understanding

### 6. **EmotionTalk** (Chinese)
- **Modalities**: Audio, Visual, Text
- **Size**: ~23.6 hours, ~19,250 utterances from 19 actors
- **Emotions**: 7 categories + sentiment + speaking style
- **Why**: Cross-lingual, emotional style, Chinese dialogues
- **Link**: ArXiv
- **Best For**: Multilingual, style-aware processing

### 7. **EAV** (EEG-Audio-Video for Conversations)
- **Modalities**: EEG + Video + Audio (conversational)
- **Size**: 42 participants × 200 interactions = 8,400 clips
- **Emotions**: 5 emotions (neutral, anger, happiness, sadness, calmness)
- **Why**: Physiological signals, internal state recognition
- **Link**: GitHub, Nature Scientific Data
- **Best For**: Physiological integration (you have EEG models!)

### 8. **CAER / CAER-S** (Context-Aware Emotion Recognition)
- **Modalities**: Video (face + context)
- **Size**: ~13,000 videos + ~70,000 frames
- **Emotions**: 7 emotion categories
- **Why**: Face + background context modeling
- **Link**: CAER dataset website
- **Best For**: Context-aware video processing

### 9. **VEATIC** (Video-based Emotion and Affect Tracking in Context)
- **Modalities**: Video with context
- **Size**: 124 clips from movies/documentaries/home videos
- **Emotions**: Continuous valence/arousal ratings
- **Why**: Continuous tracking, scene context, character info
- **Link**: ArXiv
- **Best For**: Temporal emotion tracking

### 10. **FERV39k**
- **Modalities**: Video only
- **Size**: ~38,935 clips from ~4,000 videos across 22 scenes
- **Emotions**: 7 classic expressions
- **Why**: Large-scale, varied scenes, fully annotated
- **Link**: ArXiv
- **Best For**: Facial expression in varied contexts

---

## 🎤 Audio-Only Datasets

### 11. **EmoDB** (Berlin Emotional Speech Database)
- **Modalities**: Audio (German)
- **Size**: Multiple speakers
- **Emotions**: 6+ emotions + neutral
- **Why**: High-quality acted speech, German language
- **Link**: Wikipedia, research papers
- **Best For**: Audio emotion baseline

### 12. **UniData Speech Emotion Recognition**
- **Modalities**: Audio
- **Size**: 30,000+ clips
- **Emotions**: 4 emotions (euphoria, joy, sadness, surprise)
- **Why**: Large-scale, diverse speakers
- **Link**: HuggingFace, UniDataPro
- **Best For**: Large-scale audio training

### 13. **MSP-IMPROV**
- **Modalities**: Audio, Video, Transcripts
- **Size**: Similar to IEMOCAP
- **Emotions**: Multiple emotions
- **Why**: Complementary to IEMOCAP, natural conversations
- **Link**: Research papers
- **Best For**: Audio/text alignment

### 14. **EmoGator**
- **Modalities**: Audio (vocal bursts - no speech)
- **Size**: ~16.97 hours, 32,130 samples, 357 speakers
- **Emotions**: Non-speech emotional sounds
- **Why**: Captures laughs, cries, sighs - nonverbal emotion
- **Link**: ArXiv
- **Best For**: Non-verbal audio emotion

### 15. **English Speech Emotion Dataset** (NexData)
- **Modalities**: Audio + Transcripts
- **Size**: 20 native speakers, scripted monologues
- **Emotions**: 10 discrete emotions
- **Why**: Text + speech alignment, larger emotion set
- **Link**: NexData.ai
- **Best For**: Audio/text integration

---

## 📝 Text-Only Emotion Datasets (For LLM Training)

### 16. **GoEmotions**
- **Type**: Text emotions
- **Size**: 58,000 Reddit comments
- **Emotions**: 27 emotion categories + neutral
- **Why**: Large-scale, diverse emotions, real social media
- **Link**: GitHub, HuggingFace
- **Best For**: LLM text emotion training

### 17. **EmoBank**
- **Type**: Text with dimensional labels
- **Size**: 10,000 sentences
- **Emotions**: Valence, Arousal, Dominance (VAD)
- **Why**: Dimensional emotion modeling, continuous values
- **Link**: Research papers, GitHub
- **Best For**: Dimensional emotion LLM training

### 18. **EmotionLines**
- **Type**: Conversational text emotions
- **Size**: Friends TV series dialogues
- **Emotions**: 7 emotions + sentiment
- **Why**: Conversational context, dialogue flow
- **Link**: GitHub
- **Best For**: Conversational emotion LLM

### 19. **SemEval Emotion Dataset**
- **Type**: Text emotion classification
- **Size**: Standard benchmark size
- **Emotions**: Multiple emotion categories
- **Why**: Standard benchmark, well-established
- **Link**: SemEval website
- **Best For**: Benchmarking LLM emotion understanding

### 20. **EmotionX**
- **Type**: Conversational emotion in text
- **Size**: Multiple dialogue datasets
- **Emotions**: Emotion categories in conversations
- **Why**: Focus on conversational emotion shifts
- **Link**: Research papers
- **Best For**: Conversation-aware LLM

### 21. **Emotion-Stimulus Dataset**
- **Type**: Text with emotion triggers
- **Size**: Emotion-cause pairs
- **Emotions**: Emotion + cause identification
- **Why**: Understands what triggers emotions
- **Link**: Research papers
- **Best For**: Context-aware emotion LLM

### 22. **DailyDialog**
- **Type**: Daily conversation dialogues
- **Size**: 13,000+ dialogues
- **Emotions**: Emotion labels per utterance
- **Why**: Natural daily conversations, emotion flow
- **Link**: GitHub
- **Best For**: Real-world conversation LLM

---

## 🧠 Physiological / Contextual Datasets

### 23. **Mixed Emotion Recognition Dataset**
- **Modalities**: EEG, GSR, PPG, Face Video, Self-reports
- **Size**: Multiple participants
- **Emotions**: Mixed & non-mixed emotions
- **Why**: Physiological signals, internal vs external
- **Link**: Nature Scientific Data
- **Best For**: Physiological integration (you have EEG!)

### 24. **Emognition**
- **Modalities**: Video, Physiological, Self-reports
- **Size**: Full-HD video + wearable signals
- **Emotions**: Multiple emotions + intensity + arousal
- **Why**: Intensity estimates, action units, internal vs observed
- **Link**: Nature Scientific Data
- **Best For**: Intensity and internal state modeling

### 25. **AFFEC (2025)**
- **Modalities**: Face Video, EEG, Eye Tracking, GSR, Personality
- **Size**: 73 participants, 84 dialogues, >5,000 trials
- **Emotions**: 6 emotions, "felt" vs "perceived"
- **Why**: Personality traits, internal vs perceived emotion
- **Link**: ArXiv
- **Best For**: Personalization, personality-aware emotion

### 26. **EEV** (Evoked Expressions from Video)
- **Modalities**: Video + Viewer reactions
- **Size**: ~1,700 hours, 23,574 videos
- **Emotions**: Continuous viewer expressions
- **Why**: Reaction/empathy modeling, how content triggers emotion
- **Link**: GitHub, ArXiv
- **Best For**: Empathy and reaction modeling

### 27. **NFED** (Natural Facial Expressions Dataset)
- **Modalities**: Video (3-second clips)
- **Size**: Multiple clips
- **Emotions**: Emotion + gender + ethnicity
- **Why**: Natural expressions, ambiguous boundaries
- **Link**: Nature Scientific Data
- **Best For**: Natural expression modeling

---

## 📚 Dataset Collections / Repositories

### 28. **SER-datasets** (SuperKogito Collection)
- **Type**: Repository of ~77 speech emotion datasets
- **Why**: Comprehensive collection, multiple languages
- **Link**: GitHub
- **Best For**: Surveying available datasets

### 29. **Multilingual Speech Valence Classification**
- **Type**: Raw audio datasets for valence/arousal
- **Why**: Multilingual, continuous emotion modeling
- **Link**: GitHub
- **Best For**: Multilingual continuous emotion

---

## 🎯 Recommended Training Strategy

### Phase 1: Foundation (Start Here)
1. **IEMOCAP** - Multimodal foundation
2. **MELD** - Conversational context
3. **GoEmotions** - Text emotion baseline

### Phase 2: Specialization
4. **CREMA-D** - Diverse face + audio
5. **MEAD** - Intensity levels
6. **EmoBank** - Dimensional emotions

### Phase 3: Advanced Features
7. **EAV** - Physiological integration (EEG)
8. **AFFEC** - Personality + internal state
9. **EmotionTalk** - Multilingual

### Phase 4: Context & Refinement
10. **CAER** - Context-aware
11. **VEATIC** - Continuous tracking
12. **EmotionLines** - Conversational flow

---

## 📥 Where to Download

### Direct Links (Check Licenses!)
- **IEMOCAP**: USC website (requires registration)
- **MELD**: GitHub repositories
- **CREMA-D**: TensorFlow Datasets, HuggingFace
- **GoEmotions**: HuggingFace, GitHub
- **EmoBank**: Research paper repositories
- **EAV**: GitHub, Nature Scientific Data
- **MEAD**: Project website

### Common Platforms
- **HuggingFace Datasets**: Many emotion datasets
- **TensorFlow Datasets**: CREMA-D, others
- **Papers with Code**: Dataset links
- **GitHub**: Many open-source datasets
- **Research Paper Supplements**: Original sources

---

## ⚖️ Licensing Considerations

- **Academic Use**: Most datasets require academic/research use only
- **Commercial Use**: Check individual licenses
- **Attribution**: Most require citation
- **Registration**: Some require registration (IEMOCAP, etc.)

---

## 🔧 Integration Tips

1. **Start Small**: Begin with 2-3 datasets, expand gradually
2. **Balance Emotions**: Ensure balanced emotion distribution
3. **Modality Matching**: Match datasets to your pipeline modes
4. **Preprocessing**: Standardize formats across datasets
5. **Validation Split**: Reserve test sets for evaluation
6. **Data Augmentation**: Use augmentation for smaller datasets

---

## 📊 Dataset Comparison Table

| Dataset | Modalities | Size | Emotions | Best For |
|---------|-----------|------|----------|----------|
| IEMOCAP | V+A+T+MC | ~12h | 9+3D | Multimodal foundation |
| MELD | V+A+T | 13K utt | 7 | Conversation context |
| CREMA-D | V+A | 7.4K clips | 6+1 | Baseline, diversity |
| MEAD | V+A | 60 actors | 8×3 | Intensity levels |
| GoEmotions | T | 58K | 27 | Text LLM training |
| EAV | EEG+V+A | 8.4K | 5 | Physiological |
| AFFEC | V+EEG+ET+GSR | 5K+ | 6 | Personality |

---

## 🚀 Quick Start Recommendations

**For Video Mode:**
- Start: IEMOCAP + CREMA-D
- Add: MELD for context
- Enhance: MEAD for intensity

**For Audio/Text Mode:**
- Start: MELD + GoEmotions
- Add: EmoBank for dimensions
- Enhance: EmotionLines for conversation

**For LLM Training:**
- Start: GoEmotions (large-scale)
- Add: EmoBank (dimensions)
- Enhance: EmotionLines (conversation)

**For Physiological Integration:**
- Start: EAV (EEG + audio + video)
- Add: AFFEC (personality)
- Enhance: Mixed Emotion (internal state)

---

## 📝 Notes

- **Data Quality**: IEMOCAP and MELD are high-quality, well-annotated
- **Diversity**: CREMA-D and MEAD offer good diversity
- **Scale**: GoEmotions is large-scale for text
- **Context**: MELD and EmotionLines excel at context
- **Physiological**: EAV and AFFEC add internal state

Choose datasets based on:
1. Your pipeline mode (video vs audio/text)
2. Available computational resources
3. Desired emotion granularity
4. Need for physiological signals
5. Multilingual requirements

