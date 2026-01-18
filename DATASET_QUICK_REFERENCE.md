# Dataset Quick Reference - Top Recommendations

## 🎯 Top 10 Must-Have Datasets

### 1. **IEMOCAP** ⭐⭐⭐⭐⭐
- **Why**: Best multimodal dataset, natural conversations
- **Use**: Video mode foundation
- **Get**: USC website (registration required)

### 2. **MELD** ⭐⭐⭐⭐⭐
- **Why**: TV dialogues, rich context, text+audio+video
- **Use**: Both modes, conversational context
- **Get**: GitHub repositories

### 3. **GoEmotions** ⭐⭐⭐⭐⭐
- **Why**: Large-scale text emotions (58K samples)
- **Use**: LLM text training
- **Get**: HuggingFace, GitHub

### 4. **CREMA-D** ⭐⭐⭐⭐
- **Why**: Diverse actors, good baseline
- **Use**: Face + tone detection
- **Get**: TensorFlow Datasets

### 5. **MEAD** ⭐⭐⭐⭐
- **Why**: Intensity levels, multiple angles
- **Use**: Fine-grained expression training
- **Get**: MEAD project website

### 6. **EAV** ⭐⭐⭐⭐
- **Why**: EEG + Audio + Video (you have EEG models!)
- **Use**: Physiological integration
- **Get**: GitHub, Nature Scientific Data

### 7. **EmoBank** ⭐⭐⭐⭐
- **Why**: Dimensional emotions (VAD)
- **Use**: LLM dimensional training
- **Get**: Research repositories

### 8. **EmotionLines** ⭐⭐⭐
- **Why**: Conversational emotions
- **Use**: Conversation-aware LLM
- **Get**: GitHub

### 9. **AFFEC** ⭐⭐⭐⭐
- **Why**: Personality + internal state
- **Use**: Personalization, felt vs perceived
- **Get**: ArXiv (2025)

### 10. **CAER** ⭐⭐⭐
- **Why**: Context-aware (face + background)
- **Use**: Context modeling
- **Get**: CAER dataset website

---

## 📋 By Use Case

### Video Mode Training
1. IEMOCAP
2. CREMA-D
3. MELD
4. MEAD
5. CAER

### Audio/Text Mode Training
1. MELD
2. GoEmotions
3. EmotionLines
4. EmoBank
5. MSP-IMPROV

### LLM Text Training
1. GoEmotions (large-scale)
2. EmoBank (dimensions)
3. EmotionLines (conversation)
4. SemEval (benchmark)
5. DailyDialog (daily conversations)

### Physiological Integration
1. EAV (EEG + audio + video)
2. AFFEC (personality + internal)
3. Mixed Emotion (internal vs external)
4. Emognition (intensity + internal)

### Context-Aware Training
1. MELD (conversation context)
2. CAER (scene context)
3. VEATIC (temporal context)
4. EmotionLines (dialogue context)

---

## 🚀 Quick Start (Minimum Viable)

**For Video Mode:**
- IEMOCAP (multimodal foundation)
- CREMA-D (diversity)

**For Audio/Text Mode:**
- MELD (conversation + context)
- GoEmotions (text scale)

**For LLM Training:**
- GoEmotions (text emotions)
- EmoBank (dimensions)

**Total: 4-5 datasets to get started**

---

## 📊 Dataset Sizes (Quick Comparison)

| Dataset | Size | Type |
|---------|------|------|
| GoEmotions | 58K | Text |
| MELD | 13K | Multimodal |
| CREMA-D | 7.4K | Audio+Video |
| IEMOCAP | ~12h | Multimodal |
| EAV | 8.4K | EEG+Audio+Video |
| MEAD | 60 actors | Audio+Video |
| EmoBank | 10K | Text |

---

## 🔗 Quick Links

- **HuggingFace**: GoEmotions, CREMA-D, many others
- **TensorFlow Datasets**: CREMA-D
- **GitHub**: MELD, EmotionLines, EAV, many repos
- **Papers with Code**: Dataset links
- **Research Papers**: Original sources (check supplements)

---

## 💡 Pro Tips

1. **Start with IEMOCAP + MELD** - Best multimodal foundation
2. **Add GoEmotions** - Essential for text LLM training
3. **Use CREMA-D** - Good diversity baseline
4. **Consider EAV** - You already have EEG models!
5. **Balance emotions** - Ensure good distribution
6. **Check licenses** - Most are academic-only
7. **Preprocess consistently** - Standardize formats
8. **Reserve test sets** - Don't mix train/test

---

## 📝 Download Checklist

- [ ] IEMOCAP (registration required)
- [ ] MELD (GitHub)
- [ ] GoEmotions (HuggingFace)
- [ ] CREMA-D (TensorFlow/HuggingFace)
- [ ] MEAD (project website)
- [ ] EAV (GitHub/Nature)
- [ ] EmoBank (research repos)
- [ ] EmotionLines (GitHub)

---

**See `DATASETS_FOR_LLM_TRAINING.md` for complete details!**

