# LLM Emotion Training Guide

Complete guide for training your emotion-aware LLM using the datasets in `data/NLP for emotions/`.

## 📋 Overview

This training process uses:
- **GoEmotions**: 58K samples with 27 emotion categories
- **IEMOCAP**: Multimodal emotion dataset
- **Custom emotion data**: Your prepared datasets

The trained model will be used in the quantum emotion pipeline for:
- Emotion prediction from text
- Text reframing with emotional context
- Context-aware emotion understanding

## 🚀 Quick Start

### Option 1: Python Script (Recommended)

```bash
# Install dependencies
pip install transformers datasets torch scikit-learn

# Run training
python train_emotion_llm.py
```

### Option 2: Jupyter Notebook

```bash
# Open notebook
jupyter notebook notebooks/llm_emotion_training.ipynb

# Run all cells
```

## 📁 Data Structure

Your data should be in:
```
data/NLP for emotions/
├── train.txt              # GoEmotions TXT format (text;emotion)
├── test.txt
├── val.txt
├── data 2/
│   ├── training.csv      # GoEmotions CSV format (text,label)
│   ├── test.csv
│   └── validation.csv
├── iemocap_full_dataset.csv
└── data/data/
    └── emotions.txt      # Emotion label names
```

## 🔧 Training Configuration

### Model Options

The script uses `distilbert-base-uncased` by default (fast and efficient). You can modify in the script:

```python
MODEL_NAME = "distilbert-base-uncased"  # Default
# Alternatives:
# - "bert-base-uncased" (larger, more accurate)
# - "roberta-base" (better performance)
# - "distilroberta-base" (balanced)
```

### Training Parameters

Default settings (can be modified in script):
- **Epochs**: 3
- **Batch Size**: 16
- **Max Length**: 128 tokens
- **Learning Rate**: Auto (from model)
- **Warmup Steps**: 500

## 📊 Training Process

1. **Data Loading**: Loads all available datasets
2. **Preprocessing**: Cleans and normalizes text
3. **Encoding**: Maps emotions to numeric labels
4. **Splitting**: 80% train, 10% val, 10% test
5. **Training**: Fine-tunes transformer model
6. **Evaluation**: Tests on held-out test set
7. **Saving**: Saves model, tokenizer, and label encoder

## 📤 Output Files

After training, you'll have:

```
model/
├── emotion_llm_final/          # Trained model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── emotion_label_encoder.pkl   # Label encoder
├── emotion_llm_info.json       # Model metadata
├── emotion_llm_checkpoints/    # Training checkpoints
└── emotion_llm_logs/           # Training logs
```

## 🧪 Testing the Model

After training, test the model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

# Load model
model = AutoModelForSequenceClassification.from_pretrained('model/emotion_llm_final')
tokenizer = AutoTokenizer.from_pretrained('model/emotion_llm_final')
label_encoder = joblib.load('model/emotion_label_encoder.pkl')

# Predict
text = "I'm feeling so happy and excited!"
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = torch.argmax(predictions, dim=-1).item()
    emotion = label_encoder.inverse_transform([predicted_id])[0]
    confidence = predictions[0][predicted_id].item()

print(f"Emotion: {emotion} (confidence: {confidence:.3f})")
```

## 🔗 Integration with Quantum Pipeline

The trained model can be integrated into your quantum pipeline:

```python
# In src/ollama_llm.py or quantum_pipeline.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

class EmotionLLM:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'model/emotion_llm_final'
        )
        self.tokenizer = AutoTokenizer.from_pretrained('model/emotion_llm_final')
        self.label_encoder = joblib.load('model/emotion_label_encoder.pkl')
    
    def predict_emotion(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(predictions, dim=-1).item()
            emotion = self.label_encoder.inverse_transform([predicted_id])[0]
            confidence = predictions[0][predicted_id].item()
        return emotion, confidence
```

## 📈 Expected Performance

With GoEmotions dataset:
- **Accuracy**: ~60-70% (27 classes is challenging)
- **F1-Score**: ~0.60-0.70
- **Training Time**: 1-3 hours (depending on GPU)

## 🔧 Troubleshooting

### Out of Memory
- Reduce batch size: `per_device_train_batch_size=8`
- Use smaller model: `distilbert-base-uncased`
- Reduce max_length: `max_length=64`

### Slow Training
- Use GPU: `CUDA_VISIBLE_DEVICES=0 python train_emotion_llm.py`
- Reduce epochs: `num_train_epochs=2`
- Use smaller model

### Poor Performance
- Train for more epochs: `num_train_epochs=5`
- Use larger model: `bert-base-uncased`
- Check data quality and balance

## 📚 Next Steps

1. **Train the model**: Run `python train_emotion_llm.py`
2. **Evaluate**: Check test set performance
3. **Integrate**: Use in quantum pipeline
4. **Fine-tune**: Adjust hyperparameters as needed
5. **Deploy**: Use with Ollama or directly

## 🎯 Tips

- **Start small**: Use `distilbert-base-uncased` for faster iteration
- **Monitor training**: Check logs in `model/emotion_llm_logs/`
- **Save checkpoints**: Best model is saved automatically
- **Balance data**: Ensure good emotion distribution
- **Validate early**: Check validation performance during training

## 📝 Notes

- The model is saved in HuggingFace format
- Can be converted to ONNX for faster inference
- Can be fine-tuned further on domain-specific data
- Compatible with Ollama through conversion (if needed)

---

**Ready to train?** Run `python train_emotion_llm.py` and let it train!

