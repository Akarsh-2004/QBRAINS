#!/usr/bin/env python3
"""
LLM Emotion Training Script
Trains an emotion-aware LLM using GoEmotions and IEMOCAP datasets
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Deep Learning
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Define data paths
BASE_DATA_PATH = Path("data/NLP for emotions")
EMOTIONS_TXT = BASE_DATA_PATH / "data/data/emotions.txt"

def load_goemotions_txt(file_path):
    """Load GoEmotions data from TXT format (text;emotion)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ';' in line:
                text, emotion = line.rsplit(';', 1)
                data.append({'text': text.strip(), 'emotion': emotion.strip()})
    return pd.DataFrame(data)

def load_goemotions_csv(file_path):
    """Load GoEmotions data from CSV format (text,label)"""
    df = pd.read_csv(file_path)
    
    # Load emotion mapping
    emotion_map = {0: 'neutral'}
    if EMOTIONS_TXT.exists():
        with open(EMOTIONS_TXT, 'r') as f:
            emotion_names = [line.strip() for line in f]
            for i, name in enumerate(emotion_names, 1):
                emotion_map[i] = name
    
    # Map numeric labels to emotion names
    df['emotion'] = df['label'].map(emotion_map).fillna('neutral')
    return df[['text', 'emotion']]

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = ' '.join(text.split())
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("=" * 60)
    print("LLM Emotion Training")
    print("=" * 60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    all_dataframes = []
    
    # GoEmotions TXT
    train_txt_path = BASE_DATA_PATH / "train.txt"
    test_txt_path = BASE_DATA_PATH / "test.txt"
    val_txt_path = BASE_DATA_PATH / "val.txt"
    
    if train_txt_path.exists():
        print("  Loading GoEmotions TXT...")
        train_txt = load_goemotions_txt(train_txt_path)
        test_txt = load_goemotions_txt(test_txt_path) if test_txt_path.exists() else pd.DataFrame()
        val_txt = load_goemotions_txt(val_txt_path) if val_txt_path.exists() else pd.DataFrame()
        all_dataframes.extend([train_txt, test_txt, val_txt])
        print(f"    Loaded: {len(train_txt)} train, {len(test_txt)} test, {len(val_txt)} val")
    
    # GoEmotions CSV
    train_csv_path = BASE_DATA_PATH / "data 2/training.csv"
    if train_csv_path.exists():
        print("  Loading GoEmotions CSV...")
        train_csv = load_goemotions_csv(train_csv_path)
        test_csv = load_goemotions_csv(BASE_DATA_PATH / "data 2/test.csv") if (BASE_DATA_PATH / "data 2/test.csv").exists() else pd.DataFrame()
        val_csv = load_goemotions_csv(BASE_DATA_PATH / "data 2/validation.csv") if (BASE_DATA_PATH / "data 2/validation.csv").exists() else pd.DataFrame()
        all_dataframes.extend([train_csv, test_csv, val_csv])
        print(f"    Loaded: {len(train_csv)} train, {len(test_csv)} test, {len(val_csv)} val")
    
    # Combine
    if not all_dataframes:
        print("❌ No data found! Check data paths.")
        return
    
    combined_df = pd.concat([df for df in all_dataframes if len(df) > 0], ignore_index=True)
    print(f"\n  Total samples: {len(combined_df)}")
    print(f"  Unique emotions: {combined_df['emotion'].nunique()}")
    
    # Preprocess
    print("\n2. Preprocessing data...")
    combined_df['text'] = combined_df['text'].apply(preprocess_text)
    combined_df = combined_df[combined_df['text'].str.len() > 0]
    combined_df = combined_df[
        (combined_df['text'].str.len() >= 10) & 
        (combined_df['text'].str.len() <= 512)
    ]
    print(f"  After preprocessing: {len(combined_df)} samples")
    
    # Encode emotions
    label_encoder = LabelEncoder()
    combined_df['emotion_id'] = label_encoder.fit_transform(combined_df['emotion'])
    num_labels = len(label_encoder.classes_)
    print(f"  Number of emotion classes: {num_labels}")
    
    # Split data
    print("\n3. Splitting data...")
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=combined_df['emotion_id']
    )
    test_df, val_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['emotion_id']
    )
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize model
    print("\n4. Initializing model...")
    MODEL_NAME = "distilbert-base-uncased"  # Fast and efficient
    print(f"  Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    print("\n5. Creating datasets...")
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        train_df['emotion_id'].tolist(),
        tokenizer,
        max_length=128
    )
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        val_df['emotion_id'].tolist(),
        tokenizer,
        max_length=128
    )
    test_dataset = EmotionDataset(
        test_df['text'].tolist(),
        test_df['emotion_id'].tolist(),
        tokenizer,
        max_length=128
    )
    
    # Training arguments
    print("\n6. Configuring training...")
    training_args = TrainingArguments(
        output_dir='model/emotion_llm_checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='model/emotion_llm_logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n7. Training model...")
    print("  This may take a while...")
    train_result = trainer.train()
    print(f"  ✅ Training completed! Loss: {train_result.training_loss:.4f}")
    
    # Save model
    print("\n8. Saving model...")
    trainer.save_model('model/emotion_llm_final')
    tokenizer.save_pretrained('model/emotion_llm_final')
    joblib.dump(label_encoder, 'model/emotion_label_encoder.pkl')
    print("  ✅ Model saved to model/emotion_llm_final")
    print("  ✅ Label encoder saved to model/emotion_label_encoder.pkl")
    
    # Evaluate
    print("\n9. Evaluating on test set...")
    eval_results = trainer.evaluate(test_dataset)
    print("  Test Results:")
    for key, value in eval_results.items():
        if 'loss' not in key:
            print(f"    {key}: {value:.4f}")
    
    # Save model info
    model_info = {
        'model_name': MODEL_NAME,
        'num_labels': num_labels,
        'emotion_classes': list(label_encoder.classes_),
        'training_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'test_accuracy': eval_results.get('eval_accuracy', 0),
        'test_f1': eval_results.get('eval_f1', 0)
    }
    
    with open('model/emotion_llm_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("  ✅ Model info saved to model/emotion_llm_info.json")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: model/emotion_llm_final")
    print(f"Test Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"Test F1: {eval_results.get('eval_f1', 0):.4f}")

if __name__ == "__main__":
    main()

