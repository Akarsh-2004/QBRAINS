#!/usr/bin/env python3
"""
Test script for Sound Emotion Detection
"""

import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import warnings

warnings.filterwarnings('ignore')

class SoundEmotionDetector:
    """Sound Emotion Detection System"""
    
    def __init__(self, model_path="../model/sound_emotion_detector.keras"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Audio parameters
        self.sample_rate = 22050
        self.duration = 3
        self.n_mfcc = 40
        self.n_fft = 2048
        self.hop_length = 512
        self.max_pad_length = 173
        
        # Load model and components
        self.load_model_components()
    
    def load_model_components(self):
        """Load trained model and preprocessing components"""
        try:
            # Load model
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load scaler
            scaler_path = self.model_path.replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded from {scaler_path}")
            else:
                print(f"⚠️ Scaler not found at {scaler_path}")
            
            # Load label encoder
            encoder_path = self.model_path.replace('.keras', '_label_encoder.pkl')
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                print(f"✅ Label encoder loaded from {encoder_path}")
                print(f"   Classes: {list(self.label_encoder.classes_)}")
            else:
                print(f"⚠️ Label encoder not found at {encoder_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            return False
    
    def extract_features(self, file_path):
        """Extract audio features from file"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, duration=self.duration, sr=self.sample_rate)
            
            # Ensure consistent length
            if len(y) < self.max_pad_length:
                y = np.pad(y, (0, self.max_pad_length - len(y)), mode='constant')
            else:
                y = y[:self.max_pad_length]
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                       n_fft=self.n_fft, hop_length=self.hop_length)
            chroma = librosa.feature.chroma(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
            
            # Combine features
            combined = np.concatenate([
                mfccs, chroma, spectral_contrast, zcr, spectral_rolloff, rms
            ], axis=0)
            
            return combined
            
        except Exception as e:
            print(f"❌ Error extracting features from {file_path}: {e}")
            return None
    
    def predict_emotion(self, audio_path):
        """Predict emotion from audio file"""
        if not all([self.model, self.scaler, self.label_encoder]):
            print("❌ Model components not loaded properly")
            return None
        
        # Extract features
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        try:
            # Normalize features
            normalized = self.scaler.transform([features])
            
            # Reshape for model
            total_features = normalized.shape[1]
            n_time_steps = 173
            n_features_per_step = total_features // n_time_steps
            reshaped = normalized.reshape(1, n_time_steps, n_features_per_step)
            
            # Predict
            prediction = self.model.predict(reshaped, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Decode label
            emotion = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Get all probabilities
            all_probs = {
                self.label_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(prediction[0])
            }
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'all_probabilities': all_probs
            }
            
        except Exception as e:
            print(f"❌ Error predicting emotion: {e}")
            return None
    
    def batch_predict(self, audio_directory):
        """Predict emotions for all audio files in directory"""
        if not os.path.exists(audio_directory):
            print(f"❌ Directory not found: {audio_directory}")
            return []
        
        results = []
        audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]
        
        print(f"🎵 Processing {len(audio_files)} audio files...")
        
        for i, file in enumerate(audio_files, 1):
            file_path = os.path.join(audio_directory, file)
            prediction = self.predict_emotion(file_path)
            
            if prediction:
                result = {
                    'file': file,
                    'path': file_path,
                    'prediction': prediction
                }
                results.append(result)
                print(f"  {i}/{len(audio_files)}: {file} -> {prediction['emotion']} ({prediction['confidence']:.3f})")
            else:
                print(f"  {i}/{len(audio_files)}: {file} -> Failed to predict")
        
        return results

def main():
    """Main function to test sound emotion detection"""
    print("🎵 Sound Emotion Detection Test")
    print("=" * 40)
    
    # Initialize detector
    detector = SoundEmotionDetector()
    
    if not detector.model:
        print("❌ Cannot proceed without trained model")
        print("Please run sound_emotion_detector.ipynb first to train the model")
        return
    
    # Test options
    print("\nTest Options:")
    print("1. Test single audio file")
    print("2. Test directory of audio files")
    print("3. Test with sample from dataset")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        # Test single file
        file_path = input("Enter audio file path: ").strip()
        if os.path.exists(file_path):
            prediction = detector.predict_emotion(file_path)
            if prediction:
                print(f"\n🎯 Prediction Results:")
                print(f"   Emotion: {prediction['emotion']}")
                print(f"   Confidence: {prediction['confidence']:.4f}")
                print(f"\n📊 All Probabilities:")
                for emotion, prob in sorted(prediction['all_probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True):
                    print(f"   {emotion}: {prob:.4f}")
            else:
                print("❌ Failed to predict emotion")
        else:
            print("❌ File not found")
    
    elif choice == '2':
        # Test directory
        dir_path = input("Enter directory path: ").strip()
        results = detector.batch_predict(dir_path)
        
        if results:
            # Summary statistics
            emotions = [r['prediction']['emotion'] for r in results]
            unique_emotions, counts = np.unique(emotions, return_counts=True)
            
            print(f"\n📊 Summary:")
            print(f"   Total files processed: {len(results)}")
            print(f"   Emotions detected:")
            for emotion, count in zip(unique_emotions, counts):
                percentage = (count / len(results)) * 100
                print(f"     {emotion}: {count} ({percentage:.1f}%)")
    
    elif choice == '3':
        # Test with dataset sample
        data_dir = "../data/sound_data/TESS Toronto emotional speech set data"
        if os.path.exists(data_dir):
            # Find a sample file
            for folder in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, folder)):
                    folder_path = os.path.join(data_dir, folder)
                    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                    if files:
                        sample_file = os.path.join(folder_path, files[0])
                        print(f"\n🎵 Testing sample: {sample_file}")
                        
                        prediction = detector.predict_emotion(sample_file)
                        if prediction:
                            print(f"🎯 Predicted: {prediction['emotion']} ({prediction['confidence']:.4f})")
                            
                            # Extract true emotion from folder name
                            true_emotion = folder.split('_')[-1]
                            if true_emotion == 'pleasant_surprise' or true_emotion == 'pleasant_surprised':
                                true_emotion = 'surprise'
                            
                            print(f"✅ True emotion: {true_emotion}")
                            print(f"{'✅ Correct!' if prediction['emotion'] == true_emotion else '❌ Incorrect'}")
                        break
        else:
            print("❌ Dataset directory not found")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
