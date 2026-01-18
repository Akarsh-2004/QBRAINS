#!/usr/bin/env python3
"""
Audio/Text Processing Module for Quantum Emotion Pipeline
Processes audio files and text inputs for emotion detection
"""

import librosa
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tensorflow.keras.models import load_model
import joblib
import os
import re
from collections import Counter
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

from src.utils import get_model_path, load_config, validate_model_path
from src.audio_features import AudioFeatureExtractor
from src.custom_layers import CustomDense, CustomInputLayer
from src.performance_optimizer import cached, timed, create_audio_cache_key, create_emotion_cache_key
from src.safe_model_loader import safe_load_label_encoder
from src.fallback_emotion_detector import get_fallback_emotion


class AudioTextProcessor:
    """Process audio files and text for emotion detection"""
    
    def __init__(self,
                 audio_model_path: Optional[str] = None,
                 audio_scaler_path: Optional[str] = None,
                 audio_encoder_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize audio/text processor
        
        Args:
            audio_model_path: Path to audio emotion model (optional, uses config if None)
            audio_scaler_path: Path to audio feature scaler (optional)
            audio_encoder_path: Path to audio label encoder (optional)
            config: Configuration dictionary (optional)
        """
        self.config = config or load_config()
        self.audio_model = None
        self.audio_scaler = None
        self.audio_label_encoder = None
        
        # Resolve model paths
        if audio_model_path is None:
            audio_model_path = str(get_model_path(self.config['models']['audio_model']))
        else:
            audio_model_path = str(get_model_path(audio_model_path))
        
        if audio_scaler_path is None:
            audio_scaler_path = str(get_model_path(self.config['models']['audio_scaler']))
        else:
            audio_scaler_path = str(get_model_path(audio_scaler_path))
        
        if audio_encoder_path is None:
            audio_encoder_path = str(get_model_path(self.config['models']['audio_encoder']))
        else:
            audio_encoder_path = str(get_model_path(audio_encoder_path))
        
        # Audio processing parameters from config
        audio_config = self.config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 22050)
        self.duration = audio_config.get('duration', 3)
        self.n_mfcc = audio_config.get('n_mfcc', 40)
        self.n_fft = audio_config.get('n_fft', 2048)
        self.hop_length = audio_config.get('hop_length', 512)
        self.max_pad_length = audio_config.get('max_pad_length', 173)
        self.n_time_steps = audio_config.get('n_time_steps', 173)
        self.n_features_per_step = audio_config.get('n_features_per_step', 62)
        
        # Audio feature extractor
        self.audio_feature_extractor = AudioFeatureExtractor(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            max_pad_length=self.max_pad_length
        )
        
        # Text emotion keywords
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'glad', 'pleased', 'delighted', 'cheerful', 'ecstatic'],
            'sad': ['sad', 'depressed', 'down', 'unhappy', 'melancholy', 'gloomy', 'sorrowful', 'upset'],
            'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'livid'],
            'fear': ['fear', 'afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified', 'panic'],
            'surprise': ['surprise', 'surprised', 'shocked', 'amazed', 'astonished', 'wow', 'unexpected'],
            'disgust': ['disgust', 'disgusted', 'revolted', 'sickened', 'repulsed', 'nauseated'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular', 'standard']
        }
        
        # Performance optimization: add caching
        self.feature_cache = {}
        self.cache_size_limit = 100
        self.audio_thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Load models
        self._load_models(audio_model_path, audio_scaler_path, audio_encoder_path)
    
    def _load_models(self, audio_path, scaler_path, encoder_path):
        """Load audio models"""
        try:
            from src.safe_model_loader import safe_load_keras_model
            
            audio_path_obj = Path(audio_path)
            if validate_model_path(audio_path_obj, "Audio model"):
                self.audio_model = safe_load_keras_model(str(audio_path_obj), compile_model=False)
                if self.audio_model:
                    print(f"✓ Audio emotion model loaded: {audio_path_obj}")
            
            scaler_path_obj = Path(scaler_path)
            if validate_model_path(scaler_path_obj, "Audio scaler"):
                self.audio_scaler = joblib.load(str(scaler_path_obj))
                print(f"✓ Audio scaler loaded: {scaler_path_obj}")
            
            encoder_path_obj = Path(encoder_path)
            if validate_model_path(encoder_path_obj, "Audio label encoder"):
                self.audio_label_encoder = safe_load_label_encoder(str(encoder_path_obj))
                if self.audio_label_encoder:
                    print(f"✓ Audio label encoder loaded: {encoder_path_obj}")
                else:
                    print(f"⚠️ Using fallback label encoder")
                    # Create fallback encoder
                    from sklearn.preprocessing import LabelEncoder
                    self.audio_label_encoder = LabelEncoder()
                    self.audio_label_encoder.classes_ = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    @cached(ttl_seconds=1800)
    @timed("audio_data_processing")
    def process_audio_data(self, audio_data: np.ndarray, sample_rate: Optional[int] = None) -> Dict[str, Any]:
        """
        Process raw audio data array for emotion detection (optimized)
        
        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Sample rate (defaults to self.sample_rate)
            
        Returns:
            Dictionary with emotion analysis
        """
        try:
            sr = sample_rate or self.sample_rate
            
            # Create cache key for audio data
            audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
            cache_key = f"{audio_hash}_{sr}"
            
            # Check cache first
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            # Resample if needed (optimized)
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            # Trim or pad to duration (vectorized)
            target_length = int(self.sample_rate * self.duration)
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            elif len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
            
            y = audio_data
            sr = self.sample_rate
            
            # Extract features using shared extractor
            features = self.audio_feature_extractor.extract_features(y, sr)
            
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Combine features
            combined_features = self.audio_feature_extractor.combine_features(features)
            
            # Normalize
            if self.audio_scaler:
                normalized = self.audio_scaler.transform([combined_features])
            else:
                normalized = [combined_features]
            
            # Reshape for model
            reshaped = self.audio_feature_extractor.reshape_for_model(
                normalized[0],
                n_time_steps=self.n_time_steps,
                n_features_per_step=self.n_features_per_step
            )
            
            # Predict
            if self.audio_model:
                prediction = self.audio_model.predict(reshaped, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Decode emotion
                if self.audio_label_encoder:
                    emotion = self.audio_label_encoder.inverse_transform([predicted_class])[0]
                else:
                    emotion = f"emotion_{predicted_class}"
                
                # Create emotion distribution
                emotion_dist = {}
                if self.audio_label_encoder:
                    for i, prob in enumerate(prediction):
                        emo_name = self.audio_label_encoder.inverse_transform([i])[0]
                        emotion_dist[emo_name] = float(prob)
                else:
                    emotion_dist[emotion] = float(confidence)
            else:
                # No model available, use fallback
                features_dict = {
                    'energy': features.get('energy', [0.5]),
                    'tempo': features.get('tempo', [90]),
                    'pitch': features.get('pitch', [0.5])
                }
                fallback_result = get_fallback_emotion(audio_features=features_dict)
                return fallback_result
            
            result = {
                'dominant_emotion': emotion,
                'confidence': float(confidence),
                'emotion_distribution': emotion_dist
            }
            
            # Cache the result (limit cache size)
            if len(self.feature_cache) < self.cache_size_limit:
                self.feature_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    @cached(ttl_seconds=1800)
    @timed("audio_file_processing")
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Process audio file for emotion detection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion analysis
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=self.duration, sr=self.sample_rate)
            
            # Extract features using shared extractor
            features = self.audio_feature_extractor.extract_features(y, sr)
            
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Combine features
            combined_features = self.audio_feature_extractor.combine_features(features)
            
            # Normalize
            if self.audio_scaler:
                normalized = self.audio_scaler.transform([combined_features])
            else:
                normalized = [combined_features]
            
            # Reshape for model
            reshaped = self.audio_feature_extractor.reshape_for_model(
                normalized[0],
                n_time_steps=self.n_time_steps,
                n_features_per_step=self.n_features_per_step
            )
            
            # Predict
            if self.audio_model:
                prediction = self.audio_model.predict(reshaped, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Decode emotion
                if self.audio_label_encoder:
                    emotion = self.audio_label_encoder.inverse_transform([predicted_class])[0]
                else:
                    emotion = f"emotion_{predicted_class}"
                
                # Create emotion distribution
                emotion_dist = {}
                if self.audio_label_encoder:
                    for i, prob in enumerate(prediction):
                        emo_name = self.audio_label_encoder.inverse_transform([i])[0]
                        emotion_dist[emo_name] = float(prob)
                else:
                    emotion_dist[emotion] = float(confidence)
            else:
                emotion_dist = {}
                emotion = "unknown"
                confidence = 0.0
            
            # Tone analysis
            tone_analysis = {
                'dominant_emotion': emotion,
                'confidence': float(confidence),
                'emotion_distribution': emotion_dist,
                'audio_features': {
                    'rms_energy': float(np.mean(features['rms'])),
                    'zero_crossing_rate': float(np.mean(features['zcr'])),
                    'spectral_rolloff': float(np.mean(features['spectral_rolloff'])),
                    'pitch_mean': float(np.mean(features['chroma'])),
                    'spectral_contrast_mean': float(np.mean(features['spectral_contrast']))
                }
            }
            
            return tone_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    @cached(ttl_seconds=600)  # 10 minutes cache for text
    @timed("text_processing")
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text for emotion detection
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion analysis
        """
        # Use fallback emotion detector for better results
        try:
            fallback_result = get_fallback_emotion(text=text)
            if fallback_result and fallback_result.get('confidence', 0) > 0.3:
                return fallback_result
        except Exception as e:
            print(f"⚠️ Fallback text emotion detection failed: {e}")
        
        # Fallback to basic keyword matching
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Normalize to neutral/happy for very short greeting texts
        if len(text) < 10 and text_lower.strip() in ['hi', 'hello', 'hey', 'hiya', 'sup']:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.9,
                'emotion_distribution': {'neutral': 0.8, 'happy': 0.2},
                'sentiment': 'neutral',
                'text_features': {},
                'method': 'greeting_heuristic'
            }

        # Count emotion keyword matches
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = count
        
        # Calculate probabilities
        total_matches = sum(emotion_scores.values())
        if total_matches > 0:
            emotion_probs = {
                emo: score / total_matches 
                for emo, score in emotion_scores.items()
            }
        else:
            # Default to neutral if no matches
            emotion_probs = {emo: 0.0 for emo in self.emotion_keywords.keys()}
            emotion_probs['neutral'] = 1.0
        
        # Sentiment analysis (simple)
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Find dominant emotion
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        
        # Text features
        text_features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'capitalized_words': sum(1 for word in text.split() if word[0].isupper() if word) / len(text.split()) if text.split() else 0
        }
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_probs,
            'sentiment': sentiment,
            'sentiment_scores': {
                'positive': positive_count,
                'negative': negative_count
            },
            'text_features': text_features,
            'confidence': emotion_probs[dominant_emotion]
        }
    
    def process_audio_text(self, audio_path: Optional[str] = None,
                          text: Optional[str] = None) -> Dict[str, Any]:
        """
        Process both audio and text, combining results
        
        Args:
            audio_path: Path to audio file (optional)
            text: Text input (optional)
            
        Returns:
            Combined emotion analysis
        """
        results = {
            'audio_emotion': None,
            'text_emotion': None,
            'combined_emotion': None
        }
        
        # Process audio if provided
        if audio_path:
            results['audio_emotion'] = self.process_audio(audio_path)
        
        # Process text if provided
        if text:
            results['text_emotion'] = self.process_text(text)
        
        # Combine results
        if results['audio_emotion'] and results['text_emotion']:
            results['combined_emotion'] = self._combine_emotions(
                results['audio_emotion'],
                results['text_emotion']
            )
        elif results['audio_emotion']:
            results['combined_emotion'] = results['audio_emotion']
        elif results['text_emotion']:
            results['combined_emotion'] = results['text_emotion']
        
        return results
    
    def _combine_emotions(self, audio_emotion: Dict, text_emotion: Dict) -> Dict[str, Any]:
        """Combine audio and text emotion results"""
        # Get emotion distributions
        audio_dist = audio_emotion.get('emotion_distribution', {})
        text_dist = text_emotion.get('emotion_distribution', {})
        
        # Normalize emotion names (handle case differences)
        normalized_audio = {}
        normalized_text = {}
        
        for emo, prob in audio_dist.items():
            emo_lower = emo.lower()
            normalized_audio[emo_lower] = normalized_audio.get(emo_lower, 0) + prob
        
        for emo, prob in text_dist.items():
            normalized_text[emo.lower()] = prob
        
        # Combine with weights (audio 60%, text 40%)
        combined_dist = {}
        all_emotions = set(list(normalized_audio.keys()) + list(normalized_text.keys()))
        
        for emotion in all_emotions:
            audio_prob = normalized_audio.get(emotion, 0)
            text_prob = normalized_text.get(emotion, 0)
            combined_dist[emotion] = (audio_prob * 0.6) + (text_prob * 0.4)
        
        # Normalize
        total = sum(combined_dist.values())
        if total > 0:
            combined_dist = {emo: prob / total for emo, prob in combined_dist.items()}
        
        # Find dominant
        dominant = max(combined_dist, key=combined_dist.get)
        
        return {
            'dominant_emotion': dominant,
            'emotion_distribution': combined_dist,
            'confidence': combined_dist[dominant],
            'audio_confidence': audio_emotion.get('confidence', 0),
            'text_confidence': text_emotion.get('confidence', 0)
        }
    


def test_audio_text_processor():
    """Test audio/text processor"""
    print("🎤 Testing Audio/Text Processor")
    print("=" * 50)
    
    processor = AudioTextProcessor()
    
    # Test text processing
    print("\n1. Testing text processing...")
    text_result = processor.process_text("I'm so happy and excited about this!")
    print(f"Text: 'I'm so happy and excited about this!'")
    print(f"Dominant emotion: {text_result['dominant_emotion']}")
    print(f"Confidence: {text_result['confidence']:.3f}")
    
    print("\n✓ Audio/text processor initialized")


if __name__ == "__main__":
    test_audio_text_processor()

