#!/usr/bin/env python3
"""
EEG Processing Module for Quantum Emotion Pipeline
Processes EEG signals to extract emotion-related features
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Reshape, LSTM, Dense
from tensorflow.keras.regularizers import l2
from scipy.signal import butter, filtfilt
from pathlib import Path
import os

from src.custom_layers import CustomDense, CustomInputLayer


class EEGProcessor:
    """Process EEG signals to extract emotion information"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 input_shape: Tuple[int, int, int] = (760, 775, 1)):
        """
        Initialize EEG processor
        
        Args:
            model_path: Path to trained EEG model (optional)
            input_shape: Expected input shape (height, width, channels)
        """
        self.model = None
        self.input_shape = input_shape
        self.emotion_mapping = {
            0: 'neutral',  # Low arousal
            1: 'happy',    # High positive valence
            2: 'sad',      # Low negative valence
            3: 'angry',   # High negative valence
            4: 'fear',    # High arousal, negative
            5: 'surprise', # High arousal
            6: 'disgust'   # Negative valence
        }
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Create default model architecture
            self._create_default_model()
    
    def _create_default_model(self):
        """Create default EEG model architecture"""
        self.model = Sequential([
            Conv2D(
                filters=16,
                kernel_size=3,
                strides=(2, 2),
                activation='relu',
                input_shape=self.input_shape,
                padding='same',
                kernel_regularizer=l2(0.001)
            ),
            MaxPool2D(2, 2),
            BatchNormalization(),
            Dropout(0.5),
            Reshape((190, 194 * 16)),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(7, activation='softmax')  # 7 emotion classes
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("⚠️ Using default EEG model (not trained). Train a model for better results.")
    
    def _load_model(self, model_path: str):
        """Load trained EEG model"""
        try:
            from src.safe_model_loader import safe_load_keras_model
            
            self.model = safe_load_keras_model(model_path, compile_model=False)
            if self.model:
                print(f"✓ EEG model loaded: {model_path}")
            else:
                print(f"⚠️ Failed to load EEG model, using default")
                self._create_default_model()
        except Exception as e:
            print(f"⚠️ Error loading EEG model: {e}")
            self._create_default_model()
    
    def preprocess_eeg(self, eeg_data: np.ndarray, 
                      sample_rate: int = 256,
                      lowcut: float = 1.0,
                      highcut: float = 50.0) -> np.ndarray:
        """
        Preprocess EEG data
        
        Args:
            eeg_data: Raw EEG data (channels x time_steps)
            sample_rate: Sampling rate in Hz
            lowcut: Low cutoff frequency for bandpass filter
            highcut: High cutoff frequency for bandpass filter
            
        Returns:
            Preprocessed EEG data
        """
        # Transpose if needed (expect channels x time_steps)
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T
        
        # Apply bandpass filter
        nyquist = sample_rate / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[0]):
            filtered_data[i] = filtfilt(b, a, eeg_data[i])
        
        # Normalize
        filtered_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)
        
        # Reshape to model input shape
        # Pad or crop to match expected shape
        target_h, target_w = self.input_shape[0], self.input_shape[1]
        h, w = filtered_data.shape
        
        if h < target_h or w < target_w:
            # Pad
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            filtered_data = np.pad(filtered_data, ((0, pad_h), (0, pad_w)), mode='constant')
        elif h > target_h or w > target_w:
            # Crop
            filtered_data = filtered_data[:target_h, :target_w]
        
        # Add channel dimension
        if len(filtered_data.shape) == 2:
            filtered_data = np.expand_dims(filtered_data, axis=-1)
        
        return filtered_data
    
    def load_eeg_from_csv(self, csv_path: str) -> np.ndarray:
        """
        Load EEG data from CSV file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            EEG data as numpy array
        """
        try:
            df = pd.read_csv(csv_path, header=None)
            eeg_data = df.values
            # Transpose to get channels x time_steps format
            if eeg_data.shape[0] < eeg_data.shape[1]:
                eeg_data = eeg_data.T
            return eeg_data
        except Exception as e:
            print(f"⚠️ Error loading EEG from CSV: {e}")
            return np.zeros((self.input_shape[0], self.input_shape[1]))
    
    def predict_emotion(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion from EEG data
        
        Args:
            eeg_data: Preprocessed EEG data
            
        Returns:
            Dictionary of emotion probabilities
        """
        if self.model is None:
            # Return default neutral if no model
            return {'neutral': 1.0}
        
        try:
            # Preprocess
            processed = self.preprocess_eeg(eeg_data)
            
            # Reshape for model input
            if len(processed.shape) == 3:
                processed = np.expand_dims(processed, axis=0)
            
            # Predict
            predictions = self.model.predict(processed, verbose=0)
            
            # Convert to emotion dictionary
            if predictions.shape[1] == 1:
                # Binary classification - map to emotions
                prob = float(predictions[0][0])
                if prob > 0.5:
                    return {'happy': prob, 'neutral': 1 - prob}
                else:
                    return {'sad': 1 - prob, 'neutral': prob}
            else:
                # Multi-class classification
                emotions = {}
                for i, prob in enumerate(predictions[0]):
                    emotion = self.emotion_mapping.get(i, f'emotion_{i}')
                    emotions[emotion] = float(prob)
                return emotions
                
        except Exception as e:
            print(f"⚠️ Error predicting emotion from EEG: {e}")
            return {'neutral': 1.0}
    
    def process_eeg_file(self, eeg_path: str) -> Dict[str, Any]:
        """
        Process EEG file and extract emotions
        
        Args:
            eeg_path: Path to EEG CSV file
            
        Returns:
            Dictionary with emotion predictions and metadata
        """
        # Load EEG data
        eeg_data = self.load_eeg_from_csv(eeg_path)
        
        # Predict emotions
        emotions = self.predict_emotion(eeg_data)
        
        # Calculate additional features
        mean_amplitude = np.mean(np.abs(eeg_data))
        std_amplitude = np.std(eeg_data)
        energy = np.sum(eeg_data ** 2)
        
        return {
            'emotions': emotions,
            'primary_emotion': max(emotions.items(), key=lambda x: x[1])[0],
            'confidence': max(emotions.values()),
            'metadata': {
                'mean_amplitude': float(mean_amplitude),
                'std_amplitude': float(std_amplitude),
                'energy': float(energy),
                'shape': eeg_data.shape
            }
        }
    
    def process_eeg_stream(self, eeg_window: np.ndarray) -> Dict[str, float]:
        """
        Process real-time EEG stream window
        
        Args:
            eeg_window: EEG data window (channels x time_steps)
            
        Returns:
            Emotion probabilities
        """
        return self.predict_emotion(eeg_window)
    
    def get_emotion_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract emotion-related features from EEG
        
        Args:
            eeg_data: EEG data
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Frequency domain features
        fft_data = np.fft.fft(eeg_data, axis=1)
        power_spectrum = np.abs(fft_data) ** 2
        
        # Alpha band (8-13 Hz) - relaxation
        alpha_power = np.mean(power_spectrum[:, 8:14])
        features['alpha_power'] = float(alpha_power)
        
        # Beta band (13-30 Hz) - active thinking
        beta_power = np.mean(power_spectrum[:, 13:31])
        features['beta_power'] = float(beta_power)
        
        # Gamma band (30-50 Hz) - high cognitive activity
        gamma_power = np.mean(power_spectrum[:, 30:51])
        features['gamma_power'] = float(gamma_power)
        
        # Theta band (4-8 Hz) - drowsiness/meditation
        theta_power = np.mean(power_spectrum[:, 4:8])
        features['theta_power'] = float(theta_power)
        
        # Asymmetry features (left vs right hemisphere)
        if eeg_data.shape[0] >= 2:
            left_hemisphere = np.mean(eeg_data[:eeg_data.shape[0]//2, :])
            right_hemisphere = np.mean(eeg_data[eeg_data.shape[0]//2:, :])
            features['hemisphere_asymmetry'] = float(left_hemisphere - right_hemisphere)
        
        return features

