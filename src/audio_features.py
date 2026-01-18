#!/usr/bin/env python3
"""
Shared audio feature extraction utilities
Used by both video_processor and audio_text_processor
"""

import numpy as np
import librosa
from typing import Dict, Optional


class AudioFeatureExtractor:
    """Shared audio feature extraction"""
    
    def __init__(self,
                 sample_rate: int = 22050,
                 n_mfcc: int = 40,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 max_pad_length: int = 173):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_pad_length = max_pad_length
    
    def extract_features(self, y: np.ndarray, sr: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract audio features for emotion detection
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of features or None on error
        """
        try:
            features = {}
            
            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features['mfcc'] = self._pad_or_truncate(mfccs)
            
            # Chroma
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr, hop_length=self.hop_length, n_fft=self.n_fft
            )
            features['chroma'] = self._pad_or_truncate(chroma)
            
            # Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=self.hop_length
            )
            features['spectral_contrast'] = self._pad_or_truncate(spectral_contrast)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            features['zcr'] = self._pad_or_truncate(zcr)
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=self.hop_length
            )
            features['spectral_rolloff'] = self._pad_or_truncate(spectral_rolloff)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
            features['rms'] = self._pad_or_truncate(rms)
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None
    
    def _pad_or_truncate(self, feature: np.ndarray) -> np.ndarray:
        """Pad or truncate feature to max_pad_length"""
        if feature.shape[1] < self.max_pad_length:
            feature = np.pad(
                feature,
                ((0, 0), (0, self.max_pad_length - feature.shape[1])),
                mode='constant'
            )
        else:
            feature = feature[:, :self.max_pad_length]
        return feature
    
    def combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all features into single array"""
        combined = np.concatenate([
            features['mfcc'],
            features['chroma'],
            features['spectral_contrast'],
            features['zcr'],
            features['spectral_rolloff'],
            features['rms']
        ], axis=0)
        
        # Flatten
        if combined.ndim > 1:
            combined = combined.T.flatten()
        else:
            combined = combined.flatten()
        
        return combined
    
    def reshape_for_model(self, features: np.ndarray,
                         n_time_steps: int = 173,
                         n_features_per_step: int = 62) -> np.ndarray:
        """Reshape features for model input"""
        return features.reshape(1, n_time_steps, n_features_per_step)

