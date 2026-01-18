#!/usr/bin/env python3
"""
Fallback emotion detection system for when models fail to load
Provides rule-based emotion detection as backup
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict


class FallbackEmotionDetector:
    """
    Rule-based emotion detection fallback system
    """
    
    def __init__(self):
        # Define emotion keywords
        self.emotion_keywords = {
            'happy': [
                'happy', 'joy', 'excited', 'glad', 'delighted', 'pleased', 'cheerful',
                'enthusiastic', 'euphoric', 'ecstatic', 'thrilled', 'elated', 'jubilant',
                'merry', 'content', 'satisfied', 'amused', 'laugh', 'smile', 'wonderful',
                'great', 'fantastic', 'awesome', 'excellent', 'amazing', 'love', 'like'
            ],
            'sad': [
                'sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'melancholy',
                'sorrowful', 'grief', 'heartbroken', 'disappointed', 'let down',
                'crying', 'tears', 'weep', 'sob', 'lonely', 'empty', 'blue', 'down'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'frustrated',
                'outraged', 'resentful', 'hostile', 'aggressive', 'violent', 'hate',
                'disgusted', 'infuriated', 'livid', 'irate', 'indignant', 'wrath'
            ],
            'fear': [
                'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
                'worried', 'nervous', 'panic', 'phobia', 'dread', 'apprehensive',
                'concerned', 'uneasy', 'restless', 'tense', 'alarmed', 'horrified'
            ],
            'surprise': [
                'surprise', 'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered',
                'confused', 'perplexed', 'puzzled', 'unexpected', 'sudden', 'wow',
                'incredible', 'unbelievable', 'shocking', 'startled', 'astonishment'
            ],
            'disgust': [
                'disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated',
                'appalled', 'horrified', 'disgusting', 'gross', 'awful', 'terrible',
                'horrible', 'vile', 'repugnant', 'abhorrent', 'detestable'
            ],
            'neutral': [
                'okay', 'fine', 'normal', 'calm', 'peaceful', 'relaxed', 'balanced',
                'steady', 'composed', 'serene', 'tranquil', 'unaffected', 'impartial',
                'hi', 'hello', 'hey', 'greetings', 'welcome', 'good morning', 'good evening',
                'good afternoon', 'hiya', 'sup', 'yo'
            ]
        }
        
        # Audio feature patterns (simplified)
        self.audio_patterns = {
            'happy': {'high_energy': True, 'tempo_range': (120, 140), 'pitch_variance': 'high'},
            'sad': {'low_energy': True, 'tempo_range': (60, 80), 'pitch_variance': 'low'},
            'angry': {'high_energy': True, 'tempo_range': (100, 120), 'pitch_variance': 'medium'},
            'fear': {'medium_energy': True, 'tempo_range': (80, 100), 'pitch_variance': 'high'},
            'surprise': {'high_energy': True, 'tempo_range': (90, 110), 'pitch_variance': 'very_high'},
            'neutral': {'medium_energy': True, 'tempo_range': (70, 90), 'pitch_variance': 'low'}
        }
        
        # Face expression patterns
        self.face_patterns = {
            'happy': {'smile': True, 'eyes_open': True, 'mouth_corners_up': True},
            'sad': {'smile': False, 'eyes_down': True, 'mouth_corners_down': True},
            'angry': {'eyebrows_down': True, 'mouth_tense': True, 'eyes_narrow': True},
            'fear': {'eyes_wide': True, 'mouth_open': True, 'eyebrows_up': True},
            'surprise': {'eyes_wide': True, 'mouth_open': True, 'eyebrows_raised': True},
            'disgust': {'nose_wrinkled': True, 'mouth_tight': True, 'eyebrows_down': True},
            'neutral': {'face_relaxed': True, 'mouth_neutral': True, 'eyes_normal': True}
        }
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotion from text using keyword matching
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotion analysis
        """
        if not text or not isinstance(text, str):
            return self._neutral_response()
        
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        
        # Count keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            matches = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches += len(re.findall(pattern, text_lower))
            
            # Normalize score (0-1 range)
            emotion_scores[emotion] = min(matches / 5.0, 1.0)
        
        # Add some variation based on text characteristics
        if '!' in text:
            emotion_scores['surprise'] += 0.2
            emotion_scores['happy'] += 0.1
        
        if '?' in text:
            emotion_scores['surprise'] += 0.1
            emotion_scores['fear'] += 0.1
        
        # Check for negation patterns
        if any(word in text_lower for word in ['not', 'no', 'never', "don't", "can't"]):
            # Reduce positive emotions for negated statements
            for emotion in ['happy', 'excited', 'great']:
                emotion_scores[emotion] *= 0.5
        
        # Normalize to ensure sum = 1
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        else:
            # If no emotions detected, return neutral
            return self._neutral_response()
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'emotion_distribution': dict(emotion_scores),
            'method': 'fallback_keyword'
        }
    
    def analyze_audio_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze emotion from audio features using simplified patterns
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Dictionary with emotion analysis
        """
        if not features:
            return self._neutral_response()
        
        emotion_scores = defaultdict(float)
        
        # Extract key features (with fallbacks)
        energy = np.mean(features.get('energy', [0.5]))
        tempo = np.mean(features.get('tempo', [90]))
        pitch_std = np.std(features.get('pitch', [0]))
        
        # Score each emotion based on patterns
        for emotion, pattern in self.audio_patterns.items():
            score = 0.0
            
            # Energy matching
            if pattern.get('high_energy') and energy > 0.7:
                score += 0.3
            elif pattern.get('low_energy') and energy < 0.3:
                score += 0.3
            elif pattern.get('medium_energy') and 0.3 <= energy <= 0.7:
                score += 0.3
            
            # Tempo matching
            tempo_range = pattern.get('tempo_range', (70, 110))
            if tempo_range[0] <= tempo <= tempo_range[1]:
                score += 0.4
            
            # Pitch variance matching
            pitch_req = pattern.get('pitch_variance', 'medium')
            if pitch_req == 'high' and pitch_std > 0.5:
                score += 0.3
            elif pitch_req == 'low' and pitch_std < 0.2:
                score += 0.3
            elif pitch_req == 'medium' and 0.2 <= pitch_std <= 0.5:
                score += 0.3
            elif pitch_req == 'very_high' and pitch_std > 0.7:
                score += 0.3
            
            emotion_scores[emotion] = score
        
        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        else:
            return self._neutral_response()
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'emotion_distribution': dict(emotion_scores),
            'method': 'fallback_audio'
        }
    
    def analyze_face_emotion(self, face_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze emotion from face features using simplified patterns
        
        Args:
            face_features: Dictionary of face features
            
        Returns:
            Dictionary with emotion analysis
        """
        if not face_features:
            return self._neutral_response()
        
        emotion_scores = defaultdict(float)
        
        # Simple pattern matching (this would be more sophisticated with actual face features)
        for emotion, pattern in self.face_patterns.items():
            score = 0.0
            
            # This is a simplified version - in reality, you'd have actual face landmarks
            # For now, we'll add some randomness to simulate detection
            score += np.random.random() * 0.3
            
            emotion_scores[emotion] = score
        
        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        else:
            return self._neutral_response()
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': float(confidence),
            'emotion_distribution': dict(emotion_scores),
            'method': 'fallback_face'
        }
    
    def _neutral_response(self) -> Dict[str, Any]:
        """Return a neutral emotion response"""
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
        distribution = {emotion: 1.0/len(emotions) for emotion in emotions}
        
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.5,
            'emotion_distribution': distribution,
            'method': 'fallback_neutral'
        }
    
    def get_emotion_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get emotion from contextual information
        
        Args:
            context: Context dictionary
            
        Returns:
            Dictionary with emotion analysis
        """
        if not context:
            return self._neutral_response()
        
        # Check for explicit emotions in context
        if 'emotions' in context:
            context_emotions = context['emotions']
            if isinstance(context_emotions, dict):
                # Use provided emotions directly
                total = sum(context_emotions.values())
                if total > 0:
                    normalized = {k: v / total for k, v in context_emotions.items()}
                    dominant = max(normalized, key=normalized.get)
                    return {
                        'dominant_emotion': dominant,
                        'confidence': float(normalized[dominant]),
                        'emotion_distribution': normalized,
                        'method': 'context_provided'
                    }
        
        return self._neutral_response()


# Global fallback detector instance
fallback_detector = FallbackEmotionDetector()


def get_fallback_emotion(text: Optional[str] = None, 
                        audio_features: Optional[Dict] = None,
                        face_features: Optional[Dict] = None,
                        context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get emotion analysis using fallback methods
    
    Args:
        text: Text input
        audio_features: Audio features
        face_features: Face features
        context: Context information
        
    Returns:
        Dictionary with emotion analysis
    """
    # Try context first
    if context:
        context_result = fallback_detector.get_emotion_from_context(context)
        if context_result['confidence'] > 0.5:
            return context_result
    
    # Try text analysis
    if text:
        text_result = fallback_detector.analyze_text_emotion(text)
        if text_result['confidence'] > 0.3:
            return text_result
    
    # Try audio analysis
    if audio_features:
        audio_result = fallback_detector.analyze_audio_features(audio_features)
        if audio_result['confidence'] > 0.3:
            return audio_result
    
    # Try face analysis
    if face_features:
        face_result = fallback_detector.analyze_face_emotion(face_features)
        if face_result['confidence'] > 0.3:
            return face_result
    
    # Return neutral as last resort
    return fallback_detector._neutral_response()
