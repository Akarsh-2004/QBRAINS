#!/usr/bin/env python3
"""
Video Processing Module for Quantum Emotion Pipeline
Extracts face expressions, tone, and context from video input
"""

import cv2
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
import tempfile
import os
import subprocess
import shutil
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from src.utils import get_model_path, load_config, validate_model_path
from src.audio_features import AudioFeatureExtractor
from src.custom_layers import CustomDense, CustomInputLayer
from src.performance_optimizer import cached, timed, create_audio_cache_key, create_video_cache_key
from src.fallback_emotion_detector import get_fallback_emotion


class VideoProcessor:
    """Process video to extract face expressions, audio tone, and context"""
    
    def __init__(self, 
                 face_model_path: Optional[str] = None,
                 audio_model_path: Optional[str] = None,
                 audio_scaler_path: Optional[str] = None,
                 audio_encoder_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize video processor with models
        
        Args:
            face_model_path: Path to face emotion detection model (optional, uses config if None)
            audio_model_path: Path to audio emotion detection model (optional)
            audio_scaler_path: Path to audio feature scaler (optional)
            audio_encoder_path: Path to audio label encoder (optional)
            config: Configuration dictionary (optional)
        """
        self.config = config or load_config()
        self.face_model = None
        self.audio_model = None
        self.audio_scaler = None
        self.audio_label_encoder = None
        
        # Emotion labels
        self.face_emotion_labels = {
            0: 'angry', 1: 'disgust', 2: 'fear', 
            3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
        }
        
        # Resolve model paths
        if face_model_path is None:
            face_model_path = str(get_model_path(self.config['models']['face_model']))
        else:
            face_model_path = str(get_model_path(face_model_path))
        
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
        
        # Load models
        self._load_models(face_model_path, audio_model_path, 
                         audio_scaler_path, audio_encoder_path)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
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
        
        # Performance optimization: pre-allocate face detection buffer
        self.face_detection_buffer = []
        self.max_buffer_size = 5
        self.frame_skip_counter = 0
        self.processing_thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Check for ffmpeg
        self.ffmpeg_available = self._check_ffmpeg()
        
    def _load_models(self, face_path, audio_path, scaler_path, encoder_path):
        """Load all required models"""
        try:
            from src.safe_model_loader import safe_load_keras_model
            
            # Load face model
            face_path_obj = Path(face_path)
            if validate_model_path(face_path_obj, "Face model"):
                self.face_model = safe_load_keras_model(str(face_path_obj), compile_model=False)
                if self.face_model:
                    print(f"✓ Face emotion model loaded: {face_path_obj}")
            
            # Load audio model
            audio_path_obj = Path(audio_path)
            if validate_model_path(audio_path_obj, "Audio model"):
                self.audio_model = safe_load_keras_model(str(audio_path_obj), compile_model=False)
                if self.audio_model:
                    print(f"✓ Audio emotion model loaded: {audio_path_obj}")
            
            # Load audio scaler
            scaler_path_obj = Path(scaler_path)
            if validate_model_path(scaler_path_obj, "Audio scaler"):
                self.audio_scaler = joblib.load(str(scaler_path_obj))
                print(f"✓ Audio scaler loaded: {scaler_path_obj}")
            
            # Load label encoder
            encoder_path_obj = Path(encoder_path)
            if validate_model_path(encoder_path_obj, "Audio label encoder"):
                self.audio_label_encoder = joblib.load(str(encoder_path_obj))
                print(f"✓ Audio label encoder loaded: {encoder_path_obj}")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        return shutil.which('ffmpeg') is not None
    
    def process_video(self, video_path: str, 
                     extract_audio: bool = True,
                     frame_interval: Optional[int] = None) -> Dict[str, Any]:
        """
        Process video file to extract emotions and context
        
        Args:
            video_path: Path to video file
            extract_audio: Whether to extract and analyze audio
            frame_interval: Process every Nth frame (uses config if None)
            
        Returns:
            Dictionary with face emotions, audio emotions, and context
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if frame_interval is None:
            frame_interval = self.config.get('video', {}).get('frame_interval', 30)
        
        results = {
            'face_emotions': [],
            'audio_emotion': None,
            'tone_analysis': {},
            'context': {},
            'timeline': []
        }
        
        # Open video with context manager for proper cleanup
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            results['context']['fps'] = fps
            results['context']['total_frames'] = total_frames
            results['context']['duration'] = duration
            
            # Extract audio if requested
            if extract_audio:
                audio_emotion, tone = self._extract_audio_emotion(video_path)
                results['audio_emotion'] = audio_emotion
                results['tone_analysis'] = tone
            
            # Process frames with optimization
            frame_count = 0
            batch_faces = []
            batch_frame_times = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for performance (adaptive based on processing time)
                if frame_count % max(frame_interval // 2, 1) == 0:
                    # Detect faces with caching
                    faces = self._detect_faces_fast(frame, frame_count)
                    
                    if faces:
                        batch_faces.extend(faces)
                        batch_frame_times.extend([frame_count / fps if fps > 0 else 0] * len(faces))
                        
                        # Process in batches for better performance
                        if len(batch_faces) >= 3:
                            emotions = self._extract_face_emotions_batch(batch_faces)
                            for i, emotion in enumerate(emotions):
                                if emotion:
                                    results['face_emotions'].append({
                                        'frame': frame_count,
                                        'time': batch_frame_times[i],
                                        'emotion': emotion
                                    })
                            batch_faces.clear()
                            batch_frame_times.clear()
                
                frame_count += 1
            
            # Process remaining faces in buffer
            if batch_faces:
                emotions = self._extract_face_emotions_batch(batch_faces)
                for i, emotion in enumerate(emotions):
                    if emotion:
                        results['face_emotions'].append({
                            'frame': frame_count,
                            'time': batch_frame_times[i],
                            'emotion': emotion
                        })
        finally:
            cap.release()
        
        # Aggregate face emotions
        if results['face_emotions']:
            results['face_emotion_summary'] = self._aggregate_face_emotions(
                results['face_emotions']
            )
        
        return results
    
    def process_video_stream(self, cap: cv2.VideoCapture, 
                           frame_count: int = 0) -> Dict[str, Any]:
        """
        Process single frame from video stream
        
        Args:
            cap: OpenCV VideoCapture object
            frame_count: Current frame number
            
        Returns:
            Dictionary with emotion data for current frame
        """
        ret, frame = cap.read()
        if not ret:
            return None
        
        results = {
            'frame': frame_count,
            'face_emotions': [],
            'context': {}
        }
        
        # Detect faces
        faces = self._detect_faces(frame)
        
        for face_roi in faces:
            face_emotion = self._extract_face_emotion(face_roi)
            if face_emotion:
                results['face_emotions'].append(face_emotion)
        
        return results
    
    def _detect_faces_fast(self, frame: np.ndarray, frame_count: int) -> List[np.ndarray]:
        """Fast face detection with caching and optimization"""
        # Simple frame difference detection to skip processing static scenes
        if len(self.face_detection_buffer) > 0:
            last_frame = self.face_detection_buffer[-1][1]
            if np.mean(np.abs(frame.astype(float) - last_frame.astype(float))) < 10:
                # Scene hasn't changed much, reuse last detection
                if self.face_detection_buffer:
                    return self.face_detection_buffer[-1][0]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use faster parameters for face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,  # Faster but less accurate
            minNeighbors=3,   # Reduced for speed
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_rois = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_rois.append(face_roi)
        
        # Cache the detection
        self.face_detection_buffer.append((face_rois, gray.copy()))
        if len(self.face_detection_buffer) > self.max_buffer_size:
            self.face_detection_buffer.pop(0)
        
        return face_rois
    
    def _extract_face_emotions_batch(self, face_rois: List[np.ndarray]) -> List[Optional[Dict[str, float]]]:
        """Extract emotions from multiple faces in batch for better performance"""
        if self.face_model is None or not face_rois:
            return [None] * len(face_rois)
        
        try:
            # Preprocess all faces at once
            face_batch = np.array([roi.astype('float32') / 255.0 for roi in face_rois])
            face_batch = np.expand_dims(face_batch, axis=-1)
            
            # Predict in batch
            predictions = self.face_model.predict(face_batch, verbose=0, batch_size=len(face_rois))
            
            # Convert to emotion dictionaries
            results = []
            for pred in predictions:
                emotions = {}
                for idx, prob in enumerate(pred):
                    emotion_name = self.face_emotion_labels.get(idx, f'emotion_{idx}')
                    emotions[emotion_name] = float(prob)
                results.append(emotions)
            
            return results
            
        except Exception as e:
            print(f"Error extracting face emotions in batch: {e}")
            return [None] * len(face_rois)
    
    def _detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces in frame and return face ROIs"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        
        face_rois = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_rois.append(face_roi)
        
        return face_rois
    
    @cached(ttl_seconds=1800)  # 30 minutes cache
    @timed("face_emotion_extraction")
    def _extract_face_emotion(self, face_roi: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract emotion from face ROI (optimized)"""
        if self.face_model is None:
            # Use fallback when no model available
            try:
                fallback_result = get_fallback_emotion(face_features={'face_detected': True})
                return fallback_result.get('emotion_distribution', {})
            except:
                return {'neutral': 1.0}
        
        try:
            # Preprocess
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict with optimized settings
            predictions = self.face_model.predict(face_roi, verbose=0, batch_size=1)[0]
            
            # Convert to emotion dictionary (vectorized)
            emotions = {}
            for idx, prob in enumerate(predictions):
                emotion_name = self.face_emotion_labels.get(idx, f'emotion_{idx}')
                emotions[emotion_name] = float(prob)
            
            return emotions
            
        except Exception as e:
            print(f"Error extracting face emotion: {e}")
            # Use fallback on error
            try:
                fallback_result = get_fallback_emotion(face_features={'face_detected': True})
                return fallback_result.get('emotion_distribution', {})
            except:
                return {'neutral': 1.0}
    
    @cached(ttl_seconds=1800)
    @timed("audio_emotion_extraction")
    def _extract_audio_emotion(self, video_path: str) -> Tuple[Optional[Dict], Dict]:
        """Extract audio from video and analyze emotion"""
        if self.audio_model is None:
            return None, {}
        
        if not self.ffmpeg_available:
            print("⚠️ ffmpeg not available, skipping audio extraction")
            return None, {}
        
        tmp_path = None
        try:
            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                tmp_path = tmp_audio.name
            
            # Use subprocess instead of os.system for security
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',  # Mono
                tmp_path,
                '-y'  # Overwrite
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                print(f"⚠️ Failed to extract audio: {result.stderr}")
                return None, {}
            
            # Load audio
            y, sr = librosa.load(tmp_path, duration=self.duration, sr=self.sample_rate)
            
            # Extract features using shared extractor
            features = self.audio_feature_extractor.extract_features(y, sr)
            
            if features is None:
                return None, {}
            
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
            
            # Tone analysis
            tone_analysis = {
                'dominant_emotion': emotion,
                'confidence': float(confidence),
                'emotion_distribution': emotion_dist,
                'audio_features': {
                    'rms_energy': float(np.mean(features['rms'])),
                    'zero_crossing_rate': float(np.mean(features['zcr'])),
                    'spectral_rolloff': float(np.mean(features['spectral_rolloff']))
                }
            }
            
            return emotion_dist, tone_analysis
                
        except subprocess.TimeoutExpired:
            print("⚠️ Audio extraction timed out")
            return None, {}
        except Exception as e:
            print(f"Error extracting audio emotion: {e}")
            return None, {}
        finally:
            # Cleanup temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    
    def _aggregate_face_emotions(self, face_emotions: List[Dict]) -> Dict[str, Any]:
        """Aggregate face emotions over time"""
        if not face_emotions:
            return {}
        
        # Collect all emotion probabilities
        emotion_sums = {}
        emotion_counts = {}
        
        for entry in face_emotions:
            emotion = entry.get('emotion', {})
            for emo_name, prob in emotion.items():
                if emo_name not in emotion_sums:
                    emotion_sums[emo_name] = 0
                    emotion_counts[emo_name] = 0
                emotion_sums[emo_name] += prob
                emotion_counts[emo_name] += 1
        
        # Calculate averages
        emotion_averages = {
            emo: emotion_sums[emo] / emotion_counts[emo] 
            for emo in emotion_sums
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotion_averages, key=emotion_averages.get)
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_averages,
            'total_detections': len(face_emotions),
            'confidence': emotion_averages[dominant_emotion]
        }


def test_video_processor():
    """Test video processor"""
    print("🎥 Testing Video Processor")
    print("=" * 50)
    
    processor = VideoProcessor()
    
    print("\n✓ Video processor initialized")
    print("  Use processor.process_video(video_path) to process videos")
    print("  Use processor.process_video_stream(cap) for real-time processing")


if __name__ == "__main__":
    test_video_processor()

