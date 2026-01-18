#!/usr/bin/env python3
"""
Real-time Processing Module for Quantum Emotion Pipeline
Handles live video, audio, and EEG streams
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import librosa
import sounddevice as sd
from src.video_processor import VideoProcessor
from src.audio_text_processor import AudioTextProcessor
from src.eeg_processor import EEGProcessor
from src.quantum_pipeline_integrated import IntegratedQuantumPipeline


class RealTimeProcessor:
    """Process real-time streams (video, audio, EEG)"""
    
    def __init__(self, 
                 pipeline: Optional[IntegratedQuantumPipeline] = None,
                 frame_buffer_size: int = 30,
                 audio_buffer_size: float = 3.0,
                 eeg_buffer_size: int = 256):
        """
        Initialize real-time processor
        
        Args:
            pipeline: Quantum pipeline instance
            frame_buffer_size: Number of frames to buffer
            audio_buffer_size: Audio buffer duration in seconds
            eeg_buffer_size: EEG samples to buffer
        """
        self.pipeline = pipeline or IntegratedQuantumPipeline()
        self.frame_buffer_size = frame_buffer_size
        self.audio_buffer_size = audio_buffer_size
        self.eeg_buffer_size = eeg_buffer_size
        
        # Processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioTextProcessor()
        self.eeg_processor = EEGProcessor()
        
        # Buffers
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.audio_buffer = deque()
        self.eeg_buffer = deque(maxlen=eeg_buffer_size)
        
        # Default rates (prevent AttributeError in workers)
        self.audio_sample_rate = 22050
        self.eeg_sample_rate = 256
        
        # Threading
        self.running = False
        self.video_thread = None
        self.audio_thread = None
        self.eeg_thread = None
        self.processing_thread = None
        
        # Callbacks
        self.on_emotion_update: Optional[Callable] = None
        self.on_result: Optional[Callable] = None
        
        # Results queue
        self.result_queue = queue.Queue()
    
    def start_video_stream(self, camera_id: int = 0, fps: int = 30):
        """
        Start real-time video stream processing
        
        Args:
            camera_id: Camera device ID
            fps: Target frames per second
        """
        self.running = True
        self.video_thread = threading.Thread(
            target=self._video_stream_worker,
            args=(camera_id, fps),
            daemon=True
        )
        self.video_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_streams_worker,
            daemon=True
        )
        self.processing_thread.start()
    
    def _video_stream_worker(self, camera_id: int, fps: int):
        """Video stream worker thread"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        frame_time = 1.0 / fps
        
        while self.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and extract emotions
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.video_processor.face_cascade.detectMultiScale(
                gray, 1.1, 4
            )
            
            frame_data = {
                'frame': frame_rgb,
                'timestamp': time.time(),
                'faces': faces
            }
            
            # Process face emotions if model available
            if self.video_processor.face_model and len(faces) > 0:
                face_emotions_list = []
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (48, 48))
                    face_normalized = face_resized.astype('float32') / 255.0
                    face_input = np.expand_dims(
                        np.expand_dims(face_normalized, axis=-1), axis=0
                    )
                    
                    predictions = self.video_processor.face_model.predict(
                        face_input, verbose=0
                    )
                    
                    emotions = {}
                    for i, prob in enumerate(predictions[0]):
                        emotion = self.video_processor.face_emotion_labels.get(i, f'emotion_{i}')
                        emotions[emotion] = float(prob)
                    
                    face_emotions_list.append({
                        'bbox': (x, y, w, h),
                        'emotions': emotions
                    })
                
                frame_data['face_emotions'] = face_emotions_list
            
            self.frame_buffer.append(frame_data)
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
        
        cap.release()
    
    def start_audio_stream(self, sample_rate: int = 22050, channels: int = 1):
        """
        Start real-time audio stream processing
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.running = True
        self.audio_sample_rate = sample_rate
        self.audio_channels = channels
        
        self.audio_thread = threading.Thread(
            target=self._audio_stream_worker,
            args=(sample_rate, channels),
            daemon=True
        )
        self.audio_thread.start()
        
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self._process_streams_worker,
                daemon=True
            )
            self.processing_thread.start()
    
    def _audio_stream_worker(self, sample_rate: int, channels: int):
        """Audio stream worker thread"""
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            self.audio_buffer.extend(indata[:, 0] if channels == 1 else indata.flatten())
            
            # Process when buffer is full
            buffer_samples = int(self.audio_buffer_size * sample_rate)
            if len(self.audio_buffer) >= buffer_samples:
                audio_data = np.array(list(self.audio_buffer)[:buffer_samples])
                # Keep remaining samples
                remaining = list(self.audio_buffer)[buffer_samples:]
                self.audio_buffer.clear()
                self.audio_buffer.extend(remaining)
        
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=audio_callback,
                blocksize=int(sample_rate * 0.1)  # 100ms blocks
            ):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Audio stream error: {e}")
    
    def start_eeg_stream(self, sample_rate: int = 256, channels: int = 64):
        """
        Start real-time EEG stream processing
        
        Args:
            sample_rate: EEG sample rate
            channels: Number of EEG channels
        """
        self.running = True
        self.eeg_sample_rate = sample_rate
        self.eeg_channels = channels
        
        self.eeg_thread = threading.Thread(
            target=self._eeg_stream_worker,
            daemon=True
        )
        self.eeg_thread.start()
        
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self._process_streams_worker,
                daemon=True
            )
            self.processing_thread.start()
    
    def _eeg_stream_worker(self):
        """EEG stream worker thread (placeholder - requires actual EEG hardware)"""
        # This would connect to actual EEG hardware
        # For now, it's a placeholder
        print("⚠️ EEG stream requires hardware connection")
        while self.running:
            # Simulate EEG data (replace with actual hardware reading)
            time.sleep(0.1)
    
    def add_eeg_sample(self, eeg_sample: np.ndarray):
        """Add EEG sample to buffer (for external EEG sources)"""
        self.eeg_buffer.append(eeg_sample)
    
    def process_frame_binary(self, base64_frame: str):
        """
        Process a binary/base64 frame from a WebSocket
        
        Args:
            base64_frame: Base64 encoded JPEG/PNG frame
        """
        try:
            import base64
            # Remove header if present
            if "," in base64_frame:
                base64_frame = base64_frame.split(",")[1]
            
            data = base64.b64decode(base64_frame)
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces logic (same balance as stream worker)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.video_processor.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                frame_data = {
                    'frame': frame_rgb,
                    'timestamp': time.time(),
                    'faces': faces
                }
                
                if self.video_processor.face_model and len(faces) > 0:
                    face_emotions_list = []
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_roi, (48, 48))
                        face_normalized = face_resized.astype('float32') / 255.0
                        face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
                        
                        predictions = self.video_processor.face_model.predict(face_input, verbose=0)
                        emotions = {self.video_processor.face_emotion_labels.get(i, f'em_{i}'): float(p) 
                                   for i, p in enumerate(predictions[0])}
                        
                        face_emotions_list.append({'bbox': (int(x), int(y), int(w), int(h)), 'emotions': emotions})
                    frame_data['face_emotions'] = face_emotions_list
                
                self.frame_buffer.append(frame_data)
        except Exception as e:
            print(f"⚠️ Error processing WS frame: {e}")

    def process_audio_binary(self, audio_bytes: bytes):
        """
        Process binary audio chunk from a WebSocket
        
        Args:
            audio_bytes: Raw PCM or WebM/OGG bytes
        """
        try:
            # We use librosa to decode various formats from bytes
            import io
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=self.audio_sample_rate)
            self.audio_buffer.extend(y)
        except Exception as e:
            print(f"⚠️ Error processing WS audio: {e}")
    
    def _process_streams_worker(self):
        """Process buffered streams and generate results"""
        last_process_time = 0
        process_interval = 1.0  # Process every second
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_process_time >= process_interval:
                # Aggregate emotions from all sources
                aggregated_emotions = {}
                
                # Process video frames
                if len(self.frame_buffer) > 0:
                    recent_frames = list(self.frame_buffer)[-10:]  # Last 10 frames
                    face_emotions_all = []
                    
                    for frame_data in recent_frames:
                        if 'face_emotions' in frame_data:
                            for face_data in frame_data['face_emotions']:
                                face_emotions_all.append(face_data['emotions'])
                    
                    if face_emotions_all:
                        # Average face emotions
                        for emotion_dict in face_emotions_all:
                            for emotion, prob in emotion_dict.items():
                                aggregated_emotions[emotion] = aggregated_emotions.get(emotion, 0) + prob
                        
                        # Normalize
                        total = sum(aggregated_emotions.values())
                        if total > 0:
                            aggregated_emotions = {
                                k: v / total for k, v in aggregated_emotions.items()
                            }
                
                # Process audio
                if len(self.audio_buffer) >= int(self.audio_buffer_size * self.audio_sample_rate):
                    audio_data = np.array(list(self.audio_buffer))
                    try:
                        audio_result = self.audio_processor.process_audio_data(audio_data)
                        audio_emotions = audio_result.get('emotion_distribution', {})
                        
                        # Merge audio emotions
                        for emotion, prob in audio_emotions.items():
                            aggregated_emotions[emotion] = (
                                aggregated_emotions.get(emotion, 0) * 0.6 + prob * 0.4
                            )
                    except Exception as e:
                        print(f"⚠️ Audio processing error: {e}")
                
                # Process EEG
                if len(self.eeg_buffer) >= self.eeg_buffer_size:
                    eeg_window = np.array(list(self.eeg_buffer))
                    try:
                        eeg_emotions = self.eeg_processor.process_eeg_stream(eeg_window)
                        
                        # Merge EEG emotions
                        for emotion, prob in eeg_emotions.items():
                            aggregated_emotions[emotion] = (
                                aggregated_emotions.get(emotion, 0) * 0.5 + prob * 0.5
                            )
                    except Exception as e:
                        print(f"⚠️ EEG processing error: {e}")
                
                # Process through quantum pipeline
                if aggregated_emotions:
                    try:
                        result = self.pipeline.process(
                            face_emotions=aggregated_emotions,
                            context={'source': 'realtime', 'timestamp': current_time}
                        )
                        
                        # Call callbacks
                        if self.on_emotion_update:
                            self.on_emotion_update(aggregated_emotions)
                        
                        if self.on_result:
                            self.on_result(result)
                        
                        self.result_queue.put(result)
                    except Exception as e:
                        print(f"⚠️ Pipeline processing error: {e}")
                
                last_process_time = current_time
            
            time.sleep(0.1)
    
    def get_latest_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get latest processing result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop all streams"""
        self.running = False
        
        if self.video_thread:
            self.video_thread.join(timeout=2.0)
        
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        
        if self.eeg_thread:
            self.eeg_thread.join(timeout=2.0)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("✓ Real-time processing stopped")
    
    def set_callbacks(self, 
                     on_emotion_update: Optional[Callable] = None,
                     on_result: Optional[Callable] = None):
        """
        Set callback functions
        
        Args:
            on_emotion_update: Called with emotion dict when emotions update
            on_result: Called with full result when processing completes
        """
        self.on_emotion_update = on_emotion_update
        self.on_result = on_result

