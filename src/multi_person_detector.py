#!/usr/bin/env python3
"""
Multi-Person Face Detection and Emotion Tracking
Tracks multiple people in video streams and analyzes emotions per person
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from src.video_processor import VideoProcessor
import time


@dataclass
class Person:
    """Represents a tracked person"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    emotions: Dict[str, float]
    emotion_history: deque
    track_history: deque
    last_seen: float
    confidence: float


class MultiPersonDetector:
    """Detect and track multiple people with emotion analysis"""
    
    def __init__(self,
                 face_model_path: str = "../model/improved_expression_model.keras",
                 max_track_age: float = 2.0,
                 min_track_hits: int = 3,
                 max_distance: float = 50.0):
        """
        Initialize multi-person detector
        
        Args:
            face_model_path: Path to face emotion model
            max_track_age: Maximum seconds before removing lost track
            min_track_hits: Minimum detections to create track
            max_distance: Maximum distance for track association
        """
        self.video_processor = VideoProcessor(face_model_path=face_model_path)
        self.max_track_age = max_track_age
        self.min_track_hits = min_track_hits
        self.max_distance = max_distance
        
        # Tracking
        self.next_id = 0
        self.tracks: Dict[int, Person] = {}
        self.frame_count = 0
        
        # Emotion history size
        self.history_size = 30
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_center_distance(self, box1: Tuple[int, int, int, int],
                                   box2: Tuple[int, int, int, int]) -> float:
        """Calculate center distance between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _associate_detections_to_tracks(self, detections: List[Tuple[int, int, int, int]],
                                       current_time: float) -> Tuple[Dict[int, int], List[int], List[int]]:
        """Associate detections to existing tracks"""
        if len(self.tracks) == 0:
            return {}, list(range(len(detections))), []
        
        # Calculate distance matrix
        distance_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, det in enumerate(detections):
            for j, (track_id, person) in enumerate(self.tracks.items()):
                distance = self._calculate_center_distance(det, person.bbox)
                distance_matrix[i, j] = distance
        
        # Greedy assignment
        matches = {}
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        # Sort by distance
        if distance_matrix.size > 0:
            indices = np.unravel_index(
                np.argsort(distance_matrix, axis=None),
                distance_matrix.shape
            )
            
            used_dets = set()
            used_tracks = set()
            
            for det_idx, track_idx in zip(indices[0], indices[1]):
                if det_idx in used_dets or track_idx in used_tracks:
                    continue
                
                track_id = list(self.tracks.keys())[track_idx]
                distance = distance_matrix[det_idx, track_idx]
                
                if distance <= self.max_distance:
                    matches[track_id] = det_idx
                    used_dets.add(det_idx)
                    used_tracks.add(track_idx)
                    if det_idx in unmatched_dets:
                        unmatched_dets.remove(det_idx)
                    if track_id in unmatched_tracks:
                        unmatched_tracks.remove(track_id)
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _update_track(self, track_id: int, bbox: Tuple[int, int, int, int],
                     emotions: Dict[str, float], current_time: float):
        """Update existing track"""
        person = self.tracks[track_id]
        person.bbox = bbox
        person.emotions = emotions
        person.last_seen = current_time
        person.confidence = max(emotions.values()) if emotions else 0.0
        
        # Update history
        person.emotion_history.append(emotions)
        person.track_history.append(bbox)
        
        if len(person.emotion_history) > self.history_size:
            person.emotion_history.popleft()
        if len(person.track_history) > self.history_size:
            person.track_history.popleft()
    
    def _create_track(self, bbox: Tuple[int, int, int, int],
                     emotions: Dict[str, float], current_time: float) -> int:
        """Create new track"""
        track_id = self.next_id
        self.next_id += 1
        
        person = Person(
            id=track_id,
            bbox=bbox,
            emotions=emotions,
            emotion_history=deque([emotions], maxlen=self.history_size),
            track_history=deque([bbox], maxlen=self.history_size),
            last_seen=current_time,
            confidence=max(emotions.values()) if emotions else 0.0
        )
        
        self.tracks[track_id] = person
        return track_id
    
    def _remove_old_tracks(self, current_time: float):
        """Remove tracks that haven't been seen recently"""
        to_remove = []
        for track_id, person in self.tracks.items():
            if current_time - person.last_seen > self.max_track_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def detect_and_track(self, frame: np.ndarray) -> Dict[int, Person]:
        """
        Detect faces and track people across frames
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary of tracked people (id -> Person)
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.video_processor.face_cascade.detectMultiScale(
            gray, 1.1, 4, minSize=(30, 30)
        )
        
        # Process each face for emotions
        detections_with_emotions = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
            face_input = np.expand_dims(
                np.expand_dims(face_normalized, axis=-1), axis=0
            )
            
            # Predict emotions
            emotions = {}
            if self.video_processor.face_model:
                try:
                    predictions = self.video_processor.face_model.predict(
                        face_input, verbose=0
                    )
                    for i, prob in enumerate(predictions[0]):
                        emotion = self.video_processor.face_emotion_labels.get(
                            i, f'emotion_{i}'
                        )
                        emotions[emotion] = float(prob)
                except Exception as e:
                    print(f"⚠️ Emotion prediction error: {e}")
            
            detections_with_emotions.append(((x, y, w, h), emotions))
        
        # Associate detections to tracks
        detections = [det[0] for det in detections_with_emotions]
        matches, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
            detections, current_time
        )
        
        # Update matched tracks
        for track_id, det_idx in matches.items():
            bbox, emotions = detections_with_emotions[det_idx]
            self._update_track(track_id, bbox, emotions, current_time)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox, emotions = detections_with_emotions[det_idx]
            self._create_track(bbox, emotions, current_time)
        
        # Remove old tracks
        self._remove_old_tracks(current_time)
        
        return self.tracks.copy()
    
    def get_person_emotion_summary(self, person_id: int) -> Dict[str, Any]:
        """Get emotion summary for a specific person"""
        if person_id not in self.tracks:
            return {}
        
        person = self.tracks[person_id]
        
        # Aggregate emotion history
        emotion_totals = defaultdict(float)
        for emotions in person.emotion_history:
            for emotion, prob in emotions.items():
                emotion_totals[emotion] += prob
        
        # Normalize
        total = sum(emotion_totals.values())
        if total > 0:
            emotion_averages = {
                k: v / total for k, v in emotion_totals.items()
            }
        else:
            emotion_averages = person.emotions
        
        return {
            'person_id': person_id,
            'current_emotions': person.emotions,
            'average_emotions': emotion_averages,
            'dominant_emotion': max(person.emotions.items(), key=lambda x: x[1])[0] if person.emotions else 'neutral',
            'confidence': person.confidence,
            'track_length': len(person.emotion_history),
            'bbox': person.bbox
        }
    
    def get_all_summaries(self) -> Dict[int, Dict[str, Any]]:
        """Get summaries for all tracked people"""
        return {
            person_id: self.get_person_emotion_summary(person_id)
            for person_id in self.tracks.keys()
        }
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking boxes and labels on frame"""
        output_frame = frame.copy()
        
        for person_id, person in self.tracks.items():
            x, y, w, h = person.bbox
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            
            # Get dominant emotion
            dominant_emotion = max(person.emotions.items(), key=lambda x: x[1])[0] if person.emotions else 'neutral'
            confidence = person.confidence
            
            # Draw label
            label = f"ID:{person_id} {dominant_emotion} ({confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y, label_size[1])
            
            cv2.rectangle(
                output_frame,
                (x, label_y - label_size[1] - 5),
                (x + label_size[0], label_y + 5),
                color,
                -1
            )
            cv2.putText(
                output_frame,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return output_frame

