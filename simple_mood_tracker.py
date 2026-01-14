import cv2
import numpy as np
from collections import Counter
import time
import json
import os
from datetime import datetime

class SimpleMoodTracker:
    """Simplified mood tracker that works without TensorFlow dependency issues"""
    
    def __init__(self, model_path="model/expression.h5"):
        self.model_path = model_path
        self.model = None
        self.cap = None
        self.is_running = False
        
        # Emotion labels
        self.emotion_labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        
        # Timing parameters
        self.capture_interval = 3  # seconds between captures
        self.analysis_interval = 30  # seconds between mood analysis
        self.last_capture_time = 0
        self.last_analysis_time = 0
        
        # Storage for predictions
        self.recent_predictions = []
        self.current_mood = None
        self.mood_history = []
        
        # Try to load model with error handling
        self.load_model_safely()
    
    def load_model_safely(self):
        """Safely load model with better error handling"""
        try:
            # Try different import methods
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path)
                print(f"✓ Model loaded using TensorFlow")
            except ImportError:
                print("⚠️ TensorFlow not available, trying Keras standalone...")
                from keras.models import load_model
                self.model = load_model(self.model_path)
                print(f"✓ Model loaded using Keras")
            
            print(f"✓ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("🔄 Running in DEMO MODE with simulated emotions")
            self.model = None
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion prediction"""
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            face_img = cv2.resize(face_img, (48, 48))
            
            # Normalize
            face_img = face_img.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
            face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
            
            return face_img
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_emotions(self, face_img):
        """Predict emotions from face image"""
        if self.model is None:
            # Demo mode - return simulated emotions
            import random
            emotions = {label: random.random() for label in self.emotion_labels.values()}
            # Normalize to sum to 1
            total = sum(emotions.values())
            emotions = {k: v/total for k, v in emotions.items()}
            return emotions
        
        try:
            # Preprocess
            processed_img = self.preprocess_face(face_img)
            if processed_img is None:
                return None
            
            # Predict
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Convert to emotion dictionary
            emotions = {}
            for i, label in self.emotion_labels.items():
                emotions[label] = float(predictions[0][i])
            
            return emotions
        except Exception as e:
            print(f"Error predicting emotions: {e}")
            return None
    
    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV"""
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def capture_and_predict(self):
        """Capture image and predict emotions"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            return None
        
        # Process the largest face (best quality)
        largest_face = max(faces, key=lambda f: f[2] * f[3])  # w * h
        x, y, w, h = largest_face
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Predict emotions
        emotions = self.predict_emotions(face_roi)
        
        if emotions:
            # Get dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions
            }
            
            return prediction_data
        
        return None
    
    def analyze_mood_from_predictions(self):
        """Analyze mood from recent predictions using mode of dominant emotions"""
        if len(self.recent_predictions) < 2:
            return None
        
        # Extract dominant emotions from recent predictions
        dominant_emotions = [pred['dominant_emotion'] for pred in self.recent_predictions]
        
        # Calculate mode (most frequent emotion)
        emotion_counts = Counter(dominant_emotions)
        mode_emotion = emotion_counts.most_common(1)[0][0]
        confidence = emotion_counts[mode_emotion] / len(dominant_emotions)
        
        # Calculate average confidence for the mode emotion
        mode_predictions = [pred for pred in self.recent_predictions if pred['dominant_emotion'] == mode_emotion]
        avg_confidence = np.mean([pred['confidence'] for pred in mode_predictions])
        
        mood_data = {
            'timestamp': datetime.now().isoformat(),
            'mood': mode_emotion,
            'frequency_confidence': confidence,  # How often this emotion appeared
            'average_confidence': avg_confidence,  # Average prediction confidence
            'sample_size': len(self.recent_predictions),
            'analysis_period': f"{len(self.recent_predictions) * self.capture_interval} seconds"
        }
        
        return mood_data
    
    def start_tracking(self, show_preview=False):
        """Start continuous mood tracking"""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot start tracking: Could not open camera")
            return
        
        self.is_running = True
        
        mode_text = "DEMO MODE" if self.model is None else "LIVE MODE"
        print(f"🎥 Mood tracking started in {mode_text}...")
        print(f"📸 Capturing images every {self.capture_interval} seconds")
        print(f"🧠 Analyzing mood every {self.analysis_interval} seconds")
        print("Press 'q' to quit")
        
        self.last_capture_time = time.time()
        self.last_analysis_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Capture and predict every 3 seconds
            if current_time - self.last_capture_time >= self.capture_interval:
                prediction = self.capture_and_predict()
                if prediction:
                    self.recent_predictions.append(prediction)
                    print(f"📸 {prediction['timestamp'][-8:]}: {prediction['dominant_emotion']} ({prediction['confidence']:.2f})")
                else:
                    print(f"📸 {datetime.now().strftime('%H:%M:%S')}: No face detected")
                
                self.last_capture_time = current_time
            
            # Analyze mood every 30 seconds
            if current_time - self.last_analysis_time >= self.analysis_interval:
                mood = self.analyze_mood_from_predictions()
                if mood:
                    self.current_mood = mood['mood']
                    self.mood_history.append(mood)
                    print(f"\n🧠 MOOD UPDATE: {mood['mood']}")
                    print(f"   Frequency: {mood['frequency_confidence']:.2f}")
                    print(f"   Avg Confidence: {mood['average_confidence']:.2f}")
                    print(f"   Based on {mood['sample_size']} predictions over {mood['analysis_period']}\n")
                    
                    # Clear recent predictions for next cycle
                    self.recent_predictions = []
                else:
                    print(f"\n🧠 No mood data to analyze (insufficient predictions)\n")
                
                self.last_analysis_time = current_time
            
            # Show preview if requested
            if show_preview:
                # Draw info on frame
                cv2.putText(frame, f"Current Mood: {self.current_mood or 'Analyzing...'}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Predictions: {len(self.recent_predictions)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                mode_text = "DEMO MODE" if self.model is None else "LIVE MODE"
                cv2.putText(frame, mode_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                time_until_capture = max(0, self.capture_interval - (current_time - self.last_capture_time))
                time_until_analysis = max(0, self.analysis_interval - (current_time - self.last_analysis_time))
                cv2.putText(frame, f"Next capture: {time_until_capture:.1f}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Next analysis: {time_until_analysis:.1f}s", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Simple Mood Tracker', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_tracking()
    
    def stop_tracking(self):
        """Stop mood tracking"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Mood tracking stopped")
    
    def get_current_mood(self):
        """Get the current mood parameter - THIS IS THE PARAMETER YOU REQUESTED"""
        return self.current_mood
    
    def get_mood_history(self):
        """Get the complete mood history"""
        return self.mood_history
    
    def save_mood_history(self, filename="mood_history.json"):
        """Save mood history to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.mood_history, f, indent=2)
            print(f"💾 Mood history saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save mood history: {e}")
    
    def print_summary(self):
        """Print summary of mood tracking session"""
        if not self.mood_history:
            print("No mood data available")
            return
        
        print("\n" + "="*50)
        print("📊 MOOD TRACKING SUMMARY")
        print("="*50)
        
        # Mood frequency
        moods = [entry['mood'] for entry in self.mood_history]
        mood_counts = Counter(moods)
        
        print(f"Total mood analyses: {len(self.mood_history)}")
        print(f"Tracking duration: {len(self.mood_history) * self.analysis_interval / 60:.1f} minutes")
        print("\nMood frequency:")
        for mood, count in mood_counts.most_common():
            percentage = (count / len(moods)) * 100
            print(f"  {mood}: {count} times ({percentage:.1f}%)")
        
        # Most recent mood
        if self.mood_history:
            latest_mood = self.mood_history[-1]
            print(f"\nMost recent mood: {latest_mood['mood']}")
            print(f"  Time: {latest_mood['timestamp']}")
            print(f"  Confidence: {latest_mood['average_confidence']:.2f}")

def main():
    """Main function to demonstrate simple mood tracking"""
    print("🎭 Simple Mood Tracker")
    print("="*40)
    
    # Initialize tracker
    tracker = SimpleMoodTracker()
    
    try:
        # Start tracking with preview window
        tracker.start_tracking(show_preview=True)
        
        # Print summary when done
        tracker.print_summary()
        
        # Save history
        tracker.save_mood_history()
        
    except KeyboardInterrupt:
        print("\n⚠️ Tracking interrupted by user")
        tracker.stop_tracking()
        tracker.print_summary()
    except Exception as e:
        print(f"❌ Error during tracking: {e}")
        tracker.stop_tracking()

# Simple API for integration
class SimpleMoodAPI:
    """Simple API for integrating mood tracker with other applications"""
    
    def __init__(self):
        self.tracker = SimpleMoodTracker()
        self.tracking_thread = None
    
    def start_background_tracking(self):
        """Start tracking in background thread"""
        import threading
        
        if not self.tracking_thread or not self.tracking_thread.is_alive():
            self.tracking_thread = threading.Thread(
                target=self.tracker.start_tracking, 
                kwargs={'show_preview': False}
            )
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
            print("🎥 Background mood tracking started")
    
    def stop_background_tracking(self):
        """Stop background tracking"""
        self.tracker.stop_tracking()
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
        print("🛑 Background mood tracking stopped")
    
    def get_current_mood(self):
        """Get current mood parameter - THIS IS WHAT YOU REQUESTED"""
        return self.tracker.get_current_mood()

if __name__ == "__main__":
    main()
