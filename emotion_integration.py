import cv2
import numpy as np
from tensorflow.keras.models import load_model
from personal_emotion_memory import PersonalEmotionMemory
import tkinter as tk
from tkinter import messagebox
import threading
import time
from datetime import datetime

class EmotionReaderWithMemory:
    """Integrates face emotion reader with personal memory system"""
    
    def __init__(self, model_path="../model/improved_expression_model.keras"):
        self.model_path = model_path
        self.model = None
        self.memory = PersonalEmotionMemory()
        self.current_user_id = None
        self.emotion_labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        
        # Camera settings
        self.cap = None
        self.is_running = False
        
        # Initialize model
        self.load_model()
        self.setup_user()
    
    def load_model(self):
        """Load the trained emotion model"""
        try:
            self.model = load_model(self.model_path)
            print("✓ Emotion model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("Please train the model first using the improved notebook")
    
    def setup_user(self):
        """Setup or load user profile"""
        # For demo, use a fixed user ID. In real app, this would come from login
        user_id = "demo_user"
        profile = self.memory.load_user_profile(user_id)
        
        if not profile:
            user_id = self.memory.create_user_profile("demo_user")
            print(f"✓ Created new user profile: {user_id}")
        else:
            print(f"✓ Loaded existing user profile: {user_id}")
        
        self.current_user_id = user_id
    
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
            return None
        
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
    
    def start_camera_session(self, context="camera_session"):
        """Start camera session with emotion tracking"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please train the model first.")
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.is_running = True
        self.last_emotion_save = time.time()
        emotion_save_interval = 5  # Save emotions every 5 seconds
        
        print("📷 Camera session started. Press 'q' to quit.")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotions
                emotions = self.predict_emotions(face_roi)
                
                if emotions:
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Get dominant emotion
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    
                    # Draw emotion label
                    label = f"{dominant_emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Save emotions periodically
                    current_time = time.time()
                    if current_time - self.last_emotion_save >= emotion_save_interval:
                        self.memory.store_emotion_session(emotions, context)
                        self.last_emotion_save = current_time
                        print(f"💾 Saved emotions: {dominant_emotion} ({confidence:.2f})")
            
            # Display frame
            cv2.imshow('Emotion Recognition with Memory', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.stop_camera_session()
    
    def stop_camera_session(self):
        """Stop camera session"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera session stopped")
    
    def analyze_image_file(self, image_path, context="image_analysis"):
        """Analyze emotions from image file"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please train the model first.")
            return None
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", f"Could not load image: {image_path}")
                return None
            
            # Detect faces
            faces = self.detect_faces(img)
            
            if len(faces) == 0:
                messagebox.showinfo("Info", "No faces detected in the image")
                return None
            
            # Process first detected face
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            
            # Predict emotions
            emotions = self.predict_emotions(face_roi)
            
            if emotions:
                # Store in memory
                self.memory.store_emotion_session(emotions, context, image_path)
                
                # Get dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                
                print(f"📸 Image analyzed: {dominant_emotion} ({confidence:.2f})")
                return emotions
            
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing image: {e}")
            return None
    
    def get_personal_summary(self):
        """Get personal emotion summary"""
        if not self.current_user_id:
            return None
        
        profile = self.memory.load_user_profile(self.current_user_id)
        recent_sessions = self.memory.get_emotion_history(days=7)
        insights = self.memory.get_personal_insights(days=7)
        patterns = self.memory.get_emotion_patterns()
        
        summary = {
            'user_id': self.current_user_id,
            'total_sessions': profile.total_sessions if profile else 0,
            'baseline_emotions': profile.baseline_emotions if profile else {},
            'recent_sessions_count': len(recent_sessions),
            'recent_insights': insights,
            'emotion_patterns': patterns
        }
        
        return summary
    
    def run_gui_mode(self):
        """Run the GUI interface"""
        try:
            from emotion_memory_gui import EmotionMemoryGUI
            root = tk.Tk()
            app = EmotionMemoryGUI(root)
            root.mainloop()
        except ImportError:
            print("GUI not available. Please ensure emotion_memory_gui.py exists")
    
    def interactive_mode(self):
        """Interactive command-line mode"""
        print("\n🎭 Personal Emotion Reader - Interactive Mode")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. 📷 Start Camera Session")
            print("2. 📸 Analyze Image File")
            print("3. 📊 View Personal Summary")
            print("4. 🖥️  Launch GUI")
            print("5. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\nStarting camera session...")
                self.start_camera_session()
            
            elif choice == '2':
                image_path = input("Enter image path: ").strip()
                if image_path:
                    self.analyze_image_file(image_path)
            
            elif choice == '3':
                summary = self.get_personal_summary()
                if summary:
                    print(f"\n📊 Personal Summary for User: {summary['user_id']}")
                    print(f"Total Sessions: {summary['total_sessions']}")
                    print(f"Recent Sessions (7 days): {summary['recent_sessions_count']}")
                    
                    print("\n🎯 Baseline Emotions:")
                    for emotion, value in summary['baseline_emotions'].items():
                        print(f"  {emotion}: {value:.3f}")
                    
                    if summary['recent_insights']:
                        print("\n💡 Recent Insights:")
                        for insight in summary['recent_insights'][:3]:  # Show top 3
                            print(f"  • {insight['content']}")
                else:
                    print("No data available")
            
            elif choice == '4':
                print("Launching GUI...")
                self.run_gui_mode()
            
            elif choice == '5':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function to run the integrated emotion reader"""
    print("🎭 Personal Emotion Reader with Memory System")
    print("=" * 50)
    
    # Initialize the system
    reader = EmotionReaderWithMemory()
    
    # Check if model is available
    if reader.model is None:
        print("\n⚠️  No trained model found!")
        print("Please run the improved notebook first to train the model.")
        print("The model should be saved as: ../model/improved_expression_model.keras")
        return
    
    # Start interactive mode
    reader.interactive_mode()

if __name__ == "__main__":
    main()
