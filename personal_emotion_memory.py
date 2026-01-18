import sqlite3
import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import pickle

@dataclass
class EmotionSession:
    """Single emotion detection session"""
    timestamp: datetime
    emotions: Dict[str, float]  # emotion_name: confidence
    context: str  # optional context like "work", "home", "meeting"
    image_path: Optional[str] = None

@dataclass
class UserProfile:
    """User personalization data"""
    user_id: str
    created_at: datetime
    baseline_emotions: Dict[str, float]
    emotion_patterns: Dict[str, List]  # learned patterns
    preferences: Dict[str, any]
    total_sessions: int = 0

class PersonalEmotionMemory:
    """Local personal emotion memory system"""
    
    def __init__(self, db_path: str = "personal_emotion_memory.db"):
        self.db_path = db_path
        self.current_user_id = None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                created_at TEXT,
                baseline_emotions TEXT,
                emotion_patterns TEXT,
                preferences TEXT,
                total_sessions INTEGER DEFAULT 0
            )
        ''')
        
        # Emotion sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                emotions TEXT,
                context TEXT,
                image_path TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        # Personal insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personal_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                insight_type TEXT,
                content TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user_profile(self, user_identifier: str) -> str:
        """Create new user profile with unique ID"""
        user_id = hashlib.md5(f"{user_identifier}_{datetime.now()}".encode()).hexdigest()[:16]
        
        # Initialize with neutral baseline
        baseline_emotions = {
            'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 
            'Happy': 0.0, 'Neutral': 1.0, 'Sad': 0.0, 'Surprise': 0.0
        }
        
        profile = UserProfile(
            user_id=user_id,
            created_at=datetime.now(),
            baseline_emotions=baseline_emotions,
            emotion_patterns={},
            preferences={'privacy_level': 'standard', 'insight_frequency': 'daily'}
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_profiles 
            (user_id, created_at, baseline_emotions, emotion_patterns, preferences, total_sessions)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id,
            profile.created_at.isoformat(),
            json.dumps(profile.baseline_emotions),
            json.dumps(profile.emotion_patterns),
            json.dumps(profile.preferences),
            profile.total_sessions
        ))
        conn.commit()
        conn.close()
        
        self.current_user_id = user_id
        return user_id
    
    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load existing user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, created_at, baseline_emotions, emotion_patterns, 
                   preferences, total_sessions FROM user_profiles WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self.current_user_id = user_id
            return UserProfile(
                user_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                baseline_emotions=json.loads(row[2]),
                emotion_patterns=json.loads(row[3]),
                preferences=json.loads(row[4]),
                total_sessions=row[5]
            )
        return None
    
    def store_emotion_session(self, emotions: Dict[str, float], 
                            context: str = "general", 
                            image_path: Optional[str] = None):
        """Store a new emotion detection session"""
        if not self.current_user_id:
            raise ValueError("No user profile loaded")
        
        session = EmotionSession(
            timestamp=datetime.now(),
            emotions=emotions,
            context=context,
            image_path=image_path
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO emotion_sessions 
            (user_id, timestamp, emotions, context, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.current_user_id,
            session.timestamp.isoformat(),
            json.dumps(session.emotions),
            session.context,
            session.image_path
        ))
        
        # Update session count
        cursor.execute('''
            UPDATE user_profiles SET total_sessions = total_sessions + 1 
            WHERE user_id = ?
        ''', (self.current_user_id,))
        
        conn.commit()
        conn.close()
        
        # Update baseline emotions
        self._update_baseline_emotions(emotions)
        
        # Generate personal insights
        self._generate_insights(emotions, context)
    
    def _update_baseline_emotions(self, new_emotions: Dict[str, float]):
        """Update user's baseline emotions with new data"""
        profile = self.load_user_profile(self.current_user_id)
        if not profile:
            return
        
        # Exponential moving average for baseline update
        alpha = 0.1  # learning rate
        
        for emotion, value in new_emotions.items():
            if emotion in profile.baseline_emotions:
                profile.baseline_emotions[emotion] = (
                    alpha * value + (1 - alpha) * profile.baseline_emotions[emotion]
                )
        
        # Save updated baseline
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE user_profiles SET baseline_emotions = ? WHERE user_id = ?
        ''', (json.dumps(profile.baseline_emotions), self.current_user_id))
        conn.commit()
        conn.close()
    
    def _generate_insights(self, emotions: Dict[str, float], context: str):
        """Generate personal insights based on emotion patterns"""
        profile = self.load_user_profile(self.current_user_id)
        if not profile:
            return
        
        insights = []
        
        # Check for unusual emotions
        for emotion, value in emotions.items():
            baseline = profile.baseline_emotions.get(emotion, 0.0)
            if value > baseline + 0.3:  # Significantly higher than baseline
                insights.append({
                    'type': 'emotion_spike',
                    'content': f"The {emotion} level ({value:.2f}) is higher than the usual baseline ({baseline:.2f})",
                    'confidence': min((value - baseline) * 2, 1.0)
                })
        
        # Store insights
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for insight in insights:
            cursor.execute('''
                INSERT INTO personal_insights 
                (user_id, insight_type, content, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.current_user_id,
                insight['type'],
                insight['content'],
                insight['confidence'],
                datetime.now().isoformat()
            ))
        conn.commit()
        conn.close()
    
    def get_emotion_history(self, days: int = 7) -> List[EmotionSession]:
        """Get emotion history for specified number of days"""
        if not self.current_user_id:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, emotions, context, image_path 
            FROM emotion_sessions 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (self.current_user_id, cutoff_date.isoformat()))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(EmotionSession(
                timestamp=datetime.fromisoformat(row[0]),
                emotions=json.loads(row[1]),
                context=row[2],
                image_path=row[3]
            ))
        
        conn.close()
        return sessions
    
    def get_personal_insights(self, days: int = 7) -> List[Dict]:
        """Get recent personal insights"""
        if not self.current_user_id:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT insight_type, content, confidence, created_at 
            FROM personal_insights 
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at DESC
        ''', (self.current_user_id, cutoff_date.isoformat()))
        
        insights = []
        for row in cursor.fetchall():
            insights.append({
                'type': row[0],
                'content': row[1],
                'confidence': row[2],
                'created_at': row[3]
            })
        
        conn.close()
        return insights
    
    def get_emotion_patterns(self) -> Dict[str, any]:
        """Analyze and return emotion patterns"""
        if not self.current_user_id:
            return {}
        
        sessions = self.get_emotion_history(days=30)
        if not sessions:
            return {}
        
        # Analyze patterns
        patterns = {
            'daily_average': {},
            'context_patterns': {},
            'time_patterns': {}
        }
        
        # Calculate daily averages
        emotion_totals = {emotion: [] for emotion in sessions[0].emotions.keys()}
        for session in sessions:
            for emotion, value in session.emotions.items():
                emotion_totals[emotion].append(value)
        
        for emotion, values in emotion_totals.items():
            patterns['daily_average'][emotion] = np.mean(values) if values else 0.0
        
        # Analyze context patterns
        context_emotions = {}
        for session in sessions:
            if session.context not in context_emotions:
                context_emotions[session.context] = {}
            for emotion, value in session.emotions.items():
                if emotion not in context_emotions[session.context]:
                    context_emotions[session.context][emotion] = []
                context_emotions[session.context][emotion].append(value)
        
        for context, emotions in context_emotions.items():
            patterns['context_patterns'][context] = {
                emotion: np.mean(values) for emotion, values in emotions.items()
            }
        
        return patterns
    
    def export_user_data(self, export_path: str):
        """Export all user data for privacy/portability"""
        if not self.current_user_id:
            raise ValueError("No user profile loaded")
        
        profile = self.load_user_profile(self.current_user_id)
        sessions = self.get_emotion_history(days=365)  # Full year
        insights = self.get_personal_insights(days=365)
        patterns = self.get_emotion_patterns()
        
        export_data = {
            'profile': profile,
            'sessions': sessions,
            'insights': insights,
            'patterns': patterns,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def delete_user_data(self):
        """Delete all user data (privacy compliance)"""
        if not self.current_user_id:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete all user-related data
        cursor.execute('DELETE FROM emotion_sessions WHERE user_id = ?', (self.current_user_id,))
        cursor.execute('DELETE FROM personal_insights WHERE user_id = ?', (self.current_user_id,))
        cursor.execute('DELETE FROM user_profiles WHERE user_id = ?', (self.current_user_id,))
        
        conn.commit()
        conn.close()
        
        self.current_user_id = None

# Example usage and integration with emotion reader
def integrate_with_emotion_reader(emotion_model, image_path: str, context: str = "general"):
    """Integration function to use with the emotion reader"""
    
    # Initialize personal memory
    memory = PersonalEmotionMemory()
    
    # Load or create user profile
    user_id = "demo_user"  # In real app, this would come from login/session
    profile = memory.load_user_profile(user_id)
    if not profile:
        user_id = memory.create_user_profile("demo_user")
    
    # Simulate emotion prediction (replace with your actual model prediction)
    # emotions = predict_emotions(emotion_model, image_path)
    emotions = {
        'Angry': 0.1, 'Disgust': 0.05, 'Fear': 0.1, 
        'Happy': 0.6, 'Neutral': 0.1, 'Sad': 0.05, 'Surprise': 0.1
    }
    
    # Store session
    memory.store_emotion_session(emotions, context, image_path)
    
    # Get personal insights
    insights = memory.get_personal_insights(days=7)
    patterns = memory.get_emotion_patterns()
    
    return {
        'current_emotions': emotions,
        'personal_insights': insights,
        'emotion_patterns': patterns,
        'baseline_emotions': profile.baseline_emotions
    }

if __name__ == "__main__":
    # Demo usage
    memory = PersonalEmotionMemory()
    
    # Create demo user
    user_id = memory.create_user_profile("demo_user")
    print(f"Created user: {user_id}")
    
    # Store some demo sessions
    demo_emotions = [
        {'Happy': 0.8, 'Neutral': 0.2, 'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Sad': 0.0, 'Surprise': 0.0},
        {'Happy': 0.3, 'Neutral': 0.4, 'Angry': 0.2, 'Disgust': 0.0, 'Fear': 0.1, 'Sad': 0.0, 'Surprise': 0.0},
        {'Happy': 0.9, 'Neutral': 0.1, 'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Sad': 0.0, 'Surprise': 0.0}
    ]
    
    for i, emotions in enumerate(demo_emotions):
        memory.store_emotion_session(emotions, f"context_{i}")
    
    # Get insights
    insights = memory.get_personal_insights()
    patterns = memory.get_emotion_patterns()
    
    print("Personal Insights:")
    for insight in insights:
        print(f"- {insight['content']}")
    
    print("\nEmotion Patterns:")
    print(json.dumps(patterns, indent=2))
