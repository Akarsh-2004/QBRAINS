#!/usr/bin/env python3
"""
Test script for Mood Tracker - demonstrates how to access the current mood parameter
"""

import time
import threading
from mood_tracker import MoodTracker, MoodTrackerAPI

def test_basic_tracking():
    """Test basic mood tracking functionality"""
    print("🧪 Testing Basic Mood Tracking")
    print("=" * 40)
    
    tracker = MoodTracker()
    
    if tracker.model is None:
        print("❌ Model not found. Please ensure expression.h5 exists in ../model/ directory")
        return
    
    print("✅ Model loaded successfully")
    print("🎥 Starting 60-second tracking session...")
    print("Press Ctrl+C to stop early")
    
    try:
        # Start tracking without preview (for automated testing)
        tracker.start_tracking(show_preview=False)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        tracker.stop_tracking()
        tracker.print_summary()
        tracker.save_mood_history("test_mood_history.json")

def test_api_integration():
    """Test API integration for accessing current mood parameter"""
    print("\n🧪 Testing API Integration")
    print("=" * 40)
    
    api = MoodTrackerAPI()
    
    print("🎥 Starting background tracking...")
    api.start_background_tracking()
    
    print("⏱️ Monitoring mood parameter for 90 seconds...")
    print("Current mood parameter will be checked every 10 seconds")
    
    try:
        for i in range(9):  # 90 seconds total
            time.sleep(10)
            current_mood = api.get_current_mood()
            summary = api.get_mood_summary()
            
            print(f"\n📊 Check #{i+1}:")
            if current_mood:
                print(f"   Current Mood Parameter: {current_mood}")
            else:
                print("   Current Mood Parameter: Not available yet")
            
            if summary:
                print(f"   Recent Dominant Mood: {summary['recent_dominant']}")
                print(f"   Total Analyses: {summary['total_analyses']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        print("\n🛑 Stopping background tracking...")
        api.stop_background_tracking()
        
        # Final summary
        final_summary = api.get_mood_summary()
        if final_summary:
            print(f"\n📋 Final Summary:")
            print(f"   Final Mood Parameter: {final_summary['current_mood']}")
            print(f"   Total Mood Analyses: {final_summary['total_analyses']}")

def demonstrate_mood_parameter_usage():
    """Demonstrate how to use the mood parameter in an application"""
    print("\n🧪 Demonstrating Mood Parameter Usage")
    print("=" * 40)
    
    class MockApplication:
        """Example application that uses mood parameter"""
        
        def __init__(self):
            self.mood_api = MoodTrackerAPI()
            self.mood_responses = {
                'Happy': "🎵 Playing upbeat music...",
                'Sad': "🎵 Playing calming music...",
                'Angry': "🧘 Starting breathing exercise...",
                'Fear': "🌿 Playing nature sounds...",
                'Neutral': "📚 Suggesting focus music...",
                'Surprise': "🎉 Showing exciting content...",
                'Disgust': "🌸 Showing pleasant images..."
            }
        
        def start(self):
            """Start the application with mood tracking"""
            print("🚀 Starting application with mood tracking...")
            self.mood_api.start_background_tracking()
            
            # Simulate application running
            for i in range(6):  # Run for 60 seconds
                time.sleep(10)
                self.respond_to_mood()
        
        def respond_to_mood(self):
            """Respond based on current mood parameter"""
            current_mood = self.mood_api.get_current_mood()
            
            if current_mood:
                response = self.mood_responses.get(current_mood, "🤔 Adjusting to your mood...")
                print(f"   Mood detected: {current_mood} -> {response}")
            else:
                print("   Still analyzing mood...")
        
        def stop(self):
            """Stop the application"""
            self.mood_api.stop_background_tracking()
            print("🛑 Application stopped")
    
    # Run the mock application
    app = MockApplication()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted")
    finally:
        app.stop()

def main():
    """Main test function"""
    print("🎭 Mood Tracker Test Suite")
    print("=" * 50)
    print("This script demonstrates how to:")
    print("1. Track moods automatically")
    print("2. Access the current mood parameter")
    print("3. Integrate with other applications")
    print()
    
    # Check if model exists
    import os
    model_path = "../model/expression.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure you have trained the model and saved it as expression.h5")
        return
    
    print("✅ Model file found")
    
    # Run tests
    try:
        # Test 1: Basic tracking
        test_basic_tracking()
        
        # Test 2: API integration
        test_api_integration()
        
        # Test 3: Application integration
        demonstrate_mood_parameter_usage()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
