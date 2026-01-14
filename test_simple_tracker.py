#!/usr/bin/env python3
"""
Simple test script for mood tracker - avoids TensorFlow dependency issues
"""

import time
import threading
from simple_mood_tracker import SimpleMoodTracker, SimpleMoodAPI

def test_basic_functionality():
    """Test basic mood tracking functionality"""
    print("🧪 Testing Simple Mood Tracker")
    print("=" * 40)
    
    tracker = SimpleMoodTracker()
    
    print(f"✅ Tracker initialized")
    print(f"📷 Model available: {'Yes' if tracker.model else 'No (Demo Mode)'}")
    
    print("\n🎥 Starting 60-second tracking session...")
    print("Press Ctrl+C to stop early")
    
    try:
        # Start tracking with preview
        tracker.start_tracking(show_preview=True)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        tracker.stop_tracking()
        tracker.print_summary()
        tracker.save_mood_history("simple_test_history.json")

def test_mood_parameter_access():
    """Test accessing the current mood parameter"""
    print("\n🧪 Testing Mood Parameter Access")
    print("=" * 40)
    
    api = SimpleMoodAPI()
    
    print("🎥 Starting background tracking...")
    api.start_background_tracking()
    
    print("⏱️ Testing mood parameter access for 60 seconds...")
    print("Checking current mood parameter every 5 seconds")
    
    try:
        for i in range(12):  # 60 seconds total
            time.sleep(5)
            current_mood = api.get_current_mood()  # THIS IS THE PARAMETER YOU REQUESTED
            
            print(f"📊 Check #{i+1}:")
            if current_mood:
                print(f"   🎯 Current Mood Parameter: '{current_mood}'")
            else:
                print(f"   ⏳ Current Mood Parameter: Not available yet")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        print("\n🛑 Stopping background tracking...")
        api.stop_background_tracking()
        
        # Final mood check
        final_mood = api.get_current_mood()
        print(f"\n📋 Final Mood Parameter: '{final_mood}'")

def demonstrate_integration():
    """Demonstrate how to integrate mood parameter in applications"""
    print("\n🧪 Demonstrating Application Integration")
    print("=" * 40)
    
    class MoodAwareApp:
        """Example application that uses mood parameter"""
        
        def __init__(self):
            self.mood_api = SimpleMoodAPI()
            self.responses = {
                'Happy': "🎵 Playing upbeat music...",
                'Sad': "🎵 Playing calming music...",
                'Angry': "🧘 Starting breathing exercise...",
                'Fear': "🌿 Playing nature sounds...",
                'Neutral': "📚 Suggesting focus music...",
                'Surprise': "🎉 Showing exciting content...",
                'Disgust': "🌸 Showing pleasant images..."
            }
        
        def start(self):
            """Start the application"""
            print("🚀 Starting mood-aware application...")
            self.mood_api.start_background_tracking()
            
            # Simulate application running
            for i in range(6):  # Run for 60 seconds
                time.sleep(10)
                self.check_and_respond()
        
        def check_and_respond(self):
            """Check mood parameter and respond"""
            current_mood = self.mood_api.get_current_mood()  # ACCESSING THE PARAMETER
            
            if current_mood:
                response = self.responses.get(current_mood, "🤔 Adjusting to your mood...")
                print(f"   Mood Parameter: '{current_mood}' -> {response}")
            else:
                print(f"   Mood Parameter: Still analyzing...")
        
        def stop(self):
            """Stop the application"""
            self.mood_api.stop_background_tracking()
            print("🛑 Application stopped")
    
    # Run the demo
    app = MoodAwareApp()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted")
    finally:
        app.stop()

def main():
    """Main test function"""
    print("🎭 Simple Mood Tracker Test Suite")
    print("=" * 50)
    print("This script demonstrates:")
    print("1. Basic mood tracking functionality")
    print("2. Accessing current mood parameter")
    print("3. Integration with applications")
    print()
    
    # Run tests
    try:
        # Test 1: Basic functionality
        test_basic_functionality()
        
        # Test 2: Mood parameter access
        test_mood_parameter_access()
        
        # Test 3: Application integration
        demonstrate_integration()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
