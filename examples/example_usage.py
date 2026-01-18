#!/usr/bin/env python3
"""
Example usage of Quantum Emotion Pipeline
Demonstrates both video and audio/text modes
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.quantum_pipeline import QuantumEmotionPipeline


def example_audio_text_mode():
    """Example: Process audio and text"""
    print("=" * 60)
    print("EXAMPLE 1: Audio/Text Mode")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = QuantumEmotionPipeline(mode='audio_text', ollama_model='llama2')
    
    # Example 1: Text only
    print("\n📝 Processing text: 'I'm so excited about this project!'")
    results = pipeline.process(text="I'm so excited about this project!")
    
    print(f"\n✅ Results:")
    print(f"  Primary Emotion: {results['quantum_state']['primary_emotion']}")
    print(f"  Confidence: {results['quantum_state']['emotional_superposition'].get(results['quantum_state']['primary_emotion'], 0):.3f}")
    print(f"  Sarcasm Probability: {results['quantum_state']['sarcasm_probability']:.3f}")
    print(f"\n  Reframed Output:")
    print(f"    {results['reframed_output']['reframed'][:150]}...")
    
    # Example 2: Text with negative emotion
    print("\n📝 Processing text: 'I'm really frustrated and upset about this situation.'")
    results2 = pipeline.process(text="I'm really frustrated and upset about this situation.")
    
    print(f"\n✅ Results:")
    print(f"  Primary Emotion: {results2['quantum_state']['primary_emotion']}")
    print(f"  Recommendations:")
    for rec in results2['final_output']['recommendations']:
        print(f"    - {rec}")


def example_video_mode():
    """Example: Process video (requires video file)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Video Mode")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = QuantumEmotionPipeline(mode='video', ollama_model='llama2')
    
    # Note: Replace with actual video path
    video_path = "path/to/your/video.mp4"
    
    print(f"\n🎥 Processing video: {video_path}")
    print("  (Replace video_path with actual video file)")
    
    # Uncomment to actually process:
    # results = pipeline.process(video_path=video_path)
    # print(f"\n✅ Results:")
    # print(f"  Face Emotion: {results['raw_emotions']['face'].get('dominant_emotion', 'N/A')}")
    # print(f"  Audio Emotion: {results['raw_emotions']['audio'].get('dominant_emotion', 'N/A')}")
    # print(f"  Primary Quantum Emotion: {results['quantum_state']['primary_emotion']}")


def example_batch_processing():
    """Example: Batch process multiple texts"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    pipeline = QuantumEmotionPipeline(mode='audio_text')
    
    texts = [
        "I'm feeling great today!",
        "This is really disappointing.",
        "I'm not sure how I feel about this.",
        "Wow, that's amazing!",
        "I'm worried about what might happen."
    ]
    
    print(f"\n📝 Processing {len(texts)} texts...")
    
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\n  [{i}/{len(texts)}] Processing: '{text[:50]}...'")
        result = pipeline.process(text=text)
        results.append(result)
        print(f"    → Emotion: {result['quantum_state']['primary_emotion']}")
    
    # Get summary
    summary = pipeline.get_history_summary()
    print(f"\n📊 Summary:")
    print(f"  Total Processings: {summary['total_processings']}")
    print(f"  Emotion Distribution:")
    for emotion, count in summary['emotion_distribution'].items():
        print(f"    {emotion}: {count}")
    print(f"  Average Sarcasm: {summary['average_sarcasm']:.3f}")
    print(f"  Most Common Emotion: {summary['most_common_emotion']}")


def example_custom_context():
    """Example: Process with custom context"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Context")
    print("=" * 60)
    
    pipeline = QuantumEmotionPipeline(mode='audio_text')
    
    text = "I'm feeling okay."
    
    # Process with different contexts
    contexts = [
        {'time': 'morning', 'situation': 'work_context'},
        {'time': 'evening', 'situation': 'social_context'},
        {'time': 'night', 'situation': 'stress_situation'}
    ]
    
    print(f"\n📝 Processing text: '{text}'")
    print("  With different contexts:")
    
    for context in contexts:
        result = pipeline.process(
            text=text,
            context=context
        )
        print(f"\n  Context: {context}")
        print(f"    Primary Emotion: {result['quantum_state']['primary_emotion']}")
        print(f"    Confidence: {result['quantum_state']['emotional_superposition'].get(result['quantum_state']['primary_emotion'], 0):.3f}")


def example_llm_features():
    """Example: Using LLM features directly"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Direct LLM Usage")
    print("=" * 60)
    
    from src.ollama_llm import OllamaLLM
    
    llm = OllamaLLM(model='llama2')
    
    # Reframe text
    print("\n✨ Reframing Text:")
    original = "I'm really upset about this."
    print(f"  Original: {original}")
    
    reframed = llm.reframe_output(
        original_text=original,
        target_emotion="calm"
    )
    print(f"  Reframed: {reframed.text[:200]}...")
    
    # Analyze emotion context
    print("\n🔍 Analyzing Emotion Context:")
    analysis = llm.analyze_emotion_context(
        text="I'm so happy! This is amazing!",
        face_emotion="happy",
        voice_emotion="excited"
    )
    print(f"  Analysis: {analysis.text[:200]}...")


if __name__ == "__main__":
    print("🌌 Quantum Emotion Pipeline - Examples")
    print("=" * 60)
    print("\nNote: Make sure Ollama is running for LLM features")
    print("  Run: ollama serve")
    print("  Pull model: ollama pull llama2")
    print()
    
    try:
        # Run examples
        example_audio_text_mode()
        example_video_mode()
        example_batch_processing()
        example_custom_context()
        example_llm_features()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Pull required model: ollama pull llama2")
        print("  3. Check model files exist in ../model/")
        import traceback
        traceback.print_exc()

