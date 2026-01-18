#!/usr/bin/env python3
"""
Example usage of Quantum Emotion Engine
Demonstrates the complete pipeline flow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.quantum_emotion_engine import QuantumEmotionEngine
from src.quantum_pipeline_integrated import IntegratedQuantumPipeline


def example_basic_usage():
    """Basic usage example"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Text Processing")
    print("=" * 70)
    
    pipeline = IntegratedQuantumPipeline()
    
    result = pipeline.process(
        text="I'm feeling really frustrated with this situation.",
        context={'situation': 'work_context', 'time': 'afternoon'}
    )
    
    print(f"\n✅ Result:")
    print(f"  Emotion: {result['quantum_superposition']['collapsed_emotion']}")
    print(f"  Uncertainty: {result['quantum_superposition']['uncertainty']:.3f}")
    print(f"  Output: {result['final_output']['formatted_text']}")


def example_with_memory():
    """Example with memory building"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Building Memory Over Time")
    print("=" * 70)
    
    pipeline = IntegratedQuantumPipeline()
    
    # Simulate conversation
    messages = [
        "I'm feeling great today!",
        "Actually, I'm a bit worried about tomorrow.",
        "But I'm optimistic things will work out."
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n[{i}] User: {msg}")
        result = pipeline.process(text=msg)
        print(f"    Emotion: {result['quantum_superposition']['collapsed_emotion']}")
        print(f"    Response: {result['final_output']['formatted_text'][:100]}...")
    
    # Show memory
    memory = pipeline.get_memory_summary()
    print(f"\n📊 Memory Summary:")
    print(f"  Interactions: {memory['total_interactions']}")
    print(f"  Recent Pattern: {memory['recent_pattern']}")


def example_multimodal():
    """Example with multiple input types"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multimodal Input")
    print("=" * 70)
    
    pipeline = IntegratedQuantumPipeline()
    
    # Text + face emotions
    result = pipeline.process(
        text="I'm saying I'm fine, but...",
        face_emotions={'sad': 0.7, 'neutral': 0.3},
        context={'situation': 'personal_context'}
    )
    
    print(f"\n✅ Multimodal Result:")
    print(f"  Detected mismatch between text and expression")
    print(f"  Emotion: {result['quantum_superposition']['collapsed_emotion']}")
    print(f"  Output: {result['final_output']['formatted_text']}")


def example_quantum_analysis():
    """Example showing quantum analysis details"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Quantum Analysis Details")
    print("=" * 70)
    
    engine = QuantumEmotionEngine()
    
    result = engine.process_input(
        text="I'm not sure how I feel about this.",
        context={'situation': 'uncertain_context'}
    )
    
    print(f"\n🌌 Quantum Analysis:")
    print(f"  Collapsed Emotion: {result['quantum_superposition']['collapsed_emotion']}")
    print(f"  Uncertainty: {result['quantum_superposition']['uncertainty']:.3f}")
    print(f"\n  Top Possibilities:")
    for i, poss in enumerate(result['quantum_superposition']['possibilities'][:5], 1):
        print(f"    {i}. {poss['emotion']}: {poss['probability']:.3f} (from {poss['source']})")
    
    print(f"\n  Interference Patterns:")
    if result['quantum_superposition']['interference_patterns']:
        for emotion, interference in result['quantum_superposition']['interference_patterns'].items():
            if abs(interference) > 0.1:
                print(f"    {emotion}: {interference:.3f}")


if __name__ == "__main__":
    print("🌌 Quantum Emotion Engine Examples")
    print("=" * 70)
    
    try:
        example_basic_usage()
        example_with_memory()
        example_multimodal()
        example_quantum_analysis()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

