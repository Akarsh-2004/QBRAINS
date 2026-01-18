#!/usr/bin/env python3
"""
Integrated Quantum Pipeline
Complete pipeline: Input -> Quantum Engine -> Emotion LLM -> Ollama
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.quantum_emotion_engine import QuantumEmotionEngine
from typing import Dict, Optional, Any
import json


class IntegratedQuantumPipeline:
    """
    Complete integrated pipeline:
    Input -> Tone/Sentiment + Memory + History + Expression -> 
    Quantum Engine -> Emotion LLM -> Ollama
    """
    
    def __init__(self,
                 emotion_llm_path: str = "model/emotion_llm_final",
                 label_encoder_path: str = "model/emotion_label_encoder.pkl",
                 ollama_model: str = "gemma2:2b",  # Faster model by default
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize integrated pipeline
        
        Args:
            emotion_llm_path: Path to emotion LLM model
            label_encoder_path: Path to label encoder
            ollama_model: Ollama model name (default: gemma2:2b for speed)
            ollama_url: Ollama API URL
        """
        self.engine = QuantumEmotionEngine(
            emotion_llm_path=emotion_llm_path,
            label_encoder_path=label_encoder_path,
            ollama_model=ollama_model,
            ollama_url=ollama_url
        )
        
        print("🚀 Integrated Quantum Pipeline initialized")
        print("   Flow: Input -> Quantum Engine -> Emotion LLM -> Ollama")
    
    def process(self,
                text: Optional[str] = None,
                audio_path: Optional[str] = None,
                video_path: Optional[str] = None,
                face_emotions: Optional[Dict[str, float]] = None,
                eeg_path: Optional[str] = None,
                eeg_data: Optional[Any] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input through complete pipeline
        
        Args:
            text: Input text
            audio_path: Audio file path
            video_path: Video file path
            face_emotions: Face emotion distribution
            eeg_path: Path to EEG CSV file
            eeg_data: Raw EEG data array
            context: Additional context
            
        Returns:
            Complete processing result
        """
        return self.engine.process_input(
            text=text,
            audio_path=audio_path,
            video_path=video_path,
            face_emotions=face_emotions,
            eeg_path=eeg_path,
            eeg_data=eeg_data,
            context=context
        )
    
    def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Simple chat interface
        
        Args:
            message: User message
            context: Optional context
            
        Returns:
            Formatted response
        """
        result = self.process(text=message, context=context)
        return result['final_output']['formatted_text']
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        return {
            'baseline': self.engine.memory.get_emotional_baseline(),
            'recent_pattern': self.engine.memory.get_recent_pattern(),
            'total_interactions': len(self.engine.memory.memory['interactions'])
        }
    
    def clear_memory(self):
        """Clear memory (use with caution)"""
        self.engine.memory.memory = {
            'interactions': [],
            'patterns': {},
            'emotional_baselines': {},
            'triggers': {},
            'contextual_preferences': {}
        }
        self.engine.memory.save_memory()
        print("🧹 Memory cleared")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Quantum Pipeline')
    parser.add_argument('--text', type=str, help='Input text')
    parser.add_argument('--audio', type=str, help='Audio file path')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--ollama-model', type=str, default='llama2', help='Ollama model')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IntegratedQuantumPipeline(ollama_model=args.ollama_model)
    
    # Process input
    if args.text:
        result = pipeline.process(text=args.text)
        print("\n" + "=" * 70)
        print("📊 RESULT")
        print("=" * 70)
        print(f"Emotion: {result['quantum_superposition']['collapsed_emotion']}")
        print(f"Output: {result['final_output']['formatted_text']}")
    elif args.audio:
        result = pipeline.process(audio_path=args.audio)
        print(f"\nEmotion: {result['quantum_superposition']['collapsed_emotion']}")
        print(f"Output: {result['final_output']['formatted_text']}")
    elif args.video:
        result = pipeline.process(video_path=args.video)
        print(f"\nEmotion: {result['quantum_superposition']['collapsed_emotion']}")
        print(f"Output: {result['final_output']['formatted_text']}")
    else:
        # Interactive mode
        print("\n💬 Interactive Chat Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            response = pipeline.chat(user_input)
            print(f"AI: {response}\n")
        
        # Show memory summary
        summary = pipeline.get_memory_summary()
        print("\n📊 Memory Summary:")
        print(f"  Total interactions: {summary['total_interactions']}")
        print(f"  Baseline: {summary['baseline']}")


if __name__ == "__main__":
    main()

