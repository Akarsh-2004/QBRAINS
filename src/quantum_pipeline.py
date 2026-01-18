#!/usr/bin/env python3
"""
Quantum-Inspired Emotion Pipeline
Main orchestrator for video-based and audio/text-based emotion processing
with LLM prediction and reframing
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import json
import numpy as np
from collections import Counter

# Import quantum AI
from quantum_emotion_ai import QuantumEmotionAI, QuantumEmotionState

# Import processors
from src.video_processor import VideoProcessor
from src.audio_text_processor import AudioTextProcessor
from src.ollama_llm import OllamaLLM, LLMResponse


class QuantumEmotionPipeline:
    """
    Main pipeline for quantum-inspired emotion processing
    
    Supports two modes:
    1. VIDEO: Processes video files (face + tone + context)
    2. AUDIO_TEXT: Processes audio files and/or text
    """
    
    def __init__(self, 
                 mode: Literal['video', 'audio_text'] = 'video',
                 ollama_model: str = 'llama2',
                 ollama_url: str = 'http://localhost:11434'):
        """
        Initialize quantum emotion pipeline
        
        Args:
            mode: Processing mode ('video' or 'audio_text')
            ollama_model: Ollama model name
            ollama_url: Ollama API URL
        """
        self.mode = mode
        self.quantum_ai = QuantumEmotionAI()
        
        # Initialize processors based on mode
        if mode == 'video':
            self.video_processor = VideoProcessor()
            self.audio_text_processor = None
        else:
            self.video_processor = None
            self.audio_text_processor = AudioTextProcessor()
        
        # Initialize LLM
        self.llm = OllamaLLM(base_url=ollama_url, model=ollama_model)
        
        # Processing history
        self.processing_history = []
        
        print(f"🌌 Quantum Emotion Pipeline initialized in {mode.upper()} mode")
    
    def process(self, 
                video_path: Optional[str] = None,
                audio_path: Optional[str] = None,
                text: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input based on pipeline mode
        
        Args:
            video_path: Path to video file (for video mode)
            audio_path: Path to audio file (for audio_text mode)
            text: Text input (for audio_text mode)
            context: Additional context information
            
        Returns:
            Complete processing results with quantum state and LLM predictions
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'input': {},
            'raw_emotions': {},
            'quantum_state': None,
            'llm_prediction': None,
            'reframed_output': None,
            'final_output': None
        }
        
        # Step 1: Extract raw emotions based on mode
        if self.mode == 'video':
            if not video_path:
                raise ValueError("video_path required for video mode")
            
            print(f"🎥 Processing video: {video_path}")
            video_results = self.video_processor.process_video(video_path)
            
            results['input']['video_path'] = video_path
            results['raw_emotions']['face'] = video_results.get('face_emotion_summary', {})
            results['raw_emotions']['audio'] = video_results.get('audio_emotion', {})
            results['raw_emotions']['tone'] = video_results.get('tone_analysis', {})
            results['raw_emotions']['context'] = video_results.get('context', {})
            
            # Prepare for quantum processing
            face_emotions = self._extract_face_emotions_from_video(video_results)
            voice_emotions = self._extract_voice_emotions_from_video(video_results)
            text_emotions = {}  # No text in video mode
            
        else:  # audio_text mode
            if not audio_path and not text:
                raise ValueError("audio_path or text required for audio_text mode")
            
            print(f"🎤 Processing audio/text...")
            audio_text_results = self.audio_text_processor.process_audio_text(
                audio_path=audio_path,
                text=text
            )
            
            results['input']['audio_path'] = audio_path
            results['input']['text'] = text
            results['raw_emotions']['audio'] = audio_text_results.get('audio_emotion', {})
            results['raw_emotions']['text'] = audio_text_results.get('text_emotion', {})
            results['raw_emotions']['combined'] = audio_text_results.get('combined_emotion', {})
            
            # Prepare for quantum processing
            face_emotions = {}  # No face in audio/text mode
            voice_emotions = self._extract_voice_emotions_from_audio_text(audio_text_results)
            text_emotions = self._extract_text_emotions_from_audio_text(audio_text_results)
        
        # Step 2: Create quantum superposition
        print("🌌 Creating quantum emotion superposition...")
        quantum_context = context or {}
        quantum_context.update(results['raw_emotions'].get('context', {}))
        
        quantum_state = self.quantum_ai.create_quantum_superposition(
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
            text_emotions=text_emotions,
            context=quantum_context
        )
        
        results['quantum_state'] = {
            'primary_emotion': quantum_state.primary_emotion,
            'emotional_superposition': quantum_state.emotional_superposition,
            'sarcasm_probability': quantum_state.sarcasm_probability,
            'authenticity_score': quantum_state.authenticity_score,
            'quantum_uncertainty': quantum_state.quantum_uncertainty
        }
        
        # Step 3: LLM prediction of next output
        print("🤖 Predicting next output type with LLM...")
        conversation_history = [text] if text else []
        llm_prediction = self.llm.predict_next_output(
            emotion_context=results['raw_emotions'],
            conversation_history=conversation_history,
            quantum_state=results['quantum_state']
        )
        
        results['llm_prediction'] = {
            'prediction_text': llm_prediction.text,
            'confidence': llm_prediction.confidence,
            'metadata': llm_prediction.metadata
        }
        
        # Try to parse JSON from LLM response
        try:
            import re
            json_match = re.search(r'\{.*\}', llm_prediction.text, re.DOTALL)
            if json_match:
                parsed_prediction = json.loads(json_match.group())
                results['llm_prediction']['parsed'] = parsed_prediction
        except:
            pass
        
        # Step 4: Reframe output using LLM
        print("✨ Reframing output with LLM...")
        original_output = self._generate_original_output(quantum_state, results)
        
        reframed_response = self.llm.reframe_output(
            original_text=original_output,
            target_emotion=quantum_state.primary_emotion,
            context={
                'quantum_state': results['quantum_state'],
                'raw_emotions': results['raw_emotions']
            }
        )
        
        results['reframed_output'] = {
            'original': original_output,
            'reframed': reframed_response.text,
            'confidence': reframed_response.confidence
        }
        
        # Step 5: Generate final output
        results['final_output'] = self._generate_final_output(results)
        
        # Store in history
        self.processing_history.append(results)
        
        return results
    
    def _extract_face_emotions_from_video(self, video_results: Dict) -> Dict[str, float]:
        """Extract face emotions in format expected by quantum AI"""
        face_summary = video_results.get('face_emotion_summary', {})
        emotion_dist = face_summary.get('emotion_distribution', {})
        
        # Normalize emotion names to match quantum AI expectations
        normalized = {}
        for emo, prob in emotion_dist.items():
            emo_lower = emo.lower()
            # Map to standard emotions
            if emo_lower in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
                normalized[emo_lower] = prob
        
        return normalized if normalized else {'neutral': 1.0}
    
    def _extract_voice_emotions_from_video(self, video_results: Dict) -> Dict[str, float]:
        """Extract voice emotions from video results"""
        audio_emotion = video_results.get('audio_emotion', {})
        if isinstance(audio_emotion, dict) and 'emotion_distribution' in audio_emotion:
            return audio_emotion['emotion_distribution']
        return {}
    
    def _extract_voice_emotions_from_audio_text(self, audio_text_results: Dict) -> Dict[str, float]:
        """Extract voice emotions from audio/text results"""
        audio_emotion = audio_text_results.get('audio_emotion', {})
        if isinstance(audio_emotion, dict) and 'emotion_distribution' in audio_emotion:
            return audio_emotion['emotion_distribution']
        return {}
    
    def _extract_text_emotions_from_audio_text(self, audio_text_results: Dict) -> Dict[str, float]:
        """Extract text emotions from audio/text results"""
        text_emotion = audio_text_results.get('text_emotion', {})
        if isinstance(text_emotion, dict) and 'emotion_distribution' in text_emotion:
            return text_emotion['emotion_distribution']
        return {}
    
    def _generate_original_output(self, quantum_state: QuantumEmotionState, 
                                 results: Dict) -> str:
        """Generate original output before reframing"""
        interpretation = self.quantum_ai.interpret_quantum_state(quantum_state)
        
        output_parts = [
            f"Primary emotion: {quantum_state.primary_emotion}",
            f"Confidence: {interpretation['confidence']:.2f}",
            f"Interpretation: {interpretation['interpretation']}"
        ]
        
        if quantum_state.sarcasm_probability > 0.5:
            output_parts.append(f"Sarcasm detected: {quantum_state.sarcasm_probability:.2f}")
        
        return ". ".join(output_parts)
    
    def _generate_final_output(self, results: Dict) -> Dict[str, Any]:
        """Generate final comprehensive output"""
        return {
            'summary': {
                'primary_emotion': results['quantum_state']['primary_emotion'],
                'confidence': results['quantum_state'].get('emotional_superposition', {}).get(
                    results['quantum_state']['primary_emotion'], 0
                ),
                'sarcasm_probability': results['quantum_state']['sarcasm_probability'],
                'authenticity_score': results['quantum_state']['authenticity_score']
            },
            'llm_prediction': results['llm_prediction'].get('parsed', {}),
            'reframed_text': results['reframed_output']['reframed'],
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        quantum_state = results['quantum_state']
        
        # High sarcasm
        if quantum_state['sarcasm_probability'] > 0.6:
            recommendations.append("High sarcasm detected - consider context carefully")
        
        # Low authenticity
        if quantum_state['authenticity_score'] < 0.4:
            recommendations.append("Low authenticity score - emotion may be performed")
        
        # High uncertainty
        if quantum_state['quantum_uncertainty'] > 0.6:
            recommendations.append("High uncertainty - consider gathering more context")
        
        # Negative emotions
        if quantum_state['primary_emotion'] in ['sad', 'angry', 'fear']:
            recommendations.append(f"Negative emotion detected ({quantum_state['primary_emotion']}) - may need support")
        
        return recommendations
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of processing history"""
        if not self.processing_history:
            return {'message': 'No processing history'}
        
        emotions = [r['quantum_state']['primary_emotion'] for r in self.processing_history]
        sarcasm_levels = [r['quantum_state']['sarcasm_probability'] for r in self.processing_history]
        
        return {
            'total_processings': len(self.processing_history),
            'emotion_distribution': dict(Counter(emotions)),
            'average_sarcasm': np.mean(sarcasm_levels),
            'most_common_emotion': Counter(emotions).most_common(1)[0][0] if emotions else None
        }
    
    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save processing results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_pipeline_results_{timestamp}.json"
        
        # Make results JSON-serializable
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")


def main():
    """Example usage of quantum pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Emotion Pipeline')
    parser.add_argument('--mode', choices=['video', 'audio_text'], default='audio_text',
                       help='Processing mode')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--audio', type=str, help='Audio file path')
    parser.add_argument('--text', type=str, help='Text input')
    parser.add_argument('--ollama-model', type=str, default='llama2',
                       help='Ollama model name')
    
    args = parser.parse_args()
    
    print("🌌 Quantum-Inspired Emotion Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = QuantumEmotionPipeline(
        mode=args.mode,
        ollama_model=args.ollama_model
    )
    
    # Process based on mode
    if args.mode == 'video':
        if not args.video:
            print("Error: --video required for video mode")
            return
        
        results = pipeline.process(video_path=args.video)
        
    else:  # audio_text mode
        if not args.audio and not args.text:
            print("Error: --audio or --text required for audio_text mode")
            return
        
        results = pipeline.process(
            audio_path=args.audio,
            text=args.text
        )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 PROCESSING RESULTS")
    print("=" * 60)
    print(f"\n🎯 Primary Emotion: {results['quantum_state']['primary_emotion']}")
    print(f"📈 Confidence: {results['quantum_state']['emotional_superposition'].get(results['quantum_state']['primary_emotion'], 0):.3f}")
    print(f"🎭 Sarcasm Probability: {results['quantum_state']['sarcasm_probability']:.3f}")
    print(f"✨ Authenticity Score: {results['quantum_state']['authenticity_score']:.3f}")
    
    print(f"\n🤖 LLM Prediction:")
    print(f"  {results['llm_prediction']['prediction_text'][:200]}...")
    
    print(f"\n✨ Reframed Output:")
    print(f"  Original: {results['reframed_output']['original'][:100]}...")
    print(f"  Reframed: {results['reframed_output']['reframed'][:200]}...")
    
    print(f"\n💡 Recommendations:")
    for rec in results['final_output']['recommendations']:
        print(f"  - {rec}")
    
    # Save results
    pipeline.save_results(results)
    
    print("\n🌌 Pipeline processing complete!")


if __name__ == "__main__":
    main()

