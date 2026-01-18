"""
Quantum Emotion Pipeline - Source Package
"""

from .quantum_pipeline import QuantumEmotionPipeline
from .video_processor import VideoProcessor
from .audio_text_processor import AudioTextProcessor
from .ollama_llm import OllamaLLM, LLMResponse
from .quantum_emotion_engine import QuantumEmotionEngine, LongTermMemory, ConversationHistory
from .quantum_pipeline_integrated import IntegratedQuantumPipeline

__all__ = [
    'QuantumEmotionPipeline',
    'VideoProcessor',
    'AudioTextProcessor',
    'OllamaLLM',
    'LLMResponse',
    'QuantumEmotionEngine',
    'LongTermMemory',
    'ConversationHistory',
    'IntegratedQuantumPipeline'
]

