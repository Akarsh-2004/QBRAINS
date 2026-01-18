                #!/usr/bin/env python3
"""
Quantum Emotion Engine
Compares all possibilities from multiple sources using quantum principles
and generates optimal emotion-aware outputs through LLM and Ollama
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import joblib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

# Import existing components
from quantum_emotion_ai import QuantumEmotionAI, QuantumEmotionState
from src.ollama_llm import OllamaLLM, LLMResponse
from src.audio_text_processor import AudioTextProcessor
from src.video_processor import VideoProcessor
from src.eeg_processor import EEGProcessor


@dataclass
class QuantumPossibility:
    """Represents a quantum possibility state"""
    emotion: str
    probability: float
    source: str  # 'tone', 'memory', 'expression', 'context', 'history'
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class QuantumSuperposition:
    """Quantum superposition of all emotion possibilities"""
    possibilities: List[QuantumPossibility]
    collapsed_emotion: Optional[str] = None
    uncertainty: float = 0.0
    interference_patterns: Dict[str, float] = None


class LongTermMemory:
    """Long-term memory system for emotional patterns"""
    
    def __init__(self, memory_file: str = "data/emotion_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory = self._load_memory()
        self.pattern_window = 10  # Number of recent interactions to analyze
        
    def _load_memory(self) -> Dict:
        """Load memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'interactions': [],
            'patterns': {},
            'emotional_baselines': {},
            'triggers': {},
            'contextual_preferences': {}
        }
    
    def save_memory(self):
        """Save memory to file"""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def add_interaction(self, emotion: str, context: Dict, text: str = None):
        """Add a new interaction to memory"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'context': context,
            'text': text
        }
        self.memory['interactions'].append(interaction)
        
        # Keep only recent interactions (last 1000)
        if len(self.memory['interactions']) > 1000:
            self.memory['interactions'] = self.memory['interactions'][-1000:]
        
        # Update patterns
        self._update_patterns()
        self.save_memory()
    
    def _update_patterns(self):
        """Update emotional patterns from recent interactions"""
        if len(self.memory['interactions']) < 5:
            return
        
        recent = self.memory['interactions'][-self.pattern_window:]
        emotions = [i['emotion'] for i in recent]
        
        # Count emotion frequencies
        from collections import Counter
        emotion_counts = Counter(emotions)
        
        # Update patterns
        for emotion, count in emotion_counts.items():
            if emotion not in self.memory['patterns']:
                self.memory['patterns'][emotion] = []
            self.memory['patterns'][emotion].append(count / len(recent))
            
            # Keep only recent pattern data
            if len(self.memory['patterns'][emotion]) > 100:
                self.memory['patterns'][emotion] = self.memory['patterns'][emotion][-100:]
    
    def get_emotional_baseline(self, user_id: str = "default") -> Dict[str, float]:
        """Get user's emotional baseline"""
        if user_id in self.memory['emotional_baselines']:
            return self.memory['emotional_baselines'][user_id]
        
        # Calculate from history
        if len(self.memory['interactions']) > 0:
            from collections import Counter
            emotions = [i['emotion'] for i in self.memory['interactions']]
            counts = Counter(emotions)
            total = sum(counts.values())
            
            baseline = {emotion: count / total for emotion, count in counts.items()}
            self.memory['emotional_baselines'][user_id] = baseline
            return baseline
        
        return {}
    
    def get_recent_pattern(self, window: int = 10) -> Dict[str, float]:
        """Get recent emotional pattern"""
        if len(self.memory['interactions']) < window:
            window = len(self.memory['interactions'])
        
        if window == 0:
            return {}
        
        recent = self.memory['interactions'][-window:]
        from collections import Counter
        emotions = [i['emotion'] for i in recent]
        counts = Counter(emotions)
        total = sum(counts.values())
        
        return {emotion: count / total for emotion, count in counts.items()}

    def get_relevant_memories(self, text: str, limit: int = 5) -> List[Dict]:
        """Simple keyword-based semantic retrieval of relevant past interactions"""
        if not text:
            return []
            
        relevant = []
        keywords = set(text.lower().split())
        # Filter out common stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at'}
        keywords = {k for k in keywords if k not in stop_words and len(k) > 3}
        
        if not keywords:
            return []

        # Iterate through history backwards
        for interaction in reversed(self.memory['interactions']):
            hist_text = interaction.get('text', '')
            if not hist_text:
                continue
                
            # Check for keyword overlap
            hist_words = set(hist_text.lower().split())
            overlap = keywords.intersection(hist_words)
            
            if overlap:
                # Add score based on overlap count and recency (implicitly by order)
                score = len(overlap)
                relevant.append({**interaction, 'relevance_score': score})
                
                if len(relevant) >= limit:
                    break
        
        return relevant
    
    def get_contextual_preference(self, context_key: str) -> Optional[Dict[str, float]]:
        """Get emotional preference for specific context"""
        if context_key in self.memory['contextual_preferences']:
            return self.memory['contextual_preferences'][context_key]
        return None


class ToneAnalyzer:
    """Analyzes tone and sentiment from text/audio"""
    
    def __init__(self):
        self.audio_text_processor = AudioTextProcessor()
    
    def analyze_tone(self, text: Optional[str] = None, 
                    audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze tone from text and/or audio"""
        if text and audio_path:
            result = self.audio_text_processor.process_audio_text(
                audio_path=audio_path,
                text=text
            )
        elif text:
            result = self.audio_text_processor.process_text(text)
        elif audio_path:
            result = self.audio_text_processor.process_audio(audio_path)
        else:
            return {}
        
        return {
            'dominant_emotion': result.get('dominant_emotion', 'neutral'),
            'emotion_distribution': result.get('emotion_distribution', {}),
            'sentiment': result.get('sentiment', 'neutral'),
            'confidence': result.get('confidence', 0.5)
        }


class ConversationHistory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 50):
        self.history = deque(maxlen=max_history)
        self.context_window = 5  # Number of recent messages to consider
    
    def add_message(self, role: str, content: str, emotion: Optional[str] = None):
        """Add a message to history"""
        self.history.append({
            'role': role,
            'content': content,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_recent_context(self, n: Optional[int] = None) -> List[Dict]:
        """Get recent conversation context"""
        n = n or self.context_window
        return list(self.history)[-n:]
    
    def get_emotion_flow(self, window: int = 10) -> List[str]:
        """Get emotion flow from recent messages"""
        recent = list(self.history)[-window:]
        return [msg.get('emotion', 'neutral') for msg in recent if msg.get('emotion')]


class QuantumEmotionEngine:
    """
    Quantum-based engine that compares all possibilities and generates
    optimal emotion-aware outputs
    """
    
    def __init__(self,
                 emotion_llm_path: str = "model/emotion_llm_final",
                 label_encoder_path: str = "model/emotion_label_encoder.pkl",
                 ollama_model: str = "mistral:latest",
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize Quantum Emotion Engine
        
        Args:
            emotion_llm_path: Path to trained emotion LLM
            label_encoder_path: Path to label encoder
            ollama_model: Ollama model name
            ollama_url: Ollama API URL
        """
        # Initialize components
        self.emotion_ai = QuantumEmotionAI()
        self.memory = LongTermMemory()
        self.tone_analyzer = ToneAnalyzer()
        self.conversation_history = ConversationHistory()
        self.ollama = OllamaLLM(base_url=ollama_url, model=ollama_model)
        
        # Initialize processors once to avoid slow model reloading
        self.video_processor = VideoProcessor()
        self.eeg_processor = EEGProcessor()
        
        # State tracking for "Momentum"
        self.previous_distribution = {}  # Tracks the energy landscape of the previous turn
        
        # Load emotion LLM
        self.emotion_llm = None
        self.emotion_tokenizer = None
        self.label_encoder = None
        self._load_emotion_llm(emotion_llm_path, label_encoder_path)
        
        # Performance optimization: add caching and threading
        self.possibility_cache = {}
        self.cache_size_limit = 50
        self.quantum_thread_pool = ThreadPoolExecutor(max_workers=3)
        self.interference_cache = {}
        
        # Quantum parameters
        self.interference_matrix = self._create_interference_matrix()
        self.quantum_uncertainty_threshold = 0.3
        
        print("🌌 Quantum Emotion Engine initialized")
    
    def _load_emotion_llm(self, model_path: str, encoder_path: str):
        """Load emotion LLM model"""
        try:
            model_file = Path(model_path)
            encoder_file = Path(encoder_path)
            
            if model_file.exists() and encoder_file.exists():
                self.emotion_llm = AutoModelForSequenceClassification.from_pretrained(
                    str(model_file)
                )
                self.emotion_tokenizer = AutoTokenizer.from_pretrained(str(model_file))
                self.label_encoder = joblib.load(str(encoder_file))
                
                self.emotion_llm.eval()
                print(f"✅ Emotion LLM loaded from {model_path}")
            else:
                print(f"⚠️ Emotion LLM not found at {model_path}, using fallback")
        except Exception as e:
            print(f"⚠️ Error loading emotion LLM: {e}")
    
    def _create_interference_matrix(self) -> np.ndarray:
        """Create quantum interference matrix for emotion interactions"""
        emotions = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
        n = len(emotions)
        matrix = np.zeros((n, n))
        
        # Define interference patterns
        interference_patterns = {
            ('happy', 'sad'): -0.3,      # Strong negative interference
            ('angry', 'fear'): 0.4,      # Angry amplifies fear
            ('surprise', 'neutral'): -0.2, # Surprise breaks neutrality
            ('disgust', 'happy'): -0.4,   # Disgust strongly dampens happiness
            ('fear', 'sad'): 0.3,         # Fear and sadness co-occur
            ('happy', 'surprise'): 0.2,    # Happy and surprise amplify
            ('angry', 'disgust'): 0.3,     # Angry and disgust co-occur
        }
        
        emotion_to_idx = {emotion: i for i, emotion in enumerate(emotions)}
        
        for (emotion1, emotion2), strength in interference_patterns.items():
            if emotion1 in emotion_to_idx and emotion2 in emotion_to_idx:
                i, j = emotion_to_idx[emotion1], emotion_to_idx[emotion2]
                matrix[i][j] = strength
        
        return matrix
    
    def process_input(self,
                     text: Optional[str] = None,
                     audio_path: Optional[str] = None,
                     video_path: Optional[str] = None,
                     face_emotions: Optional[Dict[str, float]] = None,
                     eeg_path: Optional[str] = None,
                     eeg_data: Optional[np.ndarray] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input through quantum engine pipeline (optimized)
        
        Args:
            text: Input text
            audio_path: Path to audio file
            video_path: Path to video file
            face_emotions: Face emotion distribution
            eeg_path: Path to EEG CSV file
            eeg_data: Raw EEG data array (channels x time_steps)
            context: Additional context
            
        Returns:
            Complete processing result with quantum state and generated output
        """
        start_time = time.time()
        
        # Create cache key for input
        input_hash = self._create_input_hash(text, audio_path, video_path, face_emotions, context)
        
        # Check cache first
        if input_hash in self.possibility_cache:
            cached_result = self.possibility_cache[input_hash]
            cached_result['cached'] = True
            cached_result['processing_time'] = time.time() - start_time
            # Update cache timestamp to keep it fresh without blocking
            self.quantum_thread_pool.submit(lambda: self._update_cache_timestamp(input_hash))
            return cached_result
        
        print("\n" + "=" * 70)
        print("🌌 QUANTUM EMOTION ENGINE - Processing Input (Turbo Optimized)")
        print("=" * 70)
        
        
        # Step 1: Collect possibilities
        possibilities = self._collect_possibilities_parallel(text, audio_path, video_path, 
                                                              face_emotions, eeg_path, eeg_data, context)
        
        # Step 2: Quantum Superposition
        superposition = self._create_quantum_superposition_optimized(possibilities)
            
        # Step 3: Collapse (Now returns (emotion, confidence, distribution))
        collapsed_emotion, collapsed_conf, emotion_distribution = self._collapse_quantum_state(superposition)
        
        # Step 4: Determine Interaction State
        interaction_state = self._get_interaction_state(emotion_distribution, superposition.uncertainty)

        # Step 5: Generate output (Optimized: Skip local LLM if using Ollama)
        # We create a dummy llm_output structure to pass to format_with_ollama
        # This avoids the slow Keras model entirely for response generation
        llm_output = {
            'text': text, 
            'emotion': collapsed_emotion, 
            'confidence': collapsed_conf,
            'distribution': emotion_distribution,
            'interaction_state': interaction_state,
            'method': 'quantum_direct'
        }
        
        # Use local LLM logic only if Ollama is NOT available as fallback
        # (Skipping fallback for now to rely on improved Ollama flow)

        # Step 6: Format/Generate with Ollama
        # Step 7: Update History BEFORE generation (to ensure current turn knows about current message if needed, or at least for immediate consistency)
        if text:
             self.conversation_history.add_message('user', text, collapsed_emotion)
        
        # Step 8: Generate output with Ollama
        final_output = self._format_with_ollama(
            llm_output=llm_output,
            quantum_state=superposition,
            context=context
        )
        
        # Add assistant response to history
        if final_output.get('formatted_text'):
            self.conversation_history.add_message('assistant', final_output['formatted_text'], collapsed_emotion)
        
        # Async memory update (non-critical for immediate response)
        self._update_memory_async(collapsed_emotion, context, text)
        
        result = {
            'input': {
                'text': text,
                'audio_path': audio_path,
                'video_path': video_path,
                'context': context
            },
            'quantum_superposition': {
                'possibilities': [asdict(p) for p in superposition.possibilities],
                'collapsed_emotion': collapsed_emotion,
                'uncertainty': superposition.uncertainty,
                'interference_patterns': superposition.interference_patterns
            },
            'emotion_llm_output': llm_output,
            'final_output': final_output,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time,
            'cached': False
        }
        
        # Cache asynchronously
        if len(self.possibility_cache) < self.cache_size_limit:
             self.quantum_thread_pool.submit(lambda: self.possibility_cache.update({input_hash: result}))

        print("\n" + "=" * 70)
        print("✅ QUANTUM PROCESSING COMPLETE")
        print("=" * 70)
        print(f"🎯 Collapsed Emotion: {collapsed_emotion}")
        print(f"🌊 Uncertainty: {superposition.uncertainty:.3f}")
        print(f"⚡ Processing Time: {result['processing_time']:.3f}s")
        print(f"✨ Final Output: {final_output.get('formatted_text', 'N/A')[:100]}...")
        
        return result
    
    def _create_input_hash(self, text: Optional[str], audio_path: Optional[str], 
                           video_path: Optional[str], face_emotions: Optional[Dict[str, float]], 
                           context: Optional[Dict[str, Any]]) -> str:
        """Create hash for input caching"""
        hash_input = f"{text}_{audio_path}_{video_path}_{str(face_emotions)}_{str(context)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _collect_possibilities_parallel(self, text: Optional[str], audio_path: Optional[str], 
                                      video_path: Optional[str], face_emotions: Optional[Dict[str, float]], 
                                      eeg_path: Optional[str], eeg_data: Optional[np.ndarray], 
                                      context: Optional[Dict[str, Any]]) -> List[QuantumPossibility]:
        """Collect possibilities from all sources in parallel"""
        possibilities = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # Tone analysis
            if text or audio_path:
                futures.append(executor.submit(self._analyze_tone, text, audio_path))
            
            # Memory analysis
            futures.append(executor.submit(self._analyze_memory))
            
            # Expression analysis
            if face_emotions or video_path:
                futures.append(executor.submit(self._analyze_expressions, face_emotions, video_path))
            
            # History analysis (New: Compensate for missing LLM)
            futures.append(executor.submit(self._analyze_history))
            
            # EEG analysis (New: Added to parallel collection)
            if eeg_path or eeg_data is not None:
                futures.append(executor.submit(self._analyze_eeg, eeg_path, eeg_data))
            
            # --- PHASE 1: INTUITION LAYER (FIRST LLM) ---
            # Run "Gemma" (or similar fast model) in parallel to get raw intent
            if self.ollama.available:
                futures.append(executor.submit(self._analyze_with_first_llm, text))

            # Collect results
            for future in futures:
                try:
                    # Reduced timeout to prevents 35s+ delays. If LLM is too slow, we skip it.
                    result = future.result(timeout=8) 
                    possibilities.extend(result)
                except Exception as e:
                    # Log silently to avoid spamming user console, or use debug log
                    pass  
                    # print(f"  ⚠️ Parallel processing info: {repr(e)}")
        
        return possibilities
    
    def _analyze_with_first_llm(self, text: str) -> List[QuantumPossibility]:
        """Use the First LLM (Intuition Layer) to analyze raw text with history"""
        possibilities = []
        try:
            # Get recent context for the LLM
            history_context = self.conversation_history.get_recent_context(3)
            # Format as strings
            history_msgs = [f"{msg.get('role', 'unknown')}: {msg.get('text', '')}" for msg in history_context]
            
            # Use a fast model like gemma2:2b
            analysis = self.ollama.analyze_sentiment(text, model="gemma2:2b", history_context=history_msgs)
            
            if analysis and 'emotion' in analysis:
                emotion = analysis['emotion'].lower()
                
                # Direct mapping for complex emotions the Quantum Engine might not standardly handle
                if emotion in ['sarcastic', 'passive_aggressive']:
                    # Map these to 'disgust' or 'mixed' for internal processing but keep metadata
                    emotion_mapped = 'disgust' 
                elif emotion == 'mixed':
                    emotion_mapped = 'neutral'
                else:
                    emotion_mapped = emotion

                possibilities.append(QuantumPossibility(
                    emotion=emotion_mapped,
                    probability=analysis.get('confidence', 0.8),
                    source='llm_intuition',
                    confidence=analysis.get('confidence', 0.8),
                    metadata={
                        'intent': analysis.get('intent', 'unknown'),
                        'original_label': emotion
                    }
                ))
        except Exception as e:
            print(f"Intuition layer failed: {repr(e)}")
            
        return possibilities

    def _analyze_history(self) -> List[QuantumPossibility]:
        """Analyze conversation history for emotional momentum"""
        possibilities = []
        recent_flow = self.conversation_history.get_emotion_flow(window=5)
        
        if not recent_flow:
            return possibilities
            
        # calculate momentum - if last 3 were 'happy', likely 'happy'
        if len(recent_flow) >= 3:
            last_3 = recent_flow[-3:]
            if len(set(last_3)) == 1: # All same
                emotion = last_3[0]
                possibilities.append(QuantumPossibility(
                    emotion=emotion,
                    probability=0.4,
                    source='history_momentum',
                    confidence=0.6,
                    metadata={'momentum': 'strong'}
                ))
        
        # Last emotion influence
        if recent_flow:
            last_emotion = recent_flow[-1]
            possibilities.append(QuantumPossibility(
                emotion=last_emotion,
                probability=0.3,
                source='history_continuity',
                confidence=0.5,
                metadata={'last_emotion': last_emotion}
            ))
            
        return possibilities

    def _analyze_tone(self, text: Optional[str], audio_path: Optional[str]) -> List[QuantumPossibility]:
        """Analyze tone and sentiment"""
        possibilities = []
        tone_result = self.tone_analyzer.analyze_tone(text=text, audio_path=audio_path)
        if tone_result:
            for emotion, prob in tone_result.get('emotion_distribution', {}).items():
                possibilities.append(QuantumPossibility(
                    emotion=emotion,
                    probability=prob,
                    source='tone',
                    confidence=tone_result.get('confidence', 0.5),
                    metadata={'sentiment': tone_result.get('sentiment', 'neutral')}
                ))
        return possibilities
    
    def _analyze_memory(self) -> List[QuantumPossibility]:
        """Analyze emotional memory pattern and relevant history"""
        possibilities = []
        
        # 1. Baseline Analysis
        baseline = self.memory.get_emotional_baseline()
        for emotion, prob in baseline.items():
            if prob > 0.2:  # Only consider significant baseline tendencies
                possibilities.append(QuantumPossibility(
                    emotion=emotion,
                    probability=prob * 0.3,  # Lower weight for general baseline
                    source='memory_baseline',
                    confidence=0.8,
                    metadata={'type': 'recent_pattern'}
                ))
        return possibilities
    
    def _analyze_expressions(self, face_emotions: Optional[Dict[str, float]], 
                           video_path: Optional[str]) -> List[QuantumPossibility]:
        """Analyze facial expressions"""
        possibilities = []
        
        if face_emotions:
            for emotion, prob in face_emotions.items():
                possibilities.append(QuantumPossibility(
                    emotion=emotion,
                    probability=prob * 0.6,
                    source='expression',
                    confidence=0.8,
                    metadata={'type': 'facial'}
                ))
        elif video_path:
            try:
                # Use cached video processor instead of creating a new one
                video_result = self.video_processor.process_video(video_path)
                face_summary = video_result.get('face_emotion_summary', {})
                face_dist = face_summary.get('emotion_distribution', {})
                if face_dist:
                    for emotion, prob in face_dist.items():
                        possibilities.append(QuantumPossibility(
                            emotion=emotion,
                            probability=prob * 0.6,
                            source='expression',
                            confidence=face_summary.get('confidence', 0.7),
                            metadata={'type': 'facial', 'from_video': True}
                        ))
            except Exception as e:
                print(f"  ⚠️ Video processing error: {e}")
        
        return possibilities
    
    def _analyze_eeg(self, eeg_path: Optional[str], eeg_data: Optional[np.ndarray]) -> List[QuantumPossibility]:
        """Analyze EEG signals for brain-state emotions"""
        possibilities = []
        try:
            if eeg_data is not None:
                eeg_result = self.eeg_processor.process_eeg_stream(eeg_data)
            elif eeg_path:
                eeg_result = self.eeg_processor.process_eeg_file(eeg_path)
            else:
                return possibilities
                
            if eeg_result:
                for emotion, prob in eeg_result.items():
                    possibilities.append(QuantumPossibility(
                        emotion=emotion,
                        probability=prob * 0.7, # Higher weight for direct brain signals
                        source='eeg',
                        confidence=0.9,
                        metadata={'type': 'brain_signal'}
                    ))
        except Exception as e:
            print(f"  ⚠️ EEG processing error: {e}")
            
        return possibilities
    
    def _create_quantum_superposition_optimized(self, possibilities: List[QuantumPossibility]) -> QuantumSuperposition:
        """Create quantum superposition with optimized calculations"""
        if not possibilities:
            return QuantumSuperposition(
                possibilities=[],
                collapsed_emotion='neutral',
                uncertainty=1.0
            )
        
        # Group possibilities by emotion (vectorized)
        emotion_groups = {}
        for poss in possibilities:
            if poss.emotion not in emotion_groups:
                emotion_groups[poss.emotion] = []
            emotion_groups[poss.emotion].append(poss)
        
        # Calculate "Energy Landscape" (Energy + Momentum + Resistance)
        quantum_amplitudes = {}
        interference_patterns = {}
        
        emotions = list(emotion_groups.keys())
        # Ensure we track all known emotions for momentum even if current input doesn't show them
        all_emotions = set(emotions) | set(self.previous_distribution.keys())
        all_emotions_list = list(all_emotions)
        
        # Vectorized amplitude calculation
        for emotion in all_emotions_list:
            # 1. Energy (Base Amplitude from current inputs)
            group = emotion_groups.get(emotion, [])
            energy = sum(p.probability * p.confidence for p in group)
            
            # 2. Momentum (From recent history)
            # Decay factor can be tuned (e.g. 0.3 means 30% retention)
            momentum = self.previous_distribution.get(emotion, 0.0) * 0.3
            
            # 3. Resistance (Interference / Friction)
            interference = 0.0
            for other_emotion in all_emotions_list:
                if other_emotion != emotion:
                    # Get energy of the conflicting emotion
                    other_group = emotion_groups.get(other_emotion, [])
                    other_energy = sum(p.probability for p in other_group)
                    
                    # Calculate friction
                    resistance_strength = self._get_interference_strength_cached(emotion, other_emotion)
                    
                    # If emotions conflict (negative resistance), it lowers energy
                    # If they resonate (positive), it boosts it
                    interference += resistance_strength * other_energy
            
            # Final Equation: State = Energy + Momentum + Resistance
            total_potential = energy + momentum + (interference * 0.15)
            quantum_amplitudes[emotion] = max(0.0, total_potential)
            interference_patterns[emotion] = interference
        
        # Normalize to probabilities (vectorized)
        total = sum(quantum_amplitudes.values())
        if total > 0:
            quantum_amplitudes = {k: max(0, v / total) for k, v in quantum_amplitudes.items()}
        else:
            quantum_amplitudes = {emotion: 1.0 / len(all_emotions_list) for emotion in all_emotions_list}
            
        # Update Momentum State
        self.previous_distribution = quantum_amplitudes.copy()
        
        # Calculate stability (Inverse of Entropy + Interference)
        probabilities = list(quantum_amplitudes.values())
        if len(probabilities) > 0:
            entropy = -sum(p * np.log(p + 1e-8) for p in probabilities if p > 0)
            max_entropy = np.log(len(probabilities)) if len(probabilities) > 1 else 1
            uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uncertainty = 1.0
            
        # Create updated possibilities
        updated_possibilities = []
        for emotion, amplitude in quantum_amplitudes.items():
            original = next((p for p in possibilities if p.emotion == emotion), None)
            updated_possibilities.append(QuantumPossibility(
                emotion=emotion,
                probability=amplitude,
                source=original.source if original else 'quantum_landscape',
                confidence=original.confidence if original else 0.5,
                metadata={**(original.metadata if original else {}), 'energy_potential': amplitude}
            ))
        
        return QuantumSuperposition(
            possibilities=updated_possibilities,
            uncertainty=uncertainty,
            interference_patterns=interference_patterns
        )
    
    def _get_interference_strength_cached(self, emotion1: str, emotion2: str) -> float:
        """Get cached interference strength"""
        cache_key = f"{emotion1}_{emotion2}"
        if cache_key in self.interference_cache:
            return self.interference_cache[cache_key]
        
        # Calculate interference strength
        interference_map = {
            ('happy', 'sad'): -0.3,
            ('angry', 'fear'): 0.4,
            ('surprise', 'neutral'): -0.2,
            ('disgust', 'happy'): -0.4,
            ('fear', 'sad'): 0.3,
        }
        
        strength = 0.0
        if (emotion1, emotion2) in interference_map:
            strength = interference_map[(emotion1, emotion2)]
        elif (emotion2, emotion1) in interference_map:
            strength = interference_map[(emotion2, emotion1)]
        
        # Cache the result
        self.interference_cache[cache_key] = strength
        return strength
    
    def _update_memory_async(self, emotion: str, context: Optional[Dict[str, Any]], text: Optional[str]):
        """Update memory asynchronously"""
        def update_task():
            try:
                self.memory.add_interaction(
                    emotion=emotion,
                    context=context or {},
                    text=text
                )
            except Exception as e:
                print(f"  ⚠️ Async memory update error: {e}")
        
        # Submit to thread pool
        self.quantum_thread_pool.submit(update_task)
    
    def _create_quantum_superposition(self, possibilities: List[QuantumPossibility]) -> QuantumSuperposition:
        """Create quantum superposition from all possibilities"""
        if not possibilities:
            return QuantumSuperposition(
                possibilities=[],
                collapsed_emotion='neutral',
                uncertainty=1.0
            )
        
        # Group possibilities by emotion
        emotion_groups = {}
        for poss in possibilities:
            if poss.emotion not in emotion_groups:
                emotion_groups[poss.emotion] = []
            emotion_groups[poss.emotion].append(poss)
        
        # Calculate quantum amplitudes with interference
        quantum_amplitudes = {}
        interference_patterns = {}
        
        for emotion, group in emotion_groups.items():
            # Base amplitude from all sources
            base_amplitude = sum(p.probability * p.confidence for p in group)
            
            # Apply interference from other emotions
            interference = 0.0
            for other_emotion, other_group in emotion_groups.items():
                if other_emotion != emotion:
                    other_amplitude = sum(p.probability for p in other_group)
                    # Get interference strength (simplified)
                    interference_strength = self._get_interference_strength_cached(emotion, other_emotion)
                    # Increased weight from 0.1 to 0.4 for higher sensitivity
                    interference += interference_strength * other_amplitude
            
            quantum_amplitudes[emotion] = base_amplitude + (interference * 0.4)
            interference_patterns[emotion] = interference
        
        # Normalize to probabilities
        total = sum(quantum_amplitudes.values())
        if total > 0:
            quantum_amplitudes = {k: max(0, v / total) for k, v in quantum_amplitudes.items()}
        else:
            quantum_amplitudes = {emotion: 1.0 / len(emotion_groups) for emotion in emotion_groups.keys()}
        
        # Calculate uncertainty (Shannon entropy)
        probabilities = list(quantum_amplitudes.values())
        entropy = -sum(p * np.log(p + 1e-8) for p in probabilities if p > 0)
        max_entropy = np.log(len(probabilities)) if len(probabilities) > 1 else 1
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        
        # Create updated possibilities with quantum amplitudes
        updated_possibilities = []
        for emotion, amplitude in quantum_amplitudes.items():
            # Find original possibility for metadata
            original = next((p for p in possibilities if p.emotion == emotion), None)
            updated_possibilities.append(QuantumPossibility(
                emotion=emotion,
                probability=amplitude,
                source=original.source if original else 'quantum',
                confidence=original.confidence if original else 0.5,
                metadata={**(original.metadata if original else {}), 'quantum_amplitude': amplitude}
            ))
        
        return QuantumSuperposition(
            possibilities=updated_possibilities,
            uncertainty=uncertainty,
            interference_patterns=interference_patterns
        )
    
    def _get_interference_strength(self, emotion1: str, emotion2: str) -> float:
        """Get interference strength between two emotions"""
        # Simplified interference lookup
        interference_map = {
            ('happy', 'sad'): -0.3,
            ('angry', 'fear'): 0.4,
            ('surprise', 'neutral'): -0.2,
            ('disgust', 'happy'): -0.4,
            ('fear', 'sad'): 0.3,
        }
        
        if (emotion1, emotion2) in interference_map:
            return interference_map[(emotion1, emotion2)]
        elif (emotion2, emotion1) in interference_map:
            return interference_map[(emotion2, emotion1)]
        return 0.0
    
    def _collapse_quantum_state(self, superposition: QuantumSuperposition) -> Tuple[str, float, Dict[str, float]]:
        """Collapse quantum superposition to primary emotion and distribution"""
        if not superposition.possibilities:
            return 'neutral', 0.5, {'neutral': 1.0}
        
        # Sort by probability
        sorted_possibilities = sorted(
            superposition.possibilities,
            key=lambda p: p.probability,
            reverse=True
        )

        # Create distribution dictionary for "Advance Emotion Logic"
        # Normalize probabilities to sum to 1
        raw_probs = {p.emotion: p.probability for p in superposition.possibilities}
        total_p = sum(raw_probs.values())
        if total_p > 0:
            distribution = {k: v / total_p for k, v in raw_probs.items()}
        else:
            distribution = {p.emotion: 1.0/len(raw_probs) for p in superposition.possibilities}
        
        # Top emotion
        top_possibility = sorted_possibilities[0]
        collapsed = top_possibility.emotion
        confidence = top_possibility.probability
        
        # --- FEAR SQUASHING & LOGIC ---
        if collapsed == 'fear' and confidence < 0.85:
             # Find next best
             if len(sorted_possibilities) > 1:
                 second = sorted_possibilities[1]
                 if second.emotion != 'fear':
                     print(f"🚫 Squashed Weak Fear ({confidence:.2f}). Fallback to: {second.emotion}")
                     collapsed = second.emotion
                     confidence = second.probability
                 else:
                     collapsed = 'neutral'
                     confidence = 0.6
             else:
                 collapsed = 'neutral'
                 confidence = 0.6
        
        # Determine Stability
        stability_score = 1.0 - superposition.uncertainty
        stability_label = "High" if stability_score > 0.7 else ("Medium" if stability_score > 0.4 else "Low")
        
        # Inject stability into metadata instead
        superposition.metadata = getattr(superposition, 'metadata', {})
        superposition.metadata['stability'] = stability_label
        superposition.metadata['stability_score'] = stability_score

        # Add slight randomness based on uncertainty (Quantum tunneling effect)
        if hasattr(superposition, 'uncertainty') and superposition.uncertainty > 0.7:
             confidence *= 0.8
             
        superposition.collapsed_emotion = collapsed
        return collapsed, confidence, distribution

    def _get_interaction_state(self, distribution: Dict[str, float], uncertainty: float) -> str:
        """Map emotion distribution to an interaction state"""
        # 1. Probing Safety Gate
        if uncertainty > 0.6:
            return "Tentative / Gentle Probing"
        
        # Get top 2 emotions
        sorted_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        if not sorted_emotions:
            return "Neutral / Attentive"
            
        top_e, top_p = sorted_emotions[0]
        try:
             sec_e, sec_p = sorted_emotions[1]
        except IndexError:
             sec_e, sec_p = "none", 0.0
             
        # 2. Logic Mapping
        if top_e == 'sad':
            if sec_e == 'neutral': return "Low-energy Supportive"
            if sec_e == 'fear': return "Reassuring / Protective"
            
        if top_e == 'anger':
            if sec_e == 'disgust': return "Calm Boundary Setting"
            if sec_e == 'sad': return "Empathetic De-escalation"
            
        if top_e == 'happy':
            if sec_e == 'surprise': return "Warm Enthusiasm"
            if sec_e == 'neutral': return "Pleasant / Professional"
            
        if top_e == 'fear':
             return "Urgent / Protective" if top_p > 0.85 else "Calm Reassurance"
             
        if top_e == 'disgust':
            return "Professional Detachment"
            
        return f"{top_e.capitalize()} / Responsive"
    
    def _generate_with_emotion_llm(self,
                                   text: Optional[str],
                                   target_emotion: str,
                                   quantum_state: QuantumSuperposition) -> Dict[str, Any]:
        """Generate output using emotion LLM"""
        if self.emotion_llm is None or text is None:
            return {
                'text': text or f"[Emotion: {target_emotion}]",
                'emotion': target_emotion,
                'confidence': 0.5,
                'method': 'fallback'
            }
        
        try:
            # Tokenize input
            inputs = self.emotion_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding='max_length'
            )
            
            # Predict emotion
            with torch.no_grad():
                outputs = self.emotion_llm(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_id].item()
                
                predicted_emotion = self.label_encoder.inverse_transform([predicted_id])[0]
            
            # --- FEAR SQUASHING LOGIC ---
            # Only allow Fear if confidence is extreme (> 0.85)
            if predicted_emotion == 'fear' and confidence < 0.85:
                # Check if there is a strong secondary emotion from the LLM's predictions
                # Get sorted probabilities and emotions from the LLM's output
                sorted_llm_predictions = sorted(
                    [(self.label_encoder.inverse_transform([i])[0], p.item()) for i, p in enumerate(predictions[0])],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if len(sorted_llm_predictions) > 1:
                    # Fallback to second best if it's not fear
                    secondary_emotion, secondary_prob = sorted_llm_predictions[1]
                    if secondary_emotion != 'fear':
                        print(f"🚫 Squashed Weak Fear ({confidence:.2f}). Fallback to: {secondary_emotion} (LLM)")
                        predicted_emotion = secondary_emotion
                        confidence = secondary_prob
                else:
                    # Default to neutral if no other strong prediction
                    print(f"🚫 Squashed Weak Fear ({confidence:.2f}). Fallback to: neutral (LLM)")
                    predicted_emotion = 'neutral'
                    confidence = 0.5
            
            # Add slight randomness based on uncertainty (Quantum tunneling effect)
            if hasattr(quantum_state, 'uncertainty') and quantum_state.uncertainty > 0.7:
                 # If highly uncertain, dampen confidence
                 confidence *= 0.8
            
            # Generate text based on emotion
            emotion_context = {
                'target_emotion': target_emotion,
                'predicted_emotion': predicted_emotion,
                'quantum_uncertainty': quantum_state.uncertainty,
                'top_possibilities': [
                    {'emotion': p.emotion, 'probability': p.probability}
                    for p in sorted(quantum_state.possibilities, key=lambda x: x.probability, reverse=True)[:3]
                ]
            }
            
            return {
                'text': text,
                'emotion': predicted_emotion,
                'target_emotion': target_emotion,
                'confidence': confidence,
                'emotion_context': emotion_context,
                'method': 'emotion_llm'
            }
            
        except Exception as e:
            print(f"⚠️ Emotion LLM error: {e}")
            return {
                'text': text or f"[Emotion: {target_emotion}]",
                'emotion': target_emotion,
                'confidence': 0.5,
                'method': 'fallback',
                'error': str(e)
            }
    
    def _format_with_ollama(self,
                           llm_output: Dict[str, Any],
                           quantum_state: QuantumSuperposition,
                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final response using Ollama with interaction states and blended emotions"""
        if not self.ollama.available:
            return {
                'original_text': llm_output.get('text', ''),
                'formatted_text': llm_output.get('text', ''),
                'target_emotion': 'neutral',
                'confidence': 0.5,
                'method': 'fallback_no_ollama'
            }

        original_text = context.get('text') if context else llm_output.get('text', '')
        
        # Extract Advanced Emotion Data
        interaction_state = llm_output.get('interaction_state', 'Responsive')
        distribution = llm_output.get('distribution', {})
        target_emotion = llm_output.get('emotion', 'neutral')
        
        # Get Conversation History Context
        history = self.conversation_history.get_recent_context(3)
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        
        # Create blend string for the LLM
        top_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        blend_str = ", ".join([f"{k}: {v:.2f}" for k, v in top_emotions])
        
        # Prepare Structured Messages for Ollama Chat API
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add historical context (user and assistant messages)
        for msg in history:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add current user input
        messages.append({"role": "user", "content": original_text})
        
        # DEBUG LOGGING
        print(f"📖 Chat API History Context: {len(history)} messages")
        
        try:
            # Use the chat API for native multi-turn support
            response = self.ollama.chat(messages=messages, temperature=0.7)
            formatted_text = response.text
            confidence = response.confidence
        except Exception as e:
            print(f"Ollama chat failed: {e}")
            formatted_text = "I'm listening..."
            confidence = 0.5
        
        return {
            'original_text': original_text,
            'formatted_text': formatted_text,
            'target_emotion': target_emotion,
            'confidence': confidence,
            'quantum_uncertainty': quantum_state.uncertainty,
            'modulation': {
                'emotion_intensity': self._calculate_emotion_intensity(target_emotion, quantum_state),
                'tone_adjustment': self._calculate_tone_adjustment(target_emotion)
            }
        }
    
    def _calculate_emotion_intensity(self, emotion: str, quantum_state: QuantumSuperposition) -> float:
        """Calculate emotion intensity based on quantum state"""
        # Find probability of the emotion
        emotion_prob = next(
            (p.probability for p in quantum_state.possibilities if p.emotion == emotion),
            0.5
        )
        
        # Intensity is inversely related to uncertainty
        intensity = emotion_prob * (1 - quantum_state.uncertainty)
        return float(np.clip(intensity, 0, 1))
    
    def _calculate_tone_adjustment(self, emotion: str) -> Dict[str, float]:
        """Calculate tone adjustment parameters"""
        tone_map = {
            'happy': {'warmth': 0.8, 'energy': 0.7, 'formality': -0.3},
            'sad': {'warmth': 0.6, 'energy': -0.5, 'formality': 0.2},
            'angry': {'warmth': -0.3, 'energy': 0.8, 'formality': 0.1},
            'fear': {'warmth': 0.3, 'energy': -0.4, 'formality': 0.4},
            'surprise': {'warmth': 0.5, 'energy': 0.9, 'formality': -0.2},
            'neutral': {'warmth': 0.5, 'energy': 0.0, 'formality': 0.0}
        }
        
        return tone_map.get(emotion, {'warmth': 0.5, 'energy': 0.0, 'formality': 0.0})


def main():
    """Example usage of Quantum Emotion Engine"""
    print("🌌 Quantum Emotion Engine - Example Usage")
    print("=" * 70)
    
    # Initialize engine
    engine = QuantumEmotionEngine()
    
    # Example 1: Text input
    print("\n📝 Example 1: Text Input")
    result1 = engine.process_input(
        text="I'm feeling really excited about this new project!",
        context={'situation': 'work_context', 'time': 'morning'}
    )
    
    print(f"\nResult:")
    print(f"  Collapsed Emotion: {result1['quantum_superposition']['collapsed_emotion']}")
    print(f"  Formatted Output: {result1['final_output']['formatted_text']}")
    
    # Example 2: With conversation history
    print("\n\n💬 Example 2: With Conversation History")
    engine.conversation_history.add_message('user', 'I was sad yesterday', 'sad')
    engine.conversation_history.add_message('assistant', 'I understand', 'neutral')
    
    result2 = engine.process_input(
        text="But today I'm feeling much better!",
        context={'situation': 'social_context'}
    )
    
    print(f"\nResult:")
    print(f"  Collapsed Emotion: {result2['quantum_superposition']['collapsed_emotion']}")
    print(f"  Formatted Output: {result2['final_output']['formatted_text']}")
    
    print("\n" + "=" * 70)
    print("✅ Examples Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

