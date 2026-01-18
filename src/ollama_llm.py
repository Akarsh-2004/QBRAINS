#!/usr/bin/env python3
"""
Ollama LLM Integration for Quantum Emotion Pipeline
Handles prediction and reframing of outputs using local LLM
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structure for LLM responses"""
    text: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OllamaLLM:
    """Integration with Ollama for LLM-based prediction and reframing"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma2:2b"):
        """
        Initialize Ollama LLM client
        
        Args:
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            model: Model name to use (default: gemma2:2b for speed)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is running and accessible"""
        self.available = False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                print(f"✓ Ollama connected. Available models: {available_models}")
                if self.model not in available_models:
                    print(f"⚠️ Warning: Model '{self.model}' not found. Using first available model.")
                    if available_models:
                        self.model = available_models[0]
                self.available = True
            else:
                print(f"⚠️ Warning: Ollama API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Warning: Cannot connect to Ollama at {self.base_url}")
            print("  Ollama features will be disabled. Install: https://ollama.ai")
            self.available = False
        except Exception as e:
            print(f"⚠️ Warning: Error checking Ollama connection: {e}")
            self.available = False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 200) -> LLMResponse:
        """
        Generate text using Ollama
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse object
        """
        if not self.available:
            return LLMResponse(
                text="[Ollama not available - LLM features disabled]",
                confidence=0.0,
                metadata={'error': 'Ollama not connected'}
            )
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make request with longer timeout
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300  # Increased to 5 minutes for slow models
            )
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    text=result.get('response', ''),
                    confidence=0.8,  # Ollama doesn't provide confidence scores
                    metadata={
                        'model': self.model,
                        'total_duration': result.get('total_duration', 0),
                        'load_duration': result.get('load_duration', 0),
                        'prompt_eval_count': result.get('prompt_eval_count', 0),
                        'eval_count': result.get('eval_count', 0)
                    }
                )
            else:
                return LLMResponse(
                    text=f"Error: {response.status_code} - {response.text}",
                    confidence=0.0
                )
                
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                text="Error: Cannot connect to Ollama. Make sure it's running.",
                confidence=0.0
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                confidence=0.0
            )
    
    def chat(self, messages: List[Dict[str, str]], 
             temperature: float = 0.7) -> LLMResponse:
        """
        Chat with Ollama using message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            LLMResponse object
        """
        if not self.available:
            return LLMResponse(
                text="[Ollama not available - LLM features disabled]",
                confidence=0.0,
                metadata={'error': 'Ollama not connected'}
            )
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=300  # Increased to 5 minutes for slow models
            )
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    text=result.get('message', {}).get('content', ''),
                    confidence=0.8,
                    metadata={
                        'model': self.model,
                        'total_duration': result.get('total_duration', 0)
                    }
                )
            else:
                return LLMResponse(
                    text=f"Error: {response.status_code}",
                    confidence=0.0
                )
                
        except Exception as e:
            return LLMResponse(
                text=f"Error in chat: {str(e)}",
                confidence=0.0
            )
    
    def predict_next_output(self, emotion_context: Dict[str, Any], 
                           conversation_history: List[str],
                           quantum_state: Dict[str, Any]) -> LLMResponse:
        """
        Predict what type of output will be next based on context
        
        Args:
            emotion_context: Current emotional context
            conversation_history: Recent conversation messages
            quantum_state: Quantum emotion state
            
        Returns:
            LLMResponse with prediction
        """
        system_prompt = """You are an advanced AI that predicts emotional responses and conversation flow.
Analyze the emotional context, conversation history, and quantum emotion state to predict:
1. What type of response is likely next (emotional, factual, question, etc.)
2. The emotional tone of the next response
3. Key topics or themes that might emerge
4. Whether the conversation is escalating, de-escalating, or stable

Provide a structured prediction in JSON format."""
        
        prompt = f"""Based on the following context, predict what type of output will come next:

EMOTION CONTEXT:
{json.dumps(emotion_context, indent=2)}

QUANTUM EMOTION STATE:
{json.dumps(quantum_state, indent=2)}

CONVERSATION HISTORY (last 5 messages):
{chr(10).join(conversation_history[-5:])}

Provide your prediction in this format:
{{
    "predicted_output_type": "emotional/factual/question/statement",
    "predicted_emotion": "emotion name",
    "predicted_topics": ["topic1", "topic2"],
    "conversation_direction": "escalating/de-escalating/stable",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}"""
        
        return self.generate(prompt, system_prompt=system_prompt, temperature=0.5)
    
    def reframe_output(self, original_text: str, 
                      target_emotion: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Reframe text to better match emotional context or improve clarity
        
        Args:
            original_text: Original text to reframe
            target_emotion: Desired emotional tone (optional)
            context: Additional context for reframing
            
        Returns:
            LLMResponse with reframed text
        """
        system_prompt = """You are an expert at reframing text to improve emotional clarity, 
tone, and context-appropriateness while preserving the original meaning.
Your reframing should:
1. Maintain the core message
2. Improve emotional clarity
3. Match the target emotional tone if specified
4. Enhance readability and naturalness
5. Consider the provided context"""
        
        prompt = f"""Reframe the following text to improve its emotional clarity and appropriateness:

ORIGINAL TEXT:
{original_text}
"""
        
        if target_emotion:
            prompt += f"\nTARGET EMOTION: {target_emotion}\n"
        
        if context:
            prompt += f"\nCONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        prompt += "\nProvide the reframed version that is clearer, more emotionally appropriate, and natural."
        
        return self.generate(prompt, system_prompt=system_prompt, temperature=0.7)
    
    def analyze_emotion_context(self, text: str, 
                               face_emotion: Optional[str] = None,
                               voice_emotion: Optional[str] = None) -> LLMResponse:
        """
        Analyze emotional context from text, optionally combining with face/voice
        
        Args:
            text: Text to analyze
            face_emotion: Detected face emotion (optional)
            voice_emotion: Detected voice emotion (optional)
            
        Returns:
            LLMResponse with emotional analysis
        """
        system_prompt = """You are an expert at analyzing emotional context from text and multimodal inputs.
Analyze the emotional content, sentiment, and underlying meaning."""
        
        prompt = f"""Analyze the emotional context of this text:

TEXT:
{text}
"""
        
        if face_emotion:
            prompt += f"\nDETECTED FACE EMOTION: {face_emotion}\n"
        
        if voice_emotion:
            prompt += f"\nDETECTED VOICE EMOTION: {voice_emotion}\n"
        
        prompt += """\nProvide analysis including:
1. Primary emotion detected
2. Sentiment (positive/negative/neutral)
3. Emotional intensity
4. Any contradictions or nuances
5. Contextual factors affecting emotion"""
        
        return self.generate(prompt, system_prompt=system_prompt, temperature=0.5)


    def analyze_sentiment(self, text: str, model: str = "gemma2:2b", history_context: List[str] = []) -> Dict[str, Any]:
        """
        Rapidly analyze sentiment using a lightweight model, considering history
        """
        if not self.available:
            return {'emotion': 'neutral', 'confidence': 0.0}

        # Quick check for greetings and neutral phrases
        text_lower = text.lower().strip()
        greetings = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
                     'good evening', 'howdy', 'sup', 'yo', 'hiya'}
        neutral_phrases = {'ok', 'okay', 'yes', 'no', 'sure', 'alright', 'fine', 'thanks', 'thank you'}
        
        # If it's just a greeting or very short neutral phrase, return neutral immediately
        if text_lower in greetings or text_lower in neutral_phrases:
            return {'emotion': 'neutral', 'confidence': 0.9, 'reason': 'greeting_or_neutral_phrase'}
        
        # If very short (< 5 chars) and no strong indicators, default to neutral
        if len(text_lower) < 5:
            return {'emotion': 'neutral', 'confidence': 0.7, 'reason': 'too_short'}

        # Override model just for this call
        original_model = self.model
        self.model = model
        
        # Format history for prompt
        history_str = "\n".join(history_context[-3:]) if history_context else "None"
        
        system_prompt = f"""You detect emotions, sarcasm, and mixed states.
History:
{history_str}

Rules:
1. Greetings like "hi", "hello" are NEUTRAL not happy.
2. Contradiction with history = 'sarcasm'.
3. Physical pain + happy = 'mixed'.
4. 'headache' != 'fear'. Only threat = 'fear'.
5. Default to 'neutral' if ambiguous.
6. JSON ONLY: {{ "emotion": "happy/sad/anger/fear/neutral/mixed/sarcastic", "confidence": 0.0-1.0 }}
"""
        
        prompt = f"Input: {text}"
        
        try:
            # Reduced tokens further for speed
            response = self.generate(prompt, system_prompt=system_prompt, temperature=0.3, max_tokens=60)
            
            # Simple parsing of JSON response
            text_resp = response.text.strip()
            # extract json if wrapped in backticks
            if "```json" in text_resp:
                text_resp = text_resp.split("```json")[1].split("```")[0]
            elif "```" in text_resp:
                text_resp = text_resp.split("```")[1].split("```")[0]
                
            data = json.loads(text_resp)
            self.model = original_model # Restore original model
            return data
            
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            self.model = original_model
            return {'emotion': 'neutral', 'confidence': 0.0}

def test_ollama():
    """Test Ollama integration"""
    print("🤖 Testing Ollama LLM Integration")
    print("=" * 50)
    
    llm = OllamaLLM()
    
    # Test basic generation
    print("\n1. Testing basic generation...")
    response = llm.generate("What is quantum computing?", max_tokens=50)
    print(f"Response: {response.text[:200]}...")
    
    print("\n✓ Ollama integration test complete!")


if __name__ == "__main__":
    test_ollama()

