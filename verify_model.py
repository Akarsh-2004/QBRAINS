
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, InputLayer
import joblib

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def verify_emotion_model():
    print("🔍 Verifying Emotion LLM Model...")
    model_path = "model/emotion_llm_final"
    encoder_path = "model/emotion_label_encoder.pkl"
    
    # 1. Check files
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder path not found: {encoder_path}")
        return False
        
    print("✅ Model files present")

    # 2. Try loading Label Encoder
    try:
        le = joblib.load(encoder_path)
        print(f"✅ Label Encoder loaded. Classes: {le.classes_}")
    except Exception as e:
        print(f"❌ Label Encoder failed: {e}")
        return False

    # 3. Try loading Keras Model
    try:
        # Custom object handling for the deserialization error
        from src.custom_layers import CustomDense, CustomInputLayer
        custom_objects = {
            'CustomDense': CustomDense,
            'CustomInputLayer': CustomInputLayer,
            'Dense': CustomDense, # HACK: Map standard Dense to Custom to catch config errors
            'InputLayer': CustomInputLayer
        }
        
        # Attempt load
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ Keras Model loaded successfully (with custom layer fix)")
        
        # 4. Basic Inference Test
        print("🧪 Running inference test...")
        # Assuming input shape from error logs: (None, 173, 62) - this looks like audio features?
        # Or text? The previous code treated it as text...
        # The logs showed "Conv1D", so it's likely an Audio/Signal model, NOT a text LLM.
        # This confirms "Emotion LLM" is a misnomer in the code, it's a CNN.
        
        dummy_input = np.random.random((1, 173, 62)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        predicted_idx = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_idx])[0]
        
        print(f"✅ Inference successful. Dummy input -> {predicted_label}")
        return True

    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_emotion_model()
    sys.exit(0 if success else 1)
