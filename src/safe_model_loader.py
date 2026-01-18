#!/usr/bin/env python3
"""
Safe model loading utilities for version compatibility
"""

import joblib
import warnings
from pathlib import Path
from typing import Any, Optional
import sklearn
import numpy as np


def safe_load_label_encoder(encoder_path: str) -> Optional[Any]:
    """
    Safely load label encoder with version compatibility handling
    
    Args:
        encoder_path: Path to the label encoder file
        
    Returns:
        Loaded label encoder or None if loading fails
    """
    try:
        # Suppress sklearn version warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*version mismatch.*")
            
            encoder = joblib.load(encoder_path)
            
            # Test if encoder works
            if hasattr(encoder, 'classes_'):
                print(f"✅ Label encoder loaded successfully")
                print(f"   Classes: {list(encoder.classes_)}")
                return encoder
            else:
                print(f"⚠️ Loaded object is not a valid label encoder")
                return None
                
    except Exception as e:
        print(f"⚠️ Error loading label encoder: {e}")
        
        # Try to create a fallback encoder
        try:
            from sklearn.preprocessing import LabelEncoder
            fallback_encoder = LabelEncoder()
            # Set proper emotion classes for face expressions
            fallback_encoder.classes_ = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
            print(f"✅ Created fallback label encoder with face expression classes")
            return fallback_encoder
        except Exception as fallback_error:
            print(f"❌ Failed to create fallback encoder: {fallback_error}")
            return None


def safe_load_keras_model(model_path: str, custom_objects: Optional[dict] = None, compile_model: bool = False) -> Optional[Any]:
    """
    Safely load Keras model with comprehensive error handling and custom layer support
    
    Args:
        model_path: Path to the model file
        custom_objects: Additional custom objects for model loading (merged with defaults)
        compile_model: Whether to compile the model after loading (default: False to avoid errors)
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        from tensorflow.keras.models import load_model
        from src.custom_layers import get_custom_objects
        
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Get comprehensive custom objects from custom_layers
        all_custom_objects = get_custom_objects()
        
        # Merge with any additional custom objects provided
        if custom_objects:
            all_custom_objects.update(custom_objects)
        
        # Load model with compile=False to avoid compilation errors
        model = load_model(
            model_path, 
            custom_objects=all_custom_objects,
            compile=compile_model  # Skip compilation to avoid optimizer/loss errors
        )
        
        print(f"✅ Keras model loaded successfully from {Path(model_path).name}")
        return model
        
    except Exception as e:
        print(f"⚠️ Error loading Keras model from {Path(model_path).name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def safe_load_model(model_path: str, custom_objects: Optional[dict] = None) -> Optional[Any]:
    """
    Legacy wrapper for safe_load_keras_model for backward compatibility
    
    Args:
        model_path: Path to the model file
        custom_objects: Custom objects for model loading
        
    Returns:
        Loaded model or None if loading fails
    """
    return safe_load_keras_model(model_path, custom_objects, compile_model=False)


def check_sklearn_compatibility() -> bool:
    """
    Check scikit-learn version compatibility
    
    Returns:
        True if compatible, False otherwise
    """
    try:
        sklearn_version = sklearn.__version__
        major, minor = map(int, sklearn_version.split('.')[:2])
        
        # Check if version is compatible (expecting 1.7.2 but model saved with 1.8.0)
        if major == 1 and minor >= 7:
            print(f"✅ Scikit-learn version {sklearn_version} is compatible")
            return True
        else:
            print(f"⚠️ Scikit-learn version {sklearn_version} may have compatibility issues")
            return False
            
    except Exception as e:
        print(f"❌ Error checking scikit-learn version: {e}")
        return False


def get_model_info(model_path: str) -> dict:
    """
    Get information about a model file
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model information
    """
    try:
        model_file = Path(model_path)
        info = {
            'path': str(model_file.absolute()),
            'exists': model_file.exists(),
            'size': model_file.stat().st_size if model_file.exists() else 0,
            'modified': model_file.stat().st_mtime if model_file.exists() else None
        }
        
        if model_file.exists():
            if model_file.suffix == '.keras':
                # Try to get Keras model info
                try:
                    import h5py
                    with h5py.File(model_path, 'r') as f:
                        if 'model_config' in f.attrs:
                            info['type'] = 'Keras'
                            info['config'] = f.attrs['model_config']
                except:
                    pass
            elif model_file.suffix == '.pkl':
                info['type'] = 'Pickle'
            else:
                info['type'] = 'Unknown'
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


# Compatibility check at import
check_sklearn_compatibility()
