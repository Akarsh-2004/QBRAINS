#!/usr/bin/env python3
"""
Custom Keras layers for model loading compatibility
Handles version mismatches and unknown parameters during deserialization
"""

try:
    from tensorflow.keras.layers import (
        Dense, InputLayer, Conv1D, Conv2D, LSTM, 
        BatchNormalization, Dropout, MaxPooling1D, MaxPool2D
    )
    from tensorflow.keras.utils import register_keras_serializable
    import logging
    
    logger = logging.getLogger(__name__)
    
    @register_keras_serializable(package='Custom', name='Dense')
    class CustomDense(Dense):
        """Custom Dense layer that handles quantization_config and other parameters"""
        
        def __init__(self, *args, **kwargs):
            # Remove problematic parameters that don't exist in older Keras versions
            problematic_keys = [
                'quantization_config', 
                'registered_name', 
                'shared_object_id',
                'activity_regularizer'  # Sometimes causes issues
            ]
            removed_keys = []
            for key in problematic_keys:
                if key in kwargs:
                    kwargs.pop(key)
                    removed_keys.append(key)
            
            if removed_keys:
                logger.debug(f"CustomDense: Removed parameters {removed_keys}")
            
            # Handle dtype policy (newer Keras uses DTypePolicy objects)
            if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
                dtype_config = kwargs.pop('dtype')
                if isinstance(dtype_config, dict) and 'name' in dtype_config:
                    kwargs['dtype'] = dtype_config['name']
            
            # Handle nested config objects (initializers, regularizers)
            for key in ['kernel_initializer', 'bias_initializer', 'kernel_regularizer', 
                       'bias_regularizer', 'recurrent_initializer', 'recurrent_regularizer']:
                if key in kwargs and isinstance(kwargs[key], dict):
                    config = kwargs[key]
                    if isinstance(config, dict) and 'class_name' in config:
                        # Extract class name and config
                        class_name = config['class_name']
                        inner_config = config.get('config', {})
                        
                        # Create proper initializer/regularizer
                        if class_name == 'GlorotUniform':
                            from tensorflow.keras.initializers import GlorotUniform
                            kwargs[key] = GlorotUniform(**inner_config)
                        elif class_name == 'Zeros':
                            from tensorflow.keras.initializers import Zeros
                            kwargs[key] = Zeros()
                        elif class_name == 'Ones':
                            from tensorflow.keras.initializers import Ones
                            kwargs[key] = Ones()
                        elif class_name == 'Orthogonal':
                            from tensorflow.keras.initializers import Orthogonal
                            kwargs[key] = Orthogonal(**inner_config)
                        elif class_name == 'L2':
                            from tensorflow.keras.regularizers import L2
                            l2_config = inner_config.get('l2', 0.01)
                            kwargs[key] = L2(l2_config)
            
            super().__init__(*args, **kwargs)
    
    @register_keras_serializable(package='Custom', name='InputLayer')
    class CustomInputLayer(InputLayer):
        """Custom InputLayer that handles batch_shape and other parameters"""
        
        def __init__(self, *args, **kwargs):
            # Remove problematic parameters
            problematic_keys = ['optional', 'registered_name', 'shared_object_id']
            for key in problematic_keys:
                if key in kwargs:
                    kwargs.pop(key)
            
            # Handle dtype policy
            if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
                dtype_config = kwargs.pop('dtype')
                if isinstance(dtype_config, dict) and 'name' in dtype_config:
                    kwargs['dtype'] = dtype_config['name']
            
            # Convert batch_shape to input_shape if needed
            if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape is not None:
                    # Remove the first dimension (batch size) from batch_shape
                    kwargs['input_shape'] = batch_shape[1:] if len(batch_shape) > 1 else batch_shape
            
            super().__init__(*args, **kwargs)
    
    logger.info("✓ Custom Keras layers registered successfully")
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ TensorFlow/Keras not available: {e}")
    CustomDense = None
    CustomInputLayer = None


# Helper function to get comprehensive custom objects
# Define outside try block so it's always available
def get_custom_objects():
    """
    Get comprehensive custom objects dictionary for model loading.
    Maps all possible layer name variants to custom implementations.
    """
    if CustomDense is None or CustomInputLayer is None:
        return {}
    
    custom_objs = {
        # Direct class references
        'Dense': CustomDense,
        'InputLayer': CustomInputLayer,
        # Module-qualified names (Keras 2.x style)
        'keras.layers.Dense': CustomDense,
        'keras.layers.InputLayer': CustomInputLayer,
        # Module-qualified names (Keras 3.x / tf.keras style)
        'keras.layers.core.dense.Dense': CustomDense,
        'keras.src.layers.core.dense.Dense': CustomDense,
        'tensorflow.keras.layers.Dense': CustomDense,
        'tensorflow.keras.layers.InputLayer': CustomInputLayer,
        # Registered names
        'Custom>Dense': CustomDense,
        'Custom>InputLayer': CustomInputLayer,
    }
    return custom_objs

