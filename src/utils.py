#!/usr/bin/env python3
"""
Utility functions for Quantum Emotion Pipeline
Handles path resolution, configuration, and shared utilities
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json


def get_project_root() -> Path:
    """Get the project root directory"""
    # If running as executable, use sys._MEIPASS (PyInstaller temp directory)
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        # PyInstaller sets _MEIPASS to the temp directory where it extracts files
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS)
        else:
            # Fallback: use executable directory
            return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).parent.parent


def get_model_path(relative_path: str) -> Path:
    """Resolve model path relative to project root"""
    project_root = get_project_root()
    model_path = project_root / "model" / Path(relative_path).name
    
    # Try relative path first (for development)
    if not model_path.exists():
        # Try absolute path from project root
        alt_path = project_root / relative_path.lstrip('/')
        if alt_path.exists():
            return alt_path
    
    return model_path


def get_data_path(relative_path: str) -> Path:
    """Resolve data path relative to project root"""
    project_root = get_project_root()
    return project_root / "data" / relative_path


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if not"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or return defaults"""
    if config_path is None:
        # Try multiple locations
        project_root = get_project_root()
        possible_paths = [
            project_root / "config.json",
            project_root.parent / "config.json",  # In case we're in src/
            Path.cwd() / "config.json"  # Current working directory
        ]
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            config_path = project_root / "config.json"
    
    default_config = {
        "models": {
            "face_model": "improved_expression_model.keras",
            "audio_model": "sound_emotion_detector.keras",
            "audio_scaler": "sound_emotion_scaler.pkl",
            "audio_encoder": "sound_emotion_label_encoder.pkl",
            "emotion_llm": "emotion_llm_final",
            "emotion_llm_encoder": "emotion_label_encoder.pkl"
        },
        "quantum": {
            "face_weight": 0.4,
            "voice_weight": 0.4,
            "text_weight": 0.2,
            "uncertainty_threshold": 0.3,
            "collapse_threshold": 0.7
        },
        "audio": {
            "sample_rate": 22050,
            "duration": 3,
            "n_mfcc": 40,
            "n_fft": 2048,
            "hop_length": 512,
            "max_pad_length": 173,
            "n_time_steps": 173,
            "n_features_per_step": 62
        },
        "video": {
            "frame_interval": 30,
            "min_face_size": 48
        },
        "ollama": {
            "enabled": True,
            "base_url": "http://localhost:11434",
            "model": "mistral:latest",
            "timeout": 30
        },
        "memory": {
            "max_interactions": 1000,
            "pattern_window": 10
        }
    }
    
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        except Exception as e:
            print(f"⚠️ Error loading config: {e}, using defaults")
    
    return default_config


def validate_model_path(path: Path, model_name: str) -> bool:
    """Validate that model file exists"""
    if not path.exists():
        print(f"⚠️ {model_name} not found at: {path}")
        return False
    return True


def safe_path_join(*parts: str) -> Path:
    """Safely join path parts"""
    return Path(*parts)

