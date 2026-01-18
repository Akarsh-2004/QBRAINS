#!/usr/bin/env python3
"""
Input validation utilities
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import os


def validate_file_path(file_path: str, file_type: str = "file") -> Tuple[bool, Optional[str]]:
    """
    Validate that a file path exists
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path:
        return False, f"{file_type} path is required"
    
    path = Path(file_path)
    if not path.exists():
        return False, f"{file_type} not found: {file_path}"
    
    if not path.is_file():
        return False, f"{file_type} is not a file: {file_path}"
    
    return True, None


def validate_text_input(text: Optional[str], min_length: int = 1) -> Tuple[bool, Optional[str]]:
    """
    Validate text input
    
    Returns:
        (is_valid, error_message)
    """
    if text is None:
        return False, "Text input is required"
    
    if not isinstance(text, str):
        return False, "Text input must be a string"
    
    text = text.strip()
    if len(text) < min_length:
        return False, f"Text must be at least {min_length} character(s)"
    
    return True, None


def validate_video_path(video_path: str) -> Tuple[bool, Optional[str]]:
    """Validate video file path"""
    valid, error = validate_file_path(video_path, "Video")
    if not valid:
        return valid, error
    
    # Check extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    if Path(video_path).suffix.lower() not in valid_extensions:
        return False, f"Unsupported video format. Supported: {', '.join(valid_extensions)}"
    
    return True, None


def validate_audio_path(audio_path: str) -> Tuple[bool, Optional[str]]:
    """Validate audio file path"""
    valid, error = validate_file_path(audio_path, "Audio")
    if not valid:
        return valid, error
    
    # Check extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    if Path(audio_path).suffix.lower() not in valid_extensions:
        return False, f"Unsupported audio format. Supported: {', '.join(valid_extensions)}"
    
    return True, None


def validate_eeg_path(eeg_path: str) -> Tuple[bool, Optional[str]]:
    """Validate EEG file path"""
    valid, error = validate_file_path(eeg_path, "EEG")
    if not valid:
        return valid, error
    
    # Check extension
    valid_extensions = {'.csv', '.txt', '.edf', '.bdf'}
    if Path(eeg_path).suffix.lower() not in valid_extensions:
        return False, f"Unsupported EEG format. Supported: {', '.join(valid_extensions)}"
    
    return True, None


def sanitize_path(path: str) -> Path:
    """Sanitize and normalize a file path"""
    return Path(path).resolve()

