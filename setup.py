#!/usr/bin/env python3
"""
Setup script for Quantum Emotion Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="quantum-emotion-pipeline",
    version="1.0.0",
    description="Quantum-inspired multi-modal emotion detection and processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="QBRAINS Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "tensorflow>=2.10.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.6.0",
        "pandas>=1.3.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "Pillow>=9.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "accelerate>=0.26.0",
        "requests>=2.28.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "sounddevice>=0.4.6",
        "joblib>=1.1.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "build": [
            "pyinstaller>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "quantum-emotion=desktop_app:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)

