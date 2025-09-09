# src/__init__.py
"""
Softmax Regression with TensorFlow
A comprehensive implementation for multi-class classification
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model.softmax_regression import SoftmaxRegression
from .data.data_preprocessing import load_and_preprocess_mnist, preprocess_data
from .training.trainer import ModelTrainer
from .training.custom_trainer import CustomTrainer

__all__ = [
    'SoftmaxRegression',
    'load_and_preprocess_mnist',
    'preprocess_data',
    'ModelTrainer',
    'CustomTrainer'
]