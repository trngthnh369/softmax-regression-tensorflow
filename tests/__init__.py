"""
Test suite for Softmax Regression TensorFlow project

This package contains unit tests for all modules in the project:
- Model tests (test_model.py)
- Data preprocessing tests (test_preprocessing.py)  
- Training tests (test_training.py)

Run tests with:
    pytest tests/ -v
    python -m pytest tests/ --cov=src
"""

import sys
import os

# Add src to Python path for testing
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test configuration
import pytest

def pytest_configure(config):
    """Configure pytest with custom settings"""
    import warnings
    # Ignore TensorFlow warnings during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*TensorFlow.*")

# Common test fixtures that can be used across test modules
@pytest.fixture
def sample_mnist_data():
    """Create sample MNIST-like data for testing"""
    import numpy as np
    import tensorflow as tf
    
    # Create fake MNIST data
    X = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
    Y = np.random.randint(0, 10, (100,))
    
    return X, Y

@pytest.fixture  
def processed_sample_data():
    """Create preprocessed sample data for testing"""
    import numpy as np
    import tensorflow as tf
    
    # Create processed data (normalized and flattened)
    X = tf.random.normal((50, 784))
    Y = tf.one_hot(tf.random.uniform((50,), maxval=10, dtype=tf.int32), 10)
    
    return X, Y

@pytest.fixture
def simple_model():
    """Create a simple compiled model for testing"""
    from src.model.softmax_regression import SoftmaxRegression
    
    model_builder = SoftmaxRegression(input_size=784, num_classes=10)
    model = model_builder.create_and_compile(learning_rate=0.01)
    
    return model

@pytest.fixture
def small_dataset():
    """Create a small dataset for quick training tests"""
    import tensorflow as tf
    
    X_train = tf.random.normal((20, 784))
    Y_train = tf.one_hot(tf.random.uniform((20,), maxval=10, dtype=tf.int32), 10)
    X_val = tf.random.normal((10, 784)) 
    Y_val = tf.one_hot(tf.random.uniform((10,), maxval=10, dtype=tf.int32), 10)
    
    return (X_train, Y_train), (X_val, Y_val)

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_tensor_shape(tensor, expected_shape):
        """Assert tensor has expected shape"""
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_tensor_range(tensor, min_val=0.0, max_val=1.0):
        """Assert tensor values are in expected range"""
        import tensorflow as tf
        assert tf.reduce_min(tensor) >= min_val, f"Minimum value {tf.reduce_min(tensor)} < {min_val}"
        assert tf.reduce_max(tensor) <= max_val, f"Maximum value {tf.reduce_max(tensor)} > {max_val}"
    
    @staticmethod
    def assert_probabilities_sum_to_one(probabilities, tolerance=1e-5):
        """Assert probability distributions sum to 1"""
        import tensorflow as tf
        sums = tf.reduce_sum(probabilities, axis=-1)
        assert tf.reduce_all(tf.abs(sums - 1.0) < tolerance), "Probabilities don't sum to 1"

# Export test utilities for use in test modules
__all__ = ['TestUtils']