"""
Unit tests for data preprocessing
"""
import pytest
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.data_preprocessing import preprocess_data, load_mnist_data

class TestDataPreprocessing:
    def test_preprocess_data(self):
        # Create sample data
        X_sample = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        Y_sample = np.random.randint(0, 10, (100,))
        
        X_processed, Y_processed = preprocess_data(X_sample, Y_sample)
        
        # Test shapes
        assert X_processed.shape == (100, 784)
        assert Y_processed.shape == (100, 10)
        
        # Test normalization
        assert tf.reduce_max(X_processed) <= 1.0
        assert tf.reduce_min(X_processed) >= 0.0
        
        # Test one-hot encoding
        assert tf.reduce_all(tf.reduce_sum(Y_processed, axis=1) == 1.0)

if __name__ == "__main__":
    pytest.main([__file__])