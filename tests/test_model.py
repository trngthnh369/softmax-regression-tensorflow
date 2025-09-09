"""
Unit tests for Softmax Regression model
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.softmax_regression import SoftmaxRegression


class TestSoftmaxRegression:
    """Test cases for SoftmaxRegression class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.input_size = 784
        self.num_classes = 10
        self.model_builder = SoftmaxRegression(self.input_size, self.num_classes)
        
    def test_init(self):
        """Test model initialization"""
        assert self.model_builder.input_size == 784
        assert self.model_builder.num_classes == 10
        assert self.model_builder.model is None
        
    def test_create_model(self):
        """Test model creation"""
        model = self.model_builder.create_model()
        
        # Check model structure
        assert isinstance(model, keras.Model)
        assert len(model.layers) == 2  # Dense + Activation layers
        
        # Check input/output shapes
        assert model.input_shape == (None, 784)
        assert model.output_shape == (None, 10)
        
        # Test with sample input
        sample_input = tf.random.normal((5, 784))
        output = model(sample_input)
        
        assert output.shape == (5, 10)
        # Check softmax output (sum should be ~1)
        assert np.allclose(tf.reduce_sum(output, axis=1).numpy(), 1.0, atol=1e-5)
        
    def test_compile_model(self):
        """Test model compilation"""
        model = self.model_builder.create_model()
        compiled_model = self.model_builder.compile_model(
            model, learning_rate=0.01, optimizer='sgd'
        )
        
        # Check if model is compiled
        assert compiled_model.optimizer is not None
        assert compiled_model.loss is not None
        assert len(compiled_model.metrics) > 0
        
    def test_create_and_compile(self):
        """Test create and compile in one step"""
        model = self.model_builder.create_and_compile(
            learning_rate=0.01,
            optimizer='adam',
            loss='categorical_crossentropy'
        )
        
        assert isinstance(model, keras.Model)
        assert model.optimizer is not None
        assert model.loss == 'categorical_crossentropy'
        
    def test_different_optimizers(self):
        """Test different optimizers"""
        model = self.model_builder.create_model()
        
        # Test SGD
        sgd_model = self.model_builder.compile_model(
            model, optimizer='sgd', learning_rate=0.01
        )
        assert isinstance(sgd_model.optimizer, keras.optimizers.SGD)
        
        # Test Adam
        adam_model = self.model_builder.compile_model(
            model, optimizer='adam', learning_rate=0.001
        )
        assert isinstance(adam_model.optimizer, keras.optimizers.Adam)
        
    def test_model_prediction(self):
        """Test model predictions"""
        model = self.model_builder.create_and_compile()
        
        # Test prediction
        sample_input = tf.random.normal((10, 784))
        predictions = self.model_builder.predict(sample_input)
        
        assert predictions.shape == (10, 10)
        assert np.all(predictions.numpy() >= 0)  # Probabilities should be non-negative
        assert np.all(predictions.numpy() <= 1)  # Probabilities should be <= 1
        
    def test_get_weights(self):
        """Test getting model weights"""
        model = self.model_builder.create_model()
        weights_dict = self.model_builder.get_weights()
        
        assert 'weights' in weights_dict
        assert 'biases' in weights_dict
        assert weights_dict['weights'].shape == (784, 10)
        assert weights_dict['biases'].shape == (10,)
        
    def test_set_weights(self):
        """Test setting model weights"""
        model = self.model_builder.create_model()
        
        # Get original weights
        original_weights = self.model_builder.get_weights()
        
        # Create new weights
        new_weights = {
            'weights': np.random.random((784, 10)).astype(np.float32),
            'biases': np.random.random(10).astype(np.float32)
        }
        
        # Set new weights
        self.model_builder.set_weights(new_weights)
        
        # Check if weights were set correctly
        updated_weights = self.model_builder.get_weights()
        assert np.allclose(updated_weights['weights'], new_weights['weights'])
        assert np.allclose(updated_weights['biases'], new_weights['biases'])
        
    def test_model_without_creation(self):
        """Test methods that require model creation"""
        with pytest.raises(ValueError, match="Model not created yet"):
            self.model_builder.get_model_summary()
            
        with pytest.raises(ValueError, match="Model not created yet"):
            self.model_builder.get_weights()
            
        with pytest.raises(ValueError, match="Model not created yet"):
            self.model_builder.predict(tf.random.normal((1, 784)))
            
    def test_different_input_output_sizes(self):
        """Test model with different input/output sizes"""
        custom_model = SoftmaxRegression(input_size=100, num_classes=5)
        model = custom_model.create_model()
        
        assert model.input_shape == (None, 100)
        assert model.output_shape == (None, 5)
        
        # Test with appropriate input
        sample_input = tf.random.normal((3, 100))
        output = model(sample_input)
        assert output.shape == (3, 5)


# Additional test for edge cases
class TestSoftmaxRegressionEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_input_size(self):
        """Test with invalid input sizes"""
        # Should work with any positive integer
        model_builder = SoftmaxRegression(input_size=1, num_classes=2)
        model = model_builder.create_model()
        assert model.input_shape == (None, 1)
        
    def test_single_class(self):
        """Test with single class (edge case)"""
        model_builder = SoftmaxRegression(input_size=10, num_classes=1)
        model = model_builder.create_model()
        
        sample_input = tf.random.normal((5, 10))
        output = model(sample_input)
        assert output.shape == (5, 1)
        
    def test_large_dimensions(self):
        """Test with larger dimensions"""
        model_builder = SoftmaxRegression(input_size=5000, num_classes=100)
        model = model_builder.create_model()
        
        # Check parameter count
        total_params = model.count_params()
        expected_params = 5000 * 100 + 100  # weights + biases
        assert total_params == expected_params


if __name__ == "__main__":
    pytest.main([__file__])