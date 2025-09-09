"""
Unit tests for training modules
"""
import pytest
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model.softmax_regression import SoftmaxRegression
from src.training.trainer import ModelTrainer
from src.training.custom_trainer import CustomTrainer

class TestTraining:
    def setup_method(self):
        self.model = SoftmaxRegression(784, 10).create_and_compile()
        self.X_train = tf.random.normal((100, 784))
        self.Y_train = tf.one_hot(tf.random.uniform((100,), maxval=10, dtype=tf.int32), 10)
        self.X_val = tf.random.normal((20, 784))
        self.Y_val = tf.one_hot(tf.random.uniform((20,), maxval=10, dtype=tf.int32), 10)
    
    def test_model_trainer(self):
        trainer = ModelTrainer(self.model)
        history = trainer.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=2, batch_size=32, verbose=0
        )
        
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 2
    
    def test_custom_trainer(self):
        model = SoftmaxRegression(784, 10).create_model()
        trainer = CustomTrainer(model)
        history = trainer.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=2, batch_size=32, verbose=0
        )
        
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 2

if __name__ == "__main__":
    pytest.main([__file__])