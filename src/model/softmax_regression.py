"""
Softmax Regression Model Implementation using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from typing import Optional


class SoftmaxRegression:
    """
    Softmax Regression model for multi-class classification
    
    Args:
        input_size (int): Size of input features (default: 784 for MNIST)
        num_classes (int): Number of output classes (default: 10 for MNIST)
    """
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self) -> keras.Model:
        """
        Create Softmax Regression model using Keras Sequential API
        
        Returns:
            tf.keras.Model: Compiled model
        """
        model = keras.Sequential([
            layers.InputLayer(input_shape=(self.input_size,)),
            layers.Dense(self.num_classes),
            layers.Activation('softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(
        self, 
        model: keras.Model,
        learning_rate: float = 0.01,
        optimizer: str = 'sgd',
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ) -> keras.Model:
        """
        Compile model with specified optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name ('sgd', 'adam', 'rmsprop')
            loss: Loss function name
            metrics: List of metrics to track
            
        Returns:
            tf.keras.Model: Compiled model
        """
        if metrics is None:
            metrics = ['accuracy']
            
        # Choose optimizer
        if optimizer.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizers.SGD(learning_rate=learning_rate)
            
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def create_and_compile(
        self,
        learning_rate: float = 0.01,
        optimizer: str = 'sgd',
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ) -> keras.Model:
        """
        Create and compile model in one step
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name
            loss: Loss function name
            metrics: List of metrics to track
            
        Returns:
            tf.keras.Model: Created and compiled model
        """
        model = self.create_model()
        compiled_model = self.compile_model(
            model, learning_rate, optimizer, loss, metrics
        )
        return compiled_model
    
    def get_model_summary(self) -> None:
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        self.model.summary()
    
    def get_weights(self) -> dict:
        """
        Get model weights and biases
        
        Returns:
            dict: Dictionary containing weights and biases
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
            
        weights = self.model.get_weights()
        return {
            'weights': weights[0],  # Weight matrix (784, 10)
            'biases': weights[1]    # Bias vector (10,)
        }
    
    def set_weights(self, weights: dict) -> None:
        """
        Set model weights and biases
        
        Args:
            weights: Dictionary containing 'weights' and 'biases' keys
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
            
        self.model.set_weights([weights['weights'], weights['biases']])
    
    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Make predictions on input data
        
        Args:
            X: Input tensor of shape (batch_size, input_size)
            
        Returns:
            tf.Tensor: Predictions of shape (batch_size, num_classes)
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
            
        return self.model(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
            
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> keras.Model:
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            tf.keras.Model: Loaded model
        """
        self.model = keras.models.load_model(filepath)
        return self.model