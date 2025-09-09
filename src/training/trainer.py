"""
Model training utilities
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Tuple, Optional, List
import time


class ModelTrainer:
    """
    High-level trainer for Softmax Regression model using Keras fit() method
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialize trainer with a compiled model
        
        Args:
            model: Compiled Keras model
        """
        self.model = model
        self.history = None
        
    def train(
        self,
        X_train: tf.Tensor,
        Y_train: tf.Tensor,
        X_val: tf.Tensor,
        Y_val: tf.Tensor,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: int = 1,
        callbacks: Optional[List] = None
    ) -> keras.callbacks.History:
        """
        Train the model using Keras fit() method
        
        Args:
            X_train: Training data
            Y_train: Training labels
            X_val: Validation data
            Y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            callbacks: List of callbacks
            
        Returns:
            keras.callbacks.History: Training history
        """
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return self.history
    
    def evaluate(self, X_test: tf.Tensor, Y_test: tf.Tensor, verbose: int = 1) -> Tuple[float, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test data
            Y_test: Test labels
            verbose: Verbosity level
            
        Returns:
            Tuple of (loss, accuracy)
        """
        results = self.model.evaluate(X_test, Y_test, verbose=verbose)
        loss, accuracy = results[0], results[1]
        
        return loss, accuracy
    
    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Make predictions on input data
        
        Args:
            X: Input data
            
        Returns:
            tf.Tensor: Predictions
        """
        return self.model.predict(X)
    
    def predict_classes(self, X: tf.Tensor) -> np.ndarray:
        """
        Predict class labels for input data
        
        Args:
            X: Input data
            
        Returns:
            np.ndarray: Predicted class labels
        """
        predictions = self.predict(X)
        return tf.argmax(predictions, axis=1).numpy()
    
    def predict_samples_with_confidence(
        self, 
        X_samples: tf.Tensor, 
        Y_true: np.ndarray, 
        num_samples: int = 5
    ) -> Dict:
        """
        Predict samples and show confidence scores
        
        Args:
            X_samples: Input samples (flattened)
            Y_true: True labels (not one-hot)
            num_samples: Number of samples to display
            
        Returns:
            Dict: Prediction results
        """
        predictions = self.predict(X_samples)
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        confidence_scores = tf.reduce_max(predictions, axis=1).numpy()
        
        results = {
            'predictions': predicted_classes,
            'true_labels': Y_true,
            'confidence_scores': confidence_scores,
            'details': []
        }
        
        print("Prediction Results:")
        for i in range(min(num_samples, len(Y_true))):
            detail = {
                'sample_id': i,
                'true_label': Y_true[i],
                'predicted_label': predicted_classes[i],
                'confidence': confidence_scores[i],
                'correct': Y_true[i] == predicted_classes[i]
            }
            results['details'].append(detail)
            
            print(f"Sample {i}: True={Y_true[i]}, Predicted={predicted_classes[i]}, "
                  f"Confidence={confidence_scores[i]:.3f}, "
                  f"Correct={'✓' if detail['correct'] else '✗'}")
        
        return results
    
    def get_training_history(self) -> Dict:
        """
        Get training history metrics
        
        Returns:
            Dict: Training history
        """
        if self.history is None:
            return {}
        
        return self.history.history
    
    def get_best_epoch(self, metric: str = 'val_accuracy') -> int:
        """
        Get epoch with best performance on specified metric
        
        Args:
            metric: Metric to optimize ('val_accuracy', 'val_loss', etc.)
            
        Returns:
            int: Best epoch (1-indexed)
        """
        if self.history is None:
            return 0
        
        if metric not in self.history.history:
            return 0
        
        values = self.history.history[metric]
        
        if 'loss' in metric:
            best_epoch = np.argmin(values)
        else:  # accuracy or other metrics (higher is better)
            best_epoch = np.argmax(values)
        
        return best_epoch + 1  # Convert to 1-indexed
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model
        
        Args:
            filepath: Path to load the model from
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class EarlyStoppingCallback(keras.callbacks.Callback):
    """
    Custom early stopping callback
    """
    
    def __init__(self, patience: int = 5, monitor: str = 'val_loss', min_delta: float = 0.001):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        if self.best_value is None:
            self.best_value = current_value
        elif self._is_improvement(current_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                
    def _is_improvement(self, current_value):
        if 'loss' in self.monitor:
            return current_value < self.best_value - self.min_delta
        else:
            return current_value > self.best_value + self.min_delta
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nEarly stopping at epoch {self.stopped_epoch + 1}")


def create_callbacks(
    patience: int = 5,
    save_best_only: bool = True,
    filepath: str = 'best_model.h5'
) -> List[keras.callbacks.Callback]:
    """
    Create common callbacks for training
    
    Args:
        patience: Patience for early stopping
        save_best_only: Whether to save only the best model
        filepath: Path to save the best model
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Model checkpoint
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_accuracy',
            save_best_only=save_best_only,
            verbose=1
        )
    )
    
    # Reduce learning rate on plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    )
    
    return callbacks