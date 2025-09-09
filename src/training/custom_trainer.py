"""
Custom training loop implementation using GradientTape
"""

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
import time
from typing import Dict, List, Tuple


class CustomTrainer:
    """
    Custom training loop for Softmax Regression using GradientTape
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize custom trainer
        
        Args:
            model: Keras model (not necessarily compiled)
        """
        self.model = model
        self.optimizer = None
        self.loss_fn = None
        self.train_loss = None
        self.train_accuracy = None
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
    def setup_training(
        self,
        optimizer: str = 'sgd',
        learning_rate: float = 0.01,
        loss: str = 'categorical_crossentropy'
    ):
        """
        Setup optimizer and loss function for training
        
        Args:
            optimizer: Optimizer name ('sgd', 'adam', 'rmsprop')
            learning_rate: Learning rate
            loss: Loss function name
        """
        # Setup optimizer
        if optimizer.lower() == 'sgd':
            self.optimizer = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            self.optimizer = optimizers.SGD(learning_rate=learning_rate)
        
        # Setup loss function
        if loss == 'categorical_crossentropy':
            self.loss_fn = losses.CategoricalCrossentropy()
        elif loss == 'sparse_categorical_crossentropy':
            self.loss_fn = losses.SparseCategoricalCrossentropy()
        else:
            self.loss_fn = losses.CategoricalCrossentropy()
        
        # Setup metrics
        self.train_loss = metrics.Mean(name='train_loss')
        self.train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')
    
    @tf.function
    def train_step(self, batch_x: tf.Tensor, batch_y: tf.Tensor) -> tf.Tensor:
        """
        Perform single training step
        
        Args:
            batch_x: Input batch
            batch_y: Target batch
            
        Returns:
            tf.Tensor: Loss value
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(batch_x, training=True)
            # Compute loss
            loss = self.loss_fn(batch_y, predictions)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(batch_y, predictions)
        
        return loss
    
    def validation_step(self, X_val: tf.Tensor, Y_val: tf.Tensor) -> Tuple[float, float]:
        """
        Perform validation
        
        Args:
            X_val: Validation inputs
            Y_val: Validation targets
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        val_predictions = self.model(X_val, training=False)
        val_loss = self.loss_fn(Y_val, val_predictions)
        val_accuracy = metrics.categorical_accuracy(Y_val, val_predictions)
        val_accuracy = tf.reduce_mean(val_accuracy)
        
        return float(val_loss), float(val_accuracy)
    
    def train(
        self,
        X_train: tf.Tensor,
        Y_train: tf.Tensor,
        X_val: tf.Tensor,
        Y_val: tf.Tensor,
        epochs: int = 30,
        batch_size: int = 128,
        optimizer: str = 'sgd',
        learning_rate: float = 0.01,
        loss: str = 'categorical_crossentropy',
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train model using custom training loop
        
        Args:
            X_train: Training data
            Y_train: Training labels
            X_val: Validation data
            Y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function name
            verbose: Verbosity level
            
        Returns:
            Dict: Training history
        """
        # Setup training components
        self.setup_training(optimizer, learning_rate, loss)
        
        # Reset history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        if verbose > 0:
            print(f"Training with custom loop for {epochs} epochs...")
            
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")
            
            # Reset metrics at start of each epoch
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            
            # Create dataset and batch it
            dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            dataset = dataset.shuffle(1000).batch(batch_size)
            
            # Training steps
            for batch_x, batch_y in dataset:
                self.train_step(batch_x, batch_y)
            
            # Validation
            val_loss, val_accuracy = self.validation_step(X_val, Y_val)
            
            # Record metrics
            epoch_loss = float(self.train_loss.result())
            epoch_accuracy = float(self.train_accuracy.result())
            
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            if verbose > 0:
                print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if verbose > 0:
            print(f"\nCustom training completed in {training_time:.2f} seconds")
        
        return self.history
    
    def evaluate(self, X_test: tf.Tensor, Y_test: tf.Tensor) -> Tuple[float, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test data
            Y_test: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.loss_fn is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.validation_step(X_test, Y_test)
    
    def get_gradients(self, X_batch: tf.Tensor, Y_batch: tf.Tensor) -> List[tf.Tensor]:
        """
        Get gradients for a batch (useful for analysis)
        
        Args:
            X_batch: Input batch
            Y_batch: Target batch
            
        Returns:
            List of gradient tensors
        """
        with tf.GradientTape() as tape:
            predictions = self.model(X_batch, training=True)
            loss = self.loss_fn(Y_batch, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients
    
    def get_weights_and_gradients_info(self, X_batch: tf.Tensor, Y_batch: tf.Tensor) -> Dict:
        """
        Get information about weights and gradients
        
        Args:
            X_batch: Input batch
            Y_batch: Target batch
            
        Returns:
            Dict: Information about weights and gradients
        """
        weights = self.model.get_weights()
        gradients = self.get_gradients(X_batch, Y_batch)
        
        info = {
            'weight_shapes': [w.shape for w in weights],
            'weight_norms': [float(tf.norm(w)) for w in weights],
            'gradient_shapes': [g.shape for g in gradients if g is not None],
            'gradient_norms': [float(tf.norm(g)) for g in gradients if g is not None]
        }
        
        return info
    
    def save_history(self, filepath: str):
        """
        Save training history to file
        
        Args:
            filepath: Path to save history
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        history_json = {}
        for key, values in self.history.items():
            history_json[key] = [float(v) for v in values]
        
        with open(filepath, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved to {filepath}")


class GradientAnalyzer:
    """
    Utility class for analyzing gradients during training
    """
    
    def __init__(self, trainer: CustomTrainer):
        self.trainer = trainer
        
    def analyze_gradient_flow(self, X_batch: tf.Tensor, Y_batch: tf.Tensor) -> Dict:
        """
        Analyze gradient flow through the network
        
        Args:
            X_batch: Input batch
            Y_batch: Target batch
            
        Returns:
            Dict: Gradient analysis results
        """
        gradients = self.trainer.get_gradients(X_batch, Y_batch)
        
        analysis = {
            'layer_names': [var.name for var in self.trainer.model.trainable_variables],
            'gradient_norms': [],
            'gradient_stats': []
        }
        
        for i, grad in enumerate(gradients):
            if grad is not None:
                norm = float(tf.norm(grad))
                mean = float(tf.reduce_mean(grad))
                std = float(tf.math.reduce_std(grad))
                
                analysis['gradient_norms'].append(norm)
                analysis['gradient_stats'].append({
                    'mean': mean,
                    'std': std,
                    'min': float(tf.reduce_min(grad)),
                    'max': float(tf.reduce_max(grad))
                })
            else:
                analysis['gradient_norms'].append(0.0)
                analysis['gradient_stats'].append(None)
        
        return analysis
    
    def check_vanishing_gradients(self, X_batch: tf.Tensor, Y_batch: tf.Tensor, threshold: float = 1e-6) -> Dict:
        """
        Check for vanishing gradients
        
        Args:
            X_batch: Input batch
            Y_batch: Target batch
            threshold: Threshold for vanishing gradient detection
            
        Returns:
            Dict: Vanishing gradient analysis
        """
        analysis = self.analyze_gradient_flow(X_batch, Y_batch)
        
        vanishing_layers = []
        for i, norm in enumerate(analysis['gradient_norms']):
            if norm < threshold:
                vanishing_layers.append({
                    'layer_index': i,
                    'layer_name': analysis['layer_names'][i],
                    'gradient_norm': norm
                })
        
        return {
            'has_vanishing_gradients': len(vanishing_layers) > 0,
            'vanishing_layers': vanishing_layers,
            'threshold': threshold
        }