#!/usr/bin/env python3
"""
Basic usage example for Softmax Regression with TensorFlow
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.softmax_regression import SoftmaxRegression
from src.data.data_preprocessing import load_and_preprocess_mnist
from src.training.trainer import ModelTrainer, create_callbacks
from src.utils.visualization import plot_training_history, plot_sample_predictions
from src.utils.evaluation import calculate_classification_metrics


def main():
    """
    Basic usage example
    """
    print("🚀 Softmax Regression with TensorFlow - Basic Usage Example")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("📊 Loading and preprocessing MNIST dataset...")
    (X_train, Y_train), (X_val, Y_val) = load_and_preprocess_mnist()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Number of classes: {Y_train.shape[1]}")
    
    # 2. Create and compile model
    print("\n🧠 Creating Softmax Regression model...")
    model_builder = SoftmaxRegression(input_size=784, num_classes=10)
    model = model_builder.create_and_compile(
        learning_rate=0.01,
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model created successfully!")
    model_builder.get_model_summary()
    
    # 3. Setup trainer with callbacks
    print("\n🏃 Setting up trainer...")
    trainer = ModelTrainer(model)
    
    callbacks = create_callbacks(
        patience=5,
        save_best_only=True,
        filepath='results/models/best_model.h5'
    )
    
    # 4. Train model
    print("\n🎯 Training model...")
    history = trainer.train(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        epochs=20,
        batch_size=128,
        verbose=1,
        callbacks=callbacks
    )
    
    # 5. Evaluate model
    print("\n📈 Evaluating model...")
    val_loss, val_accuracy = trainer.evaluate(X_val, Y_val)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    # 6. Make predictions and show samples
    print("\n🔮 Making predictions on sample data...")
    # Convert Y_val from one-hot to class indices for display
    import tensorflow as tf
    Y_val_classes = tf.argmax(Y_val, axis=1).numpy()
    
    prediction_results = trainer.predict_samples_with_confidence(
        X_val[:10], Y_val_classes[:10], num_samples=10
    )
    
    # 7. Calculate detailed metrics
    print("\n📊 Calculating detailed classification metrics...")
    Y_pred_classes = trainer.predict_classes(X_val)
    metrics = calculate_classification_metrics(Y_val_classes, Y_pred_classes)
    
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # 8. Visualizations
    print("\n📊 Creating visualizations...")
    
    # Plot training history
    plot_training_history(
        history.history, 
        save_path='results/plots/training_history.png'
    )
    
    # Plot sample predictions
    plot_sample_predictions(
        X_val[:16], Y_val_classes[:16], Y_pred_classes[:16],
        save_path='results/plots/sample_predictions.png'
    )
    
    # 9. Save final model
    print("\n💾 Saving final model...")
    trainer.save_model('results/models/final_model.h5')
    
    # 10. Display best epoch information
    best_epoch = trainer.get_best_epoch('val_accuracy')
    print(f"\n🏆 Best epoch: {best_epoch}")
    
    print("\n✅ Basic usage example completed successfully!")
    print("Check the 'results/' directory for saved models and plots.")


if __name__ == "__main__":
    # Create results directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    main()