#!/usr/bin/env python3
"""
Custom training loop example for Softmax Regression
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
from src.model.softmax_regression import SoftmaxRegression
from src.data.data_preprocessing import load_and_preprocess_mnist
from src.training.custom_trainer import CustomTrainer, GradientAnalyzer
from src.utils.visualization import plot_training_history
from src.utils.evaluation import calculate_classification_metrics


def main():
    """
    Custom training example
    """
    print("🔥 Softmax Regression - Custom Training Loop Example")
    print("=" * 60)
    
    # 1. Load and preprocess data (using subset for faster demo)
    print("📊 Loading MNIST subset...")
    (X_train, Y_train), (X_val, Y_val) = load_and_preprocess_mnist(subset_size=5000)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # 2. Create model (don't compile - custom trainer will handle optimization)
    print("\n🧠 Creating Softmax Regression model...")
    model_builder = SoftmaxRegression(input_size=784, num_classes=10)
    model = model_builder.create_model()
    
    print("Model created (not compiled - using custom training loop)")
    model_builder.get_model_summary()
    
    # 3. Setup custom trainer
    print("\n🏃 Setting up custom trainer...")
    trainer = CustomTrainer(model)
    
    # 4. Train with custom loop
    print("\n🎯 Training with custom loop...")
    history = trainer.train(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        epochs=15,
        batch_size=64,
        optimizer='sgd',
        learning_rate=0.01,
        loss='categorical_crossentropy',
        verbose=1
    )
    
    # 5. Evaluate model
    print("\n📈 Evaluating model...")
    val_loss, val_accuracy = trainer.evaluate(X_val, Y_val)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    # 6. Gradient analysis
    print("\n🔍 Analyzing gradients...")
    analyzer = GradientAnalyzer(trainer)
    
    # Analyze gradient flow with a small batch
    batch_x = X_train[:32]
    batch_y = Y_train[:32]
    
    gradient_analysis = analyzer.analyze_gradient_flow(batch_x, batch_y)
    print("\nGradient Analysis:")
    for i, (name, norm) in enumerate(zip(gradient_analysis['layer_names'], 
                                        gradient_analysis['gradient_norms'])):
        print(f"  Layer {i} ({name.split('/')[-1]}): Gradient norm = {norm:.6f}")
    
    # Check for vanishing gradients
    vanishing_check = analyzer.check_vanishing_gradients(batch_x, batch_y)
    if vanishing_check['has_vanishing_gradients']:
        print(f"\n⚠️  Warning: Vanishing gradients detected!")
        for layer_info in vanishing_check['vanishing_layers']:
            print(f"    {layer_info['layer_name']}: {layer_info['gradient_norm']:.2e}")
    else:
        print("\n✅ No vanishing gradients detected")
    
    # 7. Compare with different optimizers
    print("\n🔄 Comparing different optimizers...")
    
    optimizers_to_test = ['sgd', 'adam', 'rmsprop']
    optimizer_results = {}
    
    for opt_name in optimizers_to_test:
        print(f"\nTesting {opt_name.upper()} optimizer...")
        
        # Create fresh model
        test_model = SoftmaxRegression(784, 10).create_model()
        test_trainer = CustomTrainer(test_model)
        
        # Train for fewer epochs for comparison
        test_history = test_trainer.train(
            X_train[:1000], Y_train[:1000],  # Small subset for quick comparison
            X_val[:200], Y_val[:200],
            epochs=5,
            batch_size=32,
            optimizer=opt_name,
            learning_rate=0.01 if opt_name == 'sgd' else 0.001,
            verbose=0
        )
        
        final_acc = test_history['val_accuracy'][-1]
        optimizer_results[opt_name] = final_acc
        print(f"  Final validation accuracy: {final_acc:.4f}")
    
    # 8. Display optimizer comparison
    print(f"\n📊 Optimizer Comparison Results:")
    print("-" * 40)
    sorted_optimizers = sorted(optimizer_results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (opt_name, accuracy) in enumerate(sorted_optimizers, 1):
        print(f"{i}. {opt_name.upper():<8}: {accuracy:.4f}")
    
    print(f"\n🏆 Best optimizer: {sorted_optimizers[0][0].upper()}")
    
    # 9. Advanced gradient analysis
    print(f"\n🔬 Advanced Analysis...")
    
    # Get detailed weight and gradient information
    weight_grad_info = trainer.get_weights_and_gradients_info(batch_x, batch_y)
    
    print("\nWeight Statistics:")
    for i, (shape, norm) in enumerate(zip(weight_grad_info['weight_shapes'], 
                                         weight_grad_info['weight_norms'])):
        print(f"  Weight {i}: Shape {shape}, L2 norm: {norm:.4f}")
    
    print("\nGradient Statistics:")
    for i, (shape, norm) in enumerate(zip(weight_grad_info['gradient_shapes'], 
                                         weight_grad_info['gradient_norms'])):
        print(f"  Gradient {i}: Shape {shape}, L2 norm: {norm:.6f}")
    
    # 10. Visualizations
    print("\n📊 Creating visualizations...")
    
    # Plot training history
    plot_training_history(
        history, 
        save_path='results/plots/custom_training_history.png',
        show_plot=False
    )
    
    # 11. Save training history
    print("\n💾 Saving training history...")
    trainer.save_history('results/logs/custom_training_history.json')
    
    # 12. Final predictions and metrics
    print("\n🔮 Final evaluation...")
    
    # Get predictions
    predictions = model(X_val)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()
    true_classes = tf.argmax(Y_val, axis=1).numpy()
    
    # Calculate detailed metrics
    metrics = calculate_classification_metrics(true_classes, predicted_classes)
    
    print("\nFinal Classification Metrics:")
    print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  • Precision: {metrics['precision']:.4f}")
    print(f"  • Recall:    {metrics['recall']:.4f}")
    print(f"  • F1-Score:  {metrics['f1_score']:.4f}")
    
    # Show some example predictions
    print("\n🎯 Sample Predictions:")
    for i in range(5):
        confidence = tf.reduce_max(predictions[i]).numpy()
        print(f"  Sample {i}: True={true_classes[i]}, Pred={predicted_classes[i]}, "
              f"Confidence={confidence:.3f}")
    
    print("\n✅ Custom training example completed successfully!")
    print("Check the 'results/' directory for saved plots and logs.")


def demonstrate_gradient_tape():
    """
    Demonstrate raw TensorFlow GradientTape usage
    """
    print("\n" + "="*50)
    print("🔬 GRADIENTTAPE DEMONSTRATION")
    print("="*50)
    
    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
    ])
    
    # Sample data
    x = tf.random.normal((32, 784))
    y = tf.one_hot(tf.random.uniform((32,), maxval=10, dtype=tf.int32), 10)
    
    # Loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    print("Demonstrating a single gradient computation:")
    
    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
        print(f"  Forward pass loss: {loss:.4f}")
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    print("  Gradient information:")
    for i, grad in enumerate(gradients):
        print(f"    Variable {i}: Shape {grad.shape}, L2 norm: {tf.norm(grad):.6f}")
    
    print("✅ GradientTape demonstration complete")


if __name__ == "__main__":
    # Create results directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    main()
    demonstrate_gradient_tape()