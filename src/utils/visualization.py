"""
Visualization utilities for training results and model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import os


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    show_plot: bool = True
) -> None:
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training & validation loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_sample_predictions(
    X_samples: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    num_samples: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    show_plot: bool = True
) -> None:
    """
    Plot sample images with true and predicted labels
    
    Args:
        X_samples: Sample images (flattened or 2D)
        Y_true: True labels
        Y_pred: Predicted labels
        num_samples: Number of samples to plot
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
    """
    # Reshape images if they are flattened
    if len(X_samples.shape) == 2:
        X_samples = X_samples.reshape(-1, 28, 28)
    
    # Limit number of samples
    num_samples = min(num_samples, len(X_samples))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        # Plot image
        axes[row][col].imshow(X_samples[i], cmap='gray')
        
        # Set title with true and predicted labels
        color = 'green' if Y_true[i] == Y_pred[i] else 'red'
        title = f'True: {Y_true[i]}, Pred: {Y_pred[i]}'
        axes[row][col].set_title(title, color=color, fontweight='bold')
        axes[row][col].axis('off')
    
    # Hide extra subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show_plot: bool = True
) -> None:
    """
    Plot confusion matrix
    
    Args:
        Y_true: True labels
        Y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(Y_true, Y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names or range(len(cm)),
        yticklabels=class_names or range(len(cm))
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> None:
    """
    Plot class distribution
    
    Args:
        labels: Labels array
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(unique_labels, counts, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    
    if class_names:
        plt.xticks(unique_labels, class_names)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_model_weights(
    model: tf.keras.Model,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    show_plot: bool = True
) -> None:
    """
    Plot model weights visualization
    
    Args:
        model: Keras model
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
    """
    weights = model.get_weights()
    
    if len(weights) < 2:
        print("Model should have at least weight matrix and bias vector")
        return
    
    W, b = weights[0], weights[1]  # Weight matrix and bias
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot weight matrix
    im1 = axes[0].imshow(W.T, cmap='RdBu', aspect='auto')
    axes[0].set_title('Weight Matrix (W)', fontweight='bold')
    axes[0].set_xlabel('Input Features')
    axes[0].set_ylabel('Output Classes')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot bias vector
    axes[1].bar(range(len(b)), b, alpha=0.7)
    axes[1].set_title('Bias Vector (b)', fontweight='bold')
    axes[1].set_xlabel('Output Classes')
    axes[1].set_ylabel('Bias Value')
    axes[1].grid(True, alpha=0.3)
    
    # Plot weight distribution
    axes[2].hist(W.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_title('Weight Distribution', fontweight='bold')
    axes[2].set_xlabel('Weight Value')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model weights plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> None:
    """
    Plot learning curves
    
    Args:
        train_sizes: Training set sizes
        train_scores: Training scores
        val_scores: Validation scores
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot curves
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_visualization_report(
    history: Dict[str, List[float]],
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    save_dir: str = 'results/plots'
) -> None:
    """
    Create a comprehensive visualization report
    
    Args:
        history: Training history
        model: Trained model
        X_test: Test data
        Y_test: True labels
        Y_pred: Predicted labels
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("📊 Creating comprehensive visualization report...")
    
    # 1. Training history
    plot_training_history(
        history,
        save_path=os.path.join(save_dir, 'training_history.png'),
        show_plot=False
    )
    
    # 2. Sample predictions
    plot_sample_predictions(
        X_test[:16], Y_test[:16], Y_pred[:16],
        save_path=os.path.join(save_dir, 'sample_predictions.png'),
        show_plot=False
    )
    
    # 3. Confusion matrix
    plot_confusion_matrix(
        Y_test, Y_pred,
        class_names=[str(i) for i in range(10)],
        save_path=os.path.join(save_dir, 'confusion_matrix.png'),
        show_plot=False
    )
    
    # 4. Class distribution
    plot_class_distribution(
        Y_test,
        class_names=[str(i) for i in range(10)],
        save_path=os.path.join(save_dir, 'class_distribution.png'),
        show_plot=False
    )
    
    # 5. Model weights
    plot_model_weights(
        model,
        save_path=os.path.join(save_dir, 'model_weights.png'),
        show_plot=False
    )
    
    print(f"✅ Visualization report created in {save_dir}")