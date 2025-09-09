"""
Evaluation utilities for model performance assessment
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional
import json


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dict: Classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate per-class metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dict: Per-class metrics
    """
    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Calculate per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names[:len(precision)]):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i])
        }
    
    return per_class_metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dict: bool = True
) -> Dict:
    """
    Get detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dict: Whether to return as dictionary
        
    Returns:
        Classification report
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Calculate metrics from confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict: Confusion matrix derived metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics for each class
    num_classes = cm.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Per-class metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'class_{i}'] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1)
        }
    
    return metrics


def calculate_top_k_accuracy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        k: Top k predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = 0
    
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def evaluate_model_comprehensive(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
        verbose: Whether to print results
        
    Returns:
        Dict: Comprehensive evaluation results
    """
    # Convert one-hot to class indices if needed
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_classes = np.argmax(y_test, axis=1)
    else:
        y_test_classes = y_test
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    # Basic metrics
    basic_metrics = calculate_classification_metrics(y_test_classes, y_pred_classes)
    
    # Per-class metrics
    per_class_metrics = calculate_per_class_metrics(
        y_test_classes, y_pred_classes, class_names
    )
    
    # Classification report
    classification_rep = get_classification_report(
        y_test_classes, y_pred_classes, class_names
    )
    
    # Confusion matrix metrics
    cm_metrics = calculate_confusion_matrix_metrics(y_test_classes, y_pred_classes)
    
    # Top-k accuracy
    top_5_accuracy = calculate_top_k_accuracy(y_test_classes, y_pred_proba, k=5)
    
    # Model evaluation using Keras
    keras_metrics = model.evaluate(X_test, y_test, verbose=0)
    keras_loss = keras_metrics[0]
    keras_accuracy = keras_metrics[1] if len(keras_metrics) > 1 else None
    
    results = {
        'basic_metrics': basic_metrics,
        'per_class_metrics': per_class_metrics,
        'classification_report': classification_rep,
        'confusion_matrix_metrics': cm_metrics,
        'top_5_accuracy': float(top_5_accuracy),
        'keras_metrics': {
            'loss': float(keras_loss),
            'accuracy': float(keras_accuracy) if keras_accuracy else None
        },
        'predictions': {
            'y_pred_classes': y_pred_classes.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
    }
    
    if verbose:
        print_evaluation_results(results)
    
    return results


def print_evaluation_results(results: Dict) -> None:
    """
    Print evaluation results in a formatted way
    
    Args:
        results: Evaluation results dictionary
    """
    print("=" * 60)
    print("🔍 COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # Basic metrics
    print("\n📊 BASIC METRICS:")
    basic = results['basic_metrics']
    print(f"  • Accuracy:  {basic['accuracy']:.4f}")
    print(f"  • Precision: {basic['precision']:.4f}")
    print(f"  • Recall:    {basic['recall']:.4f}")
    print(f"  • F1-Score:  {basic['f1_score']:.4f}")
    print(f"  • Top-5 Acc: {results['top_5_accuracy']:.4f}")
    
    # Keras metrics
    print(f"\n🔥 KERAS METRICS:")
    keras = results['keras_metrics']
    print(f"  • Loss:      {keras['loss']:.4f}")
    if keras['accuracy']:
        print(f"  • Accuracy:  {keras['accuracy']:.4f}")
    
    # Per-class summary
    print(f"\n📈 PER-CLASS SUMMARY:")
    per_class = results['per_class_metrics']
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 45)
    
    for class_name, metrics in per_class.items():
        print(f"{class_name:<10} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print("=" * 60)


def save_evaluation_results(
    results: Dict,
    filepath: str,
    include_predictions: bool = False
) -> None:
    """
    Save evaluation results to JSON file
    
    Args:
        results: Evaluation results
        filepath: Path to save results
        include_predictions: Whether to include predictions in saved file
    """
    # Create a copy of results
    save_results = results.copy()
    
    # Remove predictions if not needed (can be large)
    if not include_predictions and 'predictions' in save_results:
        del save_results['predictions']
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"📄 Evaluation results saved to {filepath}")


def compare_models(
    results_list: List[Dict],
    model_names: List[str],
    metric: str = 'accuracy'
) -> Dict:
    """
    Compare multiple models based on a specific metric
    
    Args:
        results_list: List of evaluation results
        model_names: List of model names
        metric: Metric to compare
        
    Returns:
        Dict: Comparison results
    """
    comparison = {}
    
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        if metric in results['basic_metrics']:
            comparison[name] = results['basic_metrics'][metric]
        elif metric in results['keras_metrics']:
            comparison[name] = results['keras_metrics'][metric]
        else:
            comparison[name] = None
    
    # Sort by metric (higher is better for most metrics except loss)
    reverse_sort = metric != 'loss'
    sorted_comparison = dict(sorted(
        comparison.items(), 
        key=lambda x: x[1] if x[1] is not None else -float('inf'),
        reverse=reverse_sort
    ))
    
    return {
        'metric': metric,
        'rankings': sorted_comparison,
        'best_model': list(sorted_comparison.keys())[0] if sorted_comparison else None
    }


def calculate_model_confidence_stats(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray
) -> Dict:
    """
    Calculate statistics about model confidence
    
    Args:
        y_pred_proba: Prediction probabilities
        y_true: True labels
        
    Returns:
        Dict: Confidence statistics
    """
    # Get predicted classes and max probabilities
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    max_probs = np.max(y_pred_proba, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (y_pred_classes == y_true)
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    
    stats = {
        'overall': {
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs))
        },
        'correct_predictions': {
            'count': int(np.sum(correct_mask)),
            'mean_confidence': float(np.mean(correct_probs)) if len(correct_probs) > 0 else 0,
            'std_confidence': float(np.std(correct_probs)) if len(correct_probs) > 0 else 0
        },
        'incorrect_predictions': {
            'count': int(np.sum(~correct_mask)),
            'mean_confidence': float(np.mean(incorrect_probs)) if len(incorrect_probs) > 0 else 0,
            'std_confidence': float(np.std(incorrect_probs)) if len(incorrect_probs) > 0 else 0
        }
    }
    
    return stats