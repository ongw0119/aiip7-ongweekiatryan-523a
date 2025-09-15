"""
Model evaluation utilities for the gas monitoring ML pipeline.
"""

import os
# Set matplotlib backend before any other imports to avoid tkinter threading issues
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib
# Set non-interactive backend to avoid tkinter threading issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_pred_proba=None, class_names=None):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        class_names: List of class names for display
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    results = {}
    
    # Basic metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    results['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    results['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    results['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC AUC if probabilities available
    if y_pred_proba is not None:
        try:
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        except ValueError:
            results['roc_auc'] = None
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return results


def print_evaluation_summary(results, class_names=None):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Dictionary from evaluate_model()
        class_names: List of class names for display
    """
    print("=" * 50)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 50)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision (Macro): {results['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
    
    if results['roc_auc'] is not None:
        print(f"ROC AUC (Macro): {results['roc_auc']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 30)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(results['precision_per_class']))]
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {results['precision_per_class'][i]:.4f}")
        print(f"  Recall: {results['recall_per_class'][i]:.4f}")
        print(f"  F1-Score: {results['f1_per_class'][i]:.4f}")


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix with proper formatting.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


def plot_feature_importance(importances, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models.
    
    Args:
        importances: Feature importance array
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"feature_importance_top_{top_n}.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
