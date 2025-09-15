"""
Gas Monitoring ML Pipeline Package

This package contains modules for preprocessing and training machine learning models
for activity level prediction based on gas monitoring sensor data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .preprocessing import preprocessing_for_linear, preprocessing_for_tree, preprocessing_for_catboost
from .pipeline import load_data, train_and_evaluate_all, get_models
from .evaluation import evaluate_model, print_evaluation_summary, plot_confusion_matrix, plot_feature_importance

__all__ = [
    'preprocessing_for_linear',
    'preprocessing_for_tree', 
    'preprocessing_for_catboost',
    'load_data',
    'train_and_evaluate_all',
    'get_models',
    'evaluate_model',
    'print_evaluation_summary',
    'plot_confusion_matrix',
    'plot_feature_importance'
]
