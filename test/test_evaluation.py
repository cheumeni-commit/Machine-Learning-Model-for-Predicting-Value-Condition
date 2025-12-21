"""
Unit tests for evaluation module.
"""
import pytest
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.training.evaluation import get_metrics, evaluate_model


class TestGetMetrics:
    """Test cases for get_metrics function."""
    
    def test_get_metrics_success(self):
        """Test calculating metrics successfully."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        performance = get_metrics(y_true, y_pred, classes)
        
        assert isinstance(performance, dict)
        assert 'overall' in performance
        assert 'class' in performance
        
        # Check overall metrics
        assert 'precision' in performance['overall']
        assert 'recall' in performance['overall']
        assert 'f1' in performance['overall']
        assert 'num_samples' in performance['overall']
        
        # Check class metrics
        assert len(performance['class']) == len(classes)
        for cl in classes:
            assert cl in performance['class']
            assert 'precision' in performance['class'][cl]
            assert 'recall' in performance['class'][cl]
            assert 'f1' in performance['class'][cl]
            assert 'num_samples' in performance['class'][cl]
    
    def test_get_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])  # Perfect match
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        performance = get_metrics(y_true, y_pred, classes)
        
        # Perfect predictions should have high metrics
        assert performance['overall']['precision'] > 0.9
        assert performance['overall']['recall'] > 0.9
        assert performance['overall']['f1'] > 0.9
        assert performance['overall']['num_samples'] == len(y_true)
    
    def test_get_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])  # All wrong
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        performance = get_metrics(y_true, y_pred, classes)
        
        # Should still return valid metrics structure
        assert isinstance(performance, dict)
        assert 'overall' in performance
        assert 'class' in performance
        assert performance['overall']['num_samples'] == len(y_true)
    
    def test_get_metrics_single_class(self):
        """Test metrics with single class predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        performance = get_metrics(y_true, y_pred, classes)
        
        assert isinstance(performance, dict)
        assert performance['overall']['num_samples'] == len(y_true)


class TestEvaluateModel:
    """Test cases for evaluate_model function."""
    
    def test_evaluate_model_success(self, sample_model, sample_train_test_data):
        """Test evaluating a model successfully."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        # Fit the model first
        sample_model.fit(X_train, y_train)
        
        performance = evaluate_model(sample_model, X=X_test, y=y_test, classes=classes)
        
        assert isinstance(performance, dict)
        assert 'overall' in performance
        assert 'class' in performance
    
    def test_evaluate_model_with_predictions(self, sample_model, sample_train_test_data):
        """Test that evaluate_model makes predictions correctly."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        sample_model.fit(X_train, y_train)
        
        # Get predictions manually
        y_pred_manual = sample_model.predict(X_test)
        
        # Evaluate model
        performance = evaluate_model(sample_model, X=X_test, y=y_test, classes=classes)
        
        # Performance should be based on model predictions
        assert isinstance(performance, dict)
        assert performance['overall']['num_samples'] == len(y_test)
    
    def test_evaluate_model_empty_data(self, sample_model):
        """Test evaluating model with empty data."""
        X_empty = np.array([]).reshape(0, 4)
        y_empty = np.array([])
        classes = ['Non-Optimal (0)', 'Optimal (1)']
        
        # This should raise an error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            evaluate_model(sample_model, X=X_empty, y=y_empty, classes=classes)
