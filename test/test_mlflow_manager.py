"""
Unit tests for mlflow_manager module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tempfile
from pathlib import Path

from src.training.mlflow_manager import (
    MLflowManager,
    train_and_log_model,
    load_best_model
)


class TestMLflowManager:
    """Test cases for MLflowManager class."""
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_init_success(self, mock_mlflow):
        """Test MLflowManager initialization."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = '123'
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.create_experiment.side_effect = Exception("Experiment exists")
        
        manager = MLflowManager(experiment_name="Test Experiment")
        
        assert manager.experiment_name == "Test Experiment"
        assert manager.experiment_id == '123'
        mock_mlflow.set_tracking_uri.assert_called_once()
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_start_run(self, mock_mlflow):
        """Test starting a run."""
        mock_run = MagicMock()
        mock_run.info.run_id = 'run_123'
        mock_mlflow.start_run.return_value = mock_run
        
        manager = MLflowManager()
        run_id = manager.start_run(run_name="test_run", tags={'tag1': 'value1'})
        
        assert run_id == 'run_123'
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.set_tag.assert_called()
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_end_run(self, mock_mlflow):
        """Test ending a run."""
        manager = MLflowManager()
        manager.end_run()
        
        mock_mlflow.end_run.assert_called_once()
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_log_params(self, mock_mlflow):
        """Test logging parameters."""
        manager = MLflowManager()
        params = {'param1': 'value1', 'param2': 42}
        
        manager.log_params(params)
        
        assert mock_mlflow.log_param.call_count == 2
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        manager = MLflowManager()
        metrics = {'accuracy': 0.95, 'f1': 0.92}
        
        manager.log_metrics(metrics)
        
        assert mock_mlflow.log_metric.call_count == 2
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_log_model(self, mock_mlflow):
        """Test logging a model."""
        manager = MLflowManager()
        mock_model = MagicMock()
        
        manager.log_model(mock_model, model_name="test_model")
        
        mock_mlflow.sklearn.log_model.assert_called_once()
    
    @patch('src.training.mlflow_manager.mlflow')
    @patch('src.training.mlflow_manager.dirs')
    def test_log_artifact(self, mock_dirs, mock_mlflow):
        """Test logging an artifact."""
        manager = MLflowManager()
        test_path = '/test/path/file.txt'
        
        manager.log_artifact(test_path)
        
        mock_mlflow.log_artifact.assert_called_once()
    
    @patch('src.training.mlflow_manager.mlflow')
    @patch('src.training.mlflow_manager.dirs')
    def test_log_dataframe(self, mock_dirs, mock_mlflow):
        """Test logging a DataFrame."""
        manager = MLflowManager()
        mock_dirs.raw_store_dir = Path('/test')
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with patch('builtins.open', create=True):
            with patch('pandas.DataFrame.to_csv'):
                with patch('os.remove'):
                    manager.log_dataframe(df, name="test_data")
        
        # Should attempt to log artifact
        # (actual file operations are mocked)
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_log_feature_importance(self, mock_mlflow):
        """Test logging feature importance."""
        manager = MLflowManager()
        feature_names = ['feature1', 'feature2']
        importances = np.array([0.5, 0.3])
        
        with patch.object(manager, 'log_dataframe'):
            manager.log_feature_importance(feature_names, importances)
        
        # Should call log_dataframe
        # (actual implementation details are tested)
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_log_confusion_matrix(self, mock_mlflow):
        """Test logging confusion matrix."""
        manager = MLflowManager()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        with patch.object(manager, 'log_dataframe'):
            manager.log_confusion_matrix(y_true, y_pred)
        
        # Should call log_dataframe with confusion matrix
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_log_classification_report(self, mock_mlflow):
        """Test logging classification report."""
        manager = MLflowManager()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        with patch.object(manager, 'log_dataframe'):
            manager.log_classification_report(y_true, y_pred)
        
        # Should call log_dataframe with classification report
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_get_best_run(self, mock_mlflow):
        """Test getting best run."""
        manager = MLflowManager()
        manager.experiment_id = '123'
        
        mock_experiment = MagicMock()
        mock_mlflow.get_experiment.return_value = mock_experiment
        
        mock_runs = pd.DataFrame({
            'run_id': ['run1', 'run2'],
            'metrics.test_accuracy': [0.95, 0.92],
            'start_time': [1000, 2000]
        })
        mock_mlflow.search_runs.return_value = mock_runs
        
        best_run = manager.get_best_run(metric="test_accuracy")
        
        assert best_run is not None
        assert 'run_id' in best_run
        assert 'metrics' in best_run
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_get_best_run_no_runs(self, mock_mlflow):
        """Test getting best run when no runs exist."""
        manager = MLflowManager()
        manager.experiment_id = '123'
        
        mock_experiment = MagicMock()
        mock_mlflow.get_experiment.return_value = mock_experiment
        mock_mlflow.search_runs.return_value = pd.DataFrame()
        
        best_run = manager.get_best_run(metric="test_accuracy")
        
        assert best_run is None
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_load_model(self, mock_mlflow):
        """Test loading a model."""
        manager = MLflowManager()
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        result = manager.load_model('run_123')
        
        assert result == mock_model
        mock_mlflow.sklearn.load_model.assert_called_once()
    
    @patch('src.training.mlflow_manager.mlflow')
    def test_get_experiment_runs(self, mock_mlflow):
        """Test getting experiment runs."""
        manager = MLflowManager()
        manager.experiment_id = '123'
        
        mock_runs = pd.DataFrame({'run_id': ['run1', 'run2']})
        mock_mlflow.search_runs.return_value = mock_runs
        
        runs = manager.get_experiment_runs()
        
        assert isinstance(runs, pd.DataFrame)
        assert len(runs) == 2


class TestTrainAndLogModel:
    """Test cases for train_and_log_model function."""
    
    @patch('src.training.mlflow_manager.MLflowManager')
    def test_train_and_log_model_success(self, mock_manager_class, sample_model, sample_train_test_data):
        """Test training and logging a model successfully."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.start_run.return_value = 'run_123'
        
        model, metrics = train_and_log_model(
            X_train, y_train, X_test, y_test,
            sample_model, mock_manager, run_name="test_run"
        )
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        mock_manager.log_params.assert_called_once()
        mock_manager.log_metrics.assert_called_once()
        mock_manager.log_model.assert_called_once()
        mock_manager.end_run.assert_called_once()
    
    @patch('src.training.mlflow_manager.MLflowManager')
    def test_train_and_log_model_error(self, mock_manager_class, sample_model, sample_train_test_data):
        """Test training and logging model with error."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.start_run.return_value = 'run_123'
        
        # Make model.fit raise an error
        sample_model.fit = MagicMock(side_effect=Exception("Training error"))
        
        with pytest.raises(Exception):
            train_and_log_model(
                X_train, y_train, X_test, y_test,
                sample_model, mock_manager
            )
        
        # Should still end the run
        mock_manager.end_run.assert_called_once()


class TestLoadBestModel:
    """Test cases for load_best_model function."""
    
    @patch('src.training.mlflow_manager.MLflowManager')
    def test_load_best_model_success(self, mock_manager_class):
        """Test loading best model successfully."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        mock_best_run = {
            'run_id': 'run_123',
            'metrics': {'test_accuracy': 0.95}
        }
        mock_manager.get_best_run.return_value = mock_best_run
        
        mock_model = MagicMock()
        mock_manager.load_model.return_value = mock_model
        
        result = load_best_model(mock_manager, metric="test_accuracy")
        
        assert result == mock_model
        mock_manager.get_best_run.assert_called_once()
        mock_manager.load_model.assert_called_once()
    
    @patch('src.training.mlflow_manager.MLflowManager')
    def test_load_best_model_no_runs(self, mock_manager_class):
        """Test loading best model when no runs exist."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.get_best_run.return_value = None
        
        result = load_best_model(mock_manager, metric="test_accuracy")
        
        assert result is None
        mock_manager.load_model.assert_not_called()
