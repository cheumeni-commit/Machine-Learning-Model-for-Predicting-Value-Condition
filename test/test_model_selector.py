"""
Unit tests for model_selector module.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.training.model_selector import (
    ModelSelector,
    select_best_model,
    compare_all_models
)


class TestModelSelector:
    """Test cases for ModelSelector class."""
    
    def test_init_default(self):
        """Test ModelSelector initialization with default artifacts_dir."""
        selector = ModelSelector()
        
        assert selector.artifacts_dir is not None
        assert selector.tracking_uri.startswith('file:')
    
    def test_init_custom_artifacts_dir(self, temp_dir):
        """Test ModelSelector initialization with custom artifacts_dir."""
        selector = ModelSelector(artifacts_dir=temp_dir)
        
        assert selector.artifacts_dir == temp_dir
        assert selector.tracking_uri == f"file:{temp_dir}"
    
    def test_get_all_runs_empty_dir(self, temp_dir):
        """Test getting all runs from empty directory."""
        selector = ModelSelector(artifacts_dir=temp_dir)
        runs = selector.get_all_runs()
        
        assert isinstance(runs, list)
        assert len(runs) == 0
    
    def test_get_all_runs_nonexistent_dir(self):
        """Test getting all runs from nonexistent directory."""
        nonexistent_dir = Path('/nonexistent/path/12345')
        selector = ModelSelector(artifacts_dir=nonexistent_dir)
        runs = selector.get_all_runs()
        
        assert isinstance(runs, list)
        assert len(runs) == 0
    
    def test_extract_run_info_success(self, temp_dir):
        """Test extracting run info successfully."""
        # Create mock run directory structure
        exp_dir = temp_dir / '123'
        run_dir = exp_dir / 'run_123'
        metrics_dir = run_dir / 'metrics'
        params_dir = run_dir / 'params'
        artifacts_dir = run_dir / 'artifacts' / 'models'
        
        metrics_dir.mkdir(parents=True)
        params_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)
        
        # Create metric file
        metric_file = metrics_dir / 'test_accuracy'
        with open(metric_file, 'w') as f:
            f.write('1000 0.95 0\n')
        
        # Create param file
        param_file = params_dir / 'n_estimators'
        with open(param_file, 'w') as f:
            f.write('100')
        
        # Create model file
        model_file = artifacts_dir / 'model.pkl'
        model_file.touch()
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        run_info = selector._extract_run_info(run_dir)
        
        assert run_info is not None
        assert run_info['run_id'] == 'run_123'
        assert 'test_accuracy' in run_info['metrics']
        assert run_info['metrics']['test_accuracy'] == 0.95
        assert 'n_estimators' in run_info['params']
        assert run_info['has_model'] is True
    
    def test_extract_run_info_no_model(self, temp_dir):
        """Test extracting run info without model."""
        exp_dir = temp_dir / '123'
        run_dir = exp_dir / 'run_123'
        metrics_dir = run_dir / 'metrics'
        
        metrics_dir.mkdir(parents=True)
        
        metric_file = metrics_dir / 'test_accuracy'
        with open(metric_file, 'w') as f:
            f.write('1000 0.95 0\n')
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        run_info = selector._extract_run_info(run_dir)
        
        assert run_info is not None
        assert run_info['has_model'] is False
    
    def test_get_best_run_success(self, temp_dir):
        """Test getting best run successfully."""
        # Create mock runs
        exp_dir = temp_dir / '123'
        run1_dir = exp_dir / 'run_1'
        run2_dir = exp_dir / 'run_2'
        
        for run_dir, accuracy in [(run1_dir, 0.92), (run2_dir, 0.95)]:
            metrics_dir = run_dir / 'metrics'
            artifacts_dir = run_dir / 'artifacts' / 'models'
            metrics_dir.mkdir(parents=True)
            artifacts_dir.mkdir(parents=True)
            
            metric_file = metrics_dir / 'test_accuracy'
            with open(metric_file, 'w') as f:
                f.write(f'1000 {accuracy} 0\n')
            
            model_file = artifacts_dir / 'model.pkl'
            model_file.touch()
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        best_run = selector.get_best_run(metric='test_accuracy')
        
        assert best_run is not None
        assert best_run['run_id'] == 'run_2'  # Higher accuracy
        assert best_run['metrics']['test_accuracy'] == 0.95
    
    def test_get_best_run_no_valid_runs(self, temp_dir):
        """Test getting best run when no valid runs exist."""
        selector = ModelSelector(artifacts_dir=temp_dir)
        best_run = selector.get_best_run(metric='test_accuracy')
        
        assert best_run is None
    
    def test_get_best_run_lower_is_better(self, temp_dir):
        """Test getting best run when lower is better."""
        # Create mock runs with loss metric (lower is better)
        exp_dir = temp_dir / '123'
        run1_dir = exp_dir / 'run_1'
        run2_dir = exp_dir / 'run_2'
        
        for run_dir, loss in [(run1_dir, 0.5), (run2_dir, 0.3)]:
            metrics_dir = run_dir / 'metrics'
            artifacts_dir = run_dir / 'artifacts' / 'models'
            metrics_dir.mkdir(parents=True)
            artifacts_dir.mkdir(parents=True)
            
            metric_file = metrics_dir / 'loss'
            with open(metric_file, 'w') as f:
                f.write(f'1000 {loss} 0\n')
            
            model_file = artifacts_dir / 'model.pkl'
            model_file.touch()
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        best_run = selector.get_best_run(metric='loss', higher_is_better=False)
        
        assert best_run is not None
        assert best_run['run_id'] == 'run_2'  # Lower loss
        assert best_run['metrics']['loss'] == 0.3
    
    def test_compare_runs(self, temp_dir):
        """Test comparing runs."""
        # Create mock runs
        exp_dir = temp_dir / '123'
        runs_data = [
            ('run_1', 0.90),
            ('run_2', 0.95),
            ('run_3', 0.92)
        ]
        
        for run_id, accuracy in runs_data:
            run_dir = exp_dir / run_id
            metrics_dir = run_dir / 'metrics'
            metrics_dir.mkdir(parents=True)
            
            metric_file = metrics_dir / 'test_accuracy'
            with open(metric_file, 'w') as f:
                f.write(f'1000 {accuracy} 0\n')
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        sorted_runs = selector.compare_runs(metric='test_accuracy')
        
        assert len(sorted_runs) == 3
        # Should be sorted descending
        assert sorted_runs[0]['metrics']['test_accuracy'] == 0.95
        assert sorted_runs[1]['metrics']['test_accuracy'] == 0.92
        assert sorted_runs[2]['metrics']['test_accuracy'] == 0.90
    
    @patch('src.training.model_selector.mlflow')
    def test_load_model_from_run(self, mock_mlflow, temp_dir):
        """Test loading model from run."""
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        model = selector.load_model_from_run('run_123')
        
        assert model == mock_model
        mock_mlflow.sklearn.load_model.assert_called_once()
    
    @patch('src.training.model_selector.mlflow')
    def test_load_model_from_run_error(self, mock_mlflow, temp_dir):
        """Test loading model from run with error."""
        mock_mlflow.sklearn.load_model.side_effect = Exception("Load error")
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        model = selector.load_model_from_run('run_123')
        
        assert model is None
    
    @patch('src.training.model_selector.mlflow')
    def test_load_model_from_registry(self, mock_mlflow, temp_dir):
        """Test loading model from registry."""
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        model = selector.load_model_from_registry(
            model_name="test_model",
            version=1
        )
        
        assert model == mock_model
        mock_mlflow.sklearn.load_model.assert_called_once()
    
    @patch('src.training.model_selector.mlflow')
    def test_load_model_from_registry_latest(self, mock_mlflow, temp_dir):
        """Test loading latest model from registry."""
        mock_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = 2
        mock_client.get_latest_versions.return_value = [mock_version]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        model = selector.load_model_from_registry(model_name="test_model")
        
        assert model == mock_model
    
    def test_get_model_summary(self, temp_dir):
        """Test getting model summary."""
        # Create mock runs
        exp_dir = temp_dir / '123'
        run_dir = exp_dir / 'run_123'
        metrics_dir = run_dir / 'metrics'
        artifacts_dir = run_dir / 'artifacts' / 'models'
        
        metrics_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)
        
        metric_file = metrics_dir / 'test_accuracy'
        with open(metric_file, 'w') as f:
            f.write('1000 0.95 0\n')
        
        model_file = artifacts_dir / 'model.pkl'
        model_file.touch()
        
        selector = ModelSelector(artifacts_dir=temp_dir)
        summary = selector.get_model_summary()
        
        assert isinstance(summary, dict)
        assert 'total_runs' in summary
        assert 'runs_with_models' in summary
        assert 'available_metrics' in summary
        assert 'runs' in summary
        assert 'test_accuracy' in summary['available_metrics']


class TestCompareAllModels:
    """Test cases for compare_all_models function."""
    
    @patch('src.training.model_selector.ModelSelector')
    def test_compare_all_models_success(self, mock_selector_class):
        """Test comparing all models successfully."""
        mock_selector = MagicMock()
        mock_selector_class.return_value = mock_selector
        
        mock_runs = [
            {'run_id': 'run_1', 'metrics': {'test_accuracy': 0.95}},
            {'run_id': 'run_2', 'metrics': {'test_accuracy': 0.92}}
        ]
        mock_selector.compare_runs.return_value = mock_runs
        
        result = compare_all_models(metric="test_accuracy")
        
        assert result == mock_runs
        mock_selector.compare_runs.assert_called_once_with("test_accuracy")
