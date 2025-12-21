"""
Unit tests for models module.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.training.models import get_models, _load_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


class TestLoadModels:
    """Test cases for _load_models function."""
    
    @patch('src.training.models.get_config')
    def test_load_models_success(self, mock_get_config):
        """Test loading models successfully."""
        mock_config = {
            'RandomForestClassifier': {
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10]
                }
            },
            'SGDClassifier': {
                'params': {
                    'alpha': [0.01, 0.03],
                    'max_iter': [10, 20]
                }
            }
        }
        mock_get_config.return_value = mock_config
        
        models, params = _load_models()
        
        assert len(models) == 2
        assert len(params) == 2
        assert RandomForestClassifier in models
        assert SGDClassifier in models
        
        # Check params structure
        assert isinstance(params[0], dict)
        assert isinstance(params[1], dict)
    
    @patch('src.training.models.get_config')
    def test_load_models_missing_config(self, mock_get_config):
        """Test loading models with missing config raises error."""
        mock_get_config.return_value = {}
        
        with pytest.raises(Exception):
            _load_models()
    
    @patch('src.training.models.get_config')
    def test_load_models_incomplete_config(self, mock_get_config):
        """Test loading models with incomplete config."""
        mock_config = {
            'RandomForestClassifier': {
                'params': {'n_estimators': [50]}
            }
            # Missing SGDClassifier
        }
        mock_get_config.return_value = mock_config
        
        models, params = _load_models()
        
        # Should only return available models
        assert len(models) == 1
        assert RandomForestClassifier in models


class TestGetModels:
    """Test cases for get_models function."""
    
    @patch('src.training.models._load_models')
    def test_get_models_success(self, mock_load_models):
        """Test getting models successfully."""
        mock_models = [RandomForestClassifier, SGDClassifier]
        mock_params = [
            {'n_estimators': [50, 100]},
            {'alpha': [0.01, 0.03]}
        ]
        mock_load_models.return_value = (mock_models, mock_params)
        
        models, params = get_models()
        
        assert models == mock_models
        assert params == mock_params
        mock_load_models.assert_called_once()
    
    @patch('src.training.models._load_models')
    def test_get_models_error(self, mock_load_models):
        """Test getting models with error raises exception."""
        mock_load_models.side_effect = Exception("Load error")
        
        with pytest.raises(Exception):
            get_models()
    
    @patch('src.training.models.get_config')
    def test_get_models_integration(self, mock_get_config):
        """Test get_models integration with config."""
        mock_config = {
            'RandomForestClassifier': {
                'params': {
                    'n_estimators': [50],
                    'max_depth': [5]
                }
            },
            'SGDClassifier': {
                'params': {
                    'alpha': [0.01],
                    'max_iter': [10]
                }
            }
        }
        mock_get_config.return_value = mock_config
        
        models, params = get_models()
        
        assert len(models) == 2
        assert len(params) == 2
        assert all(isinstance(m, type) for m in models)
        assert all(isinstance(p, dict) for p in params)
