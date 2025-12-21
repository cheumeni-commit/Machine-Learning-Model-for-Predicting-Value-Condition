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
    def test_load_models_missing_config(self, mock_get_config):
        """Test loading models with missing config raises error."""
        mock_get_config.return_value = {}
        
        with pytest.raises(Exception):
            _load_models()


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
    