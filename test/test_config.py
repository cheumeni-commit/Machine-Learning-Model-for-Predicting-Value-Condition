"""
Unit tests for config module.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.config.config import load_param, load_config_file, get_config


class TestLoadParam:
    """Test cases for load_param function."""
    
    def test_load_param_success(self, sample_config_file):
        """Test loading parameters from YAML file successfully."""
        result = load_param(str(sample_config_file))
        
        assert isinstance(result, dict)
        assert 'RandomForestClassifier' in result
        assert 'SGDClassifier' in result
        assert 'params' in result['RandomForestClassifier']
    
    def test_load_param_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(Exception):
            load_param('/nonexistent/file.yml')
    
    def test_load_param_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML file raises error."""
        invalid_yaml_path = temp_dir / 'invalid.yml'
        with open(invalid_yaml_path, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with pytest.raises(Exception):
            load_param(str(invalid_yaml_path))


class TestLoadConfigFile:
    """Test cases for load_config_file function."""
    
    @patch('src.config.config.dirs')
    @patch('src.config.config.load_param')
    def test_load_config_file_success(self, mock_load_param, mock_dirs, sample_config_file):
        """Test loading config file successfully."""
        mock_dirs.config = Path(sample_config_file).parent
        mock_load_param.return_value = {'test': 'config'}
        
        result = load_config_file()
        
        assert isinstance(result, dict)
        mock_load_param.assert_called_once()
    
    @patch('src.config.config.dirs')
    @patch('src.config.config.load_param')
    def test_load_config_file_error(self, mock_load_param, mock_dirs):
        """Test loading config file with error raises exception."""
        mock_dirs.config = Path('/nonexistent')
        mock_load_param.side_effect = Exception("File not found")
        
        with pytest.raises(Exception):
            load_config_file()


class TestGetConfig:
    """Test cases for get_config function."""
    
    @patch('src.config.config.load_config_file')
    def test_get_config_success(self, mock_load_config_file):
        """Test getting config successfully."""
        mock_config = {'test': 'config', 'models': {}}
        mock_load_config_file.return_value = mock_config
        
        result = get_config()
        
        assert result == mock_config
        mock_load_config_file.assert_called_once()
    
    @patch('src.config.config.load_config_file')
    def test_get_config_error(self, mock_load_config_file):
        """Test getting config with error raises exception."""
        mock_load_config_file.side_effect = Exception("Config error")
        
        with pytest.raises(Exception):
            get_config()
