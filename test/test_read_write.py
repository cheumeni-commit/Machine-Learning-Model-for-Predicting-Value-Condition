"""
Unit tests for read_write module.
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import tempfile
import shutil

from src.read_write import (
    save_dataset,
    load_dataset,
    save_model,
    load_model,
    save_metrics,
    load_json_file,
    save_prediction
)
from src.config.directories import directories as dirs


class TestSaveDataset:
    """Test cases for save_dataset function."""

    
    def test_save_dataset_invalid_path(self, sample_dataframe):
        """Test saving dataset with invalid path raises error."""
        invalid_path = dirs.test_dir / 'test.csv'
        with pytest.raises(Exception):
            save_dataset(sample_dataframe, path=invalid_path)


class TestLoadDataset:
    """Test cases for load_dataset function."""
    
    def test_load_dataset_success(self, temp_dir):
        """Test loading a dataset successfully."""
        # Create a test file with tab separator
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        test_path = temp_dir / 'test_data.txt'
        test_data.to_csv(test_path, sep='\t', index=False, header=False)
        
        loaded_data = load_dataset(str(test_path))
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == (3, 2)
    
    def test_load_dataset_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(Exception):
            load_dataset('/nonexistent/file.txt')


class TestSaveModel:
    """Test cases for save_model function."""
    
    def test_save_model_success(self, temp_dir, sample_model):
        """Test saving a model successfully."""
        path = temp_dir / 'test_model.pkl'
        save_model(sample_model, path=str(path))
        
        assert path.exists()
        loaded_model = joblib.load(path)
        assert type(loaded_model) == type(sample_model)
    
    def test_save_model_invalid_path(self, sample_model):
        """Test saving model with invalid path raises error."""
        with pytest.raises(Exception):
            save_model(sample_model, path='/nonexistent/path/model.pkl')


class TestLoadModel:
    """Test cases for load_model function."""
    
    def test_load_model_success(self, temp_dir, sample_model):
        """Test loading a model successfully."""
        path = temp_dir / 'test_model.pkl'
        joblib.dump(sample_model, path)
        
        loaded_model = load_model(str(path))
        assert type(loaded_model) == type(sample_model)
    
    def test_load_model_nonexistent_file_with_default(self):
        """Test loading nonexistent model returns default."""
        default = {'test': 'value'}
        result = load_model('/nonexistent/model.pkl', default=default)
        assert result == default
    


class TestSaveMetrics:
    """Test cases for save_metrics function."""
    
    def test_save_metrics_success(self, temp_dir, sample_metrics):
        """Test saving metrics successfully."""
        path = temp_dir / 'test_metrics.json'
        save_metrics(sample_metrics, path=str(path))
        
        assert path.exists()
        with open(path, 'r') as f:
            loaded_metrics = json.load(f)
        assert loaded_metrics == sample_metrics
    
    def test_save_metrics_invalid_path(self, sample_metrics):
        """Test saving metrics with invalid path raises error."""
        with pytest.raises(Exception):
            save_metrics(sample_metrics, path='/nonexistent/path/metrics.json')


class TestLoadJsonFile:
    """Test cases for load_json_file function."""
    
    def test_load_json_file_success(self, temp_dir):
        """Test loading JSON file successfully."""
        test_data = {'key1': 'value1', 'key2': 42}
        path = temp_dir / 'test.json'
        with open(path, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = load_json_file(str(path))
        assert loaded_data == test_data


class TestSavePrediction:
    """Test cases for save_prediction function."""
    
    def test_save_prediction_success(self, temp_dir):
        """Test saving predictions successfully."""
        predictions = {'prediction1': 0, 'prediction2': 1}
        path = temp_dir / 'predictions.json'
        save_prediction(predictions, path=str(path))
        
        assert path.exists()
        with open(path, 'r') as f:
            loaded_predictions = json.load(f)
        assert loaded_predictions == predictions
    
    def test_save_prediction_invalid_path(self):
        """Test saving predictions with invalid path raises error."""
        predictions = {'prediction1': 0}
        with pytest.raises(Exception):
            save_prediction(predictions, path='/nonexistent/path/predictions.json')
