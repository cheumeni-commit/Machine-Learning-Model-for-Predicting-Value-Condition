"""
Unit tests for features module.
"""
import pytest
import pandas as pd

from src.training.features import extract_features


class TestExtractFeatures:
    """Test cases for extract_features function."""
    
    def test_extract_features_success(self, sample_sensor_data):
        """Test extracting features successfully."""
        sensor_name = 'PS2'
        features_df = extract_features(sample_sensor_data, sensor_name, 'PS2')
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_sensor_data)
        
        # Check that expected features are present
        expected_features = [
            f'{sensor_name}_q25',
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns

    
    def test_extract_features_empty_dataframe(self):
        """Test extracting features from empty dataframe raises error."""
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            extract_features(empty_df, 'TEST')
