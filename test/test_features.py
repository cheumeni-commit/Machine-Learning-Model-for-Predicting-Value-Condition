"""
Unit tests for features module.
"""
import pytest
import pandas as pd
import numpy as np

from src.training.features import extract_features


class TestExtractFeatures:
    """Test cases for extract_features function."""
    
    def test_extract_features_success(self, sample_sensor_data):
        """Test extracting features successfully."""
        sensor_name = 'PS2'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_sensor_data)
        
        # Check that expected features are present
        expected_features = [
            f'{sensor_name}_mean',
            f'{sensor_name}_std',
            f'{sensor_name}_min',
            f'{sensor_name}_max',
            f'{sensor_name}_median',
            f'{sensor_name}_q25',
            f'{sensor_name}_q75',
            f'{sensor_name}_range',
            f'{sensor_name}_rms'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns
    
    def test_extract_features_mean(self, sample_sensor_data):
        """Test that mean feature is calculated correctly."""
        sensor_name = 'TEST'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        # Check that mean values are reasonable
        mean_col = f'{sensor_name}_mean'
        assert mean_col in features_df.columns
        assert not features_df[mean_col].isna().any()
    
    def test_extract_features_std(self, sample_sensor_data):
        """Test that std feature is calculated correctly."""
        sensor_name = 'TEST'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        std_col = f'{sensor_name}_std'
        assert std_col in features_df.columns
        assert not features_df[std_col].isna().any()
        # Standard deviation should be non-negative
        assert (features_df[std_col] >= 0).all()
    
    def test_extract_features_min_max(self, sample_sensor_data):
        """Test that min and max features are calculated correctly."""
        sensor_name = 'TEST'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        min_col = f'{sensor_name}_min'
        max_col = f'{sensor_name}_max'
        
        assert min_col in features_df.columns
        assert max_col in features_df.columns
        
        # Max should be >= min
        assert (features_df[max_col] >= features_df[min_col]).all()
    
    def test_extract_features_range(self, sample_sensor_data):
        """Test that range feature is calculated correctly."""
        sensor_name = 'TEST'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        range_col = f'{sensor_name}_range'
        min_col = f'{sensor_name}_min'
        max_col = f'{sensor_name}_max'
        
        # Range should equal max - min
        expected_range = features_df[max_col] - features_df[min_col]
        pd.testing.assert_series_equal(features_df[range_col], expected_range)
    
    def test_extract_features_quantiles(self, sample_sensor_data):
        """Test that quantile features are calculated correctly."""
        sensor_name = 'TEST'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        q25_col = f'{sensor_name}_q25'
        q75_col = f'{sensor_name}_q75'
        median_col = f'{sensor_name}_median'
        
        assert q25_col in features_df.columns
        assert q75_col in features_df.columns
        assert median_col in features_df.columns
        
        # q25 <= median <= q75
        assert (features_df[q25_col] <= features_df[median_col]).all()
        assert (features_df[median_col] <= features_df[q75_col]).all()
    
    def test_extract_features_rms(self, sample_sensor_data):
        """Test that RMS feature is calculated correctly."""
        sensor_name = 'TEST'
        features_df = extract_features(sample_sensor_data, sensor_name)
        
        rms_col = f'{sensor_name}_rms'
        assert rms_col in features_df.columns
        
        # RMS should be non-negative
        assert (features_df[rms_col] >= 0).all()
    
    def test_extract_features_empty_dataframe(self):
        """Test extracting features from empty dataframe raises error."""
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            extract_features(empty_df, 'TEST')
    
    def test_extract_features_different_sensor_names(self, sample_sensor_data):
        """Test that different sensor names produce different column names."""
        ps2_features = extract_features(sample_sensor_data, 'PS2')
        fs1_features = extract_features(sample_sensor_data, 'FS1')
        
        # Column names should be different
        assert all(col.startswith('PS2_') for col in ps2_features.columns)
        assert all(col.startswith('FS1_') for col in fs1_features.columns)
        
        # But should have same number of features
        assert len(ps2_features.columns) == len(fs1_features.columns)
