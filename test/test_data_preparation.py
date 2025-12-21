"""
Unit tests for data_preparation module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.training.data_preparation import (
    data_preparation,
    data_preparation_train_test,
    save_data_train_test
)
from src.constants import c_PROFILE_COLUMNS, c_TRAIN_CYCLES


class TestDataPreparation:
    """Test cases for data_preparation function."""
    
    @patch('src.training.data_preparation.get_data')
    @patch('src.training.data_preparation.extract_features')
    def test_data_preparation_success(self, mock_extract_features, mock_get_data):
        """Test data preparation successfully."""
        # Mock data
        mock_profile = pd.DataFrame({
            'Cooler_Condition': [100, 100, 90],
            'Valve_Condition': [100, 100, 80],
            'Internal_Pump_Leakage': [0, 0, 0],
            'Hydraulic_Accumulator': [130, 130, 130],
            'Stable_Flag': [1, 1, 1]
        })
        
        mock_ps2 = pd.DataFrame(np.random.randn(3, 100))
        mock_fs1 = pd.DataFrame(np.random.randn(3, 100))
        
        mock_get_data.return_value = {
            'data_profile': mock_profile,
            'data_ps2': mock_ps2,
            'data_fs1': mock_fs1
        }
        
        mock_ps2_features = pd.DataFrame({
            'PS2_features_mean': [1.0, 2.0, 3.0],
            'PS2_features_std': [0.5, 0.6, 0.7]
        })
        
        mock_fs1_features = pd.DataFrame({
            'FS1_features_mean': [1.5, 2.5, 3.5],
            'FS1_features_std': [0.4, 0.5, 0.6]
        })
        
        mock_extract_features.side_effect = [mock_ps2_features, mock_fs1_features]
        
        result = data_preparation()
        
        assert isinstance(result, pd.DataFrame)
        assert 'Target' in result.columns
        assert 'Valve_Condition' in result.columns
        # Check that Target is binary (0 or 1)
        assert set(result['Target'].unique()).issubset({0, 1})
    
    @patch('src.training.data_preparation.get_data')
    def test_data_preparation_target_creation(self, mock_get_data):
        """Test that Target column is created correctly."""
        mock_profile = pd.DataFrame({
            'Cooler_Condition': [100, 100, 90],
            'Valve_Condition': [100, 50, 100],  # 100 = optimal, <100 = non-optimal
            'Internal_Pump_Leakage': [0, 0, 0],
            'Hydraulic_Accumulator': [130, 130, 130],
            'Stable_Flag': [1, 1, 1]
        })
        
        mock_ps2 = pd.DataFrame(np.random.randn(3, 100))
        mock_fs1 = pd.DataFrame(np.random.randn(3, 100))
        
        mock_get_data.return_value = {
            'data_profile': mock_profile,
            'data_ps2': mock_ps2,
            'data_fs1': mock_fs1
        }
        
        with patch('src.training.data_preparation.extract_features') as mock_extract:
            mock_extract.side_effect = [
                pd.DataFrame({'PS2_features_mean': [1, 2, 3]}),
                pd.DataFrame({'FS1_features_mean': [1, 2, 3]})
            ]
            
            result = data_preparation()
            
            # Valve_Condition == 100 should give Target == 1
            assert result.iloc[0]['Target'] == 1  # Valve_Condition = 100
            assert result.iloc[1]['Target'] == 0  # Valve_Condition = 50
            assert result.iloc[2]['Target'] == 1  # Valve_Condition = 100


class TestDataPreparationTrainTest:
    """Test cases for data_preparation_train_test function."""
    
    @patch('src.training.data_preparation.data_preparation')
    def test_data_preparation_train_test_success(self, mock_data_prep):
        """Test train/test split preparation successfully."""
        # Create mock data with more than TRAIN_CYCLES rows
        n_total = c_TRAIN_CYCLES + 100
        mock_data = pd.DataFrame({
            'Cooler_Condition': [100] * n_total,
            'Valve_Condition': [100] * n_total,
            'Internal_Pump_Leakage': [0] * n_total,
            'Hydraulic_Accumulator': [130] * n_total,
            'Stable_Flag': [1] * n_total,
            'Target': [1] * n_total,
            'PS2_features_mean': np.random.randn(n_total),
            'PS2_features_std': np.random.randn(n_total),
            'FS1_features_mean': np.random.randn(n_total),
            'FS1_features_std': np.random.randn(n_total)
        })
        
        mock_data_prep.return_value = mock_data
        
        X_train, X_test, y_train, y_test = data_preparation_train_test()
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check shapes
        assert len(X_train) == c_TRAIN_CYCLES
        assert len(y_train) == c_TRAIN_CYCLES
        assert len(X_test) == n_total - c_TRAIN_CYCLES
        assert len(y_test) == n_total - c_TRAIN_CYCLES
        
        # Check that profile columns are not in X
        for col in c_PROFILE_COLUMNS + ['Target']:
            assert col not in X_train.columns
            assert col not in X_test.columns
    
    @patch('src.training.data_preparation.data_preparation')
    def test_data_preparation_train_test_insufficient_data(self, mock_data_prep):
        """Test train/test split with insufficient data."""
        # Create mock data with less than TRAIN_CYCLES rows
        n_total = 100
        mock_data = pd.DataFrame({
            'Cooler_Condition': [100] * n_total,
            'Valve_Condition': [100] * n_total,
            'Internal_Pump_Leakage': [0] * n_total,
            'Hydraulic_Accumulator': [130] * n_total,
            'Stable_Flag': [1] * n_total,
            'Target': [1] * n_total,
            'PS2_features_mean': np.random.randn(n_total),
            'FS1_features_mean': np.random.randn(n_total)
        })
        
        mock_data_prep.return_value = mock_data
        
        X_train, X_test, y_train, y_test = data_preparation_train_test()
        
        # X_train should have all available data
        assert len(X_train) == n_total
        # X_test should be empty
        assert len(X_test) == 0
        assert len(y_test) == 0


class TestSaveDataTrainTest:
    """Test cases for save_data_train_test function."""
    
    @patch('src.training.data_preparation.data_preparation_train_test')
    @patch('src.training.data_preparation.save_dataset')
    def test_save_data_train_test_success(self, mock_save_dataset, mock_data_prep):
        """Test saving train/test data successfully."""
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        X_test = pd.DataFrame({'feature1': [4, 5]})
        y_train = pd.Series([0, 1, 0])
        y_test = pd.Series([1, 0])
        
        mock_data_prep.return_value = (X_train, X_test, y_train, y_test)
        
        save_data_train_test()
        
        # Should be called 4 times (X_train, X_test, y_train, y_test)
        assert mock_save_dataset.call_count == 4
    
    @patch('src.training.data_preparation.data_preparation_train_test')
    def test_save_data_train_test_error(self, mock_data_prep):
        """Test saving train/test data with error."""
        mock_data_prep.side_effect = Exception("Data preparation error")
        
        with pytest.raises(Exception):
            save_data_train_test()
