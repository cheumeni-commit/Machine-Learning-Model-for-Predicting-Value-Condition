"""
Pytest configuration and shared fixtures for unit tests.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import joblib
import json
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data (PS2 or FS1 format)."""
    # Simulate sensor data with 10 cycles and 100 measurements per cycle
    np.random.seed(42)
    data = np.random.randn(10, 100)
    return pd.DataFrame(data)


@pytest.fixture
def sample_profile_data():
    """Create sample profile data."""
    return pd.DataFrame({
        'Cooler_Condition': [100, 100, 90, 100, 100],
        'Valve_Condition': [100, 100, 80, 100, 100],
        'Internal_Pump_Leakage': [0, 0, 0, 0, 0],
        'Hydraulic_Accumulator': [130, 130, 130, 130, 130],
        'Stable_Flag': [1, 1, 1, 1, 1]
    })


@pytest.fixture
def sample_train_test_data():
    """Create sample train/test split data."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'PS2_features_mean': np.random.randn(20),
        'PS2_features_std': np.random.randn(20),
        'FS1_features_mean': np.random.randn(20),
        'FS1_features_std': np.random.randn(20)
    })
    X_test = pd.DataFrame({
        'PS2_features_mean': np.random.randn(5),
        'PS2_features_std': np.random.randn(5),
        'FS1_features_mean': np.random.randn(5),
        'FS1_features_std': np.random.randn(5)
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_test = pd.Series([0, 1, 0, 1, 0])
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.96,
        'f1': 0.95
    }


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample config YAML file."""
    config = {
        'RandomForestClassifier': {
            'name': 'RandomForestClassifier',
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10]
            }
        },
        'SGDClassifier': {
            'name': 'SGDClassifier',
            'params': {
                'alpha': [0.01, 0.03],
                'max_iter': [10, 20]
            }
        }
    }
    config_path = temp_dir / 'test_config.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    csv_path = temp_dir / 'test.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file."""
    data = {'key1': 'value1', 'key2': 42}
    json_path = temp_dir / 'test.json'
    with open(json_path, 'w') as f:
        json.dump(data, f)
    return json_path
