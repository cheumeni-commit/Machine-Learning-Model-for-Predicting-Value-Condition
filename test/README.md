# Unit Tests

This directory contains unit tests for the Machine Learning Model for Predicting Value Condition project.

## Test Structure

The tests are organized to mirror the source code structure:

- `test_constants.py` - Tests for constants module
- `test_read_write.py` - Tests for data I/O utilities
- `test_config.py` - Tests for configuration loading
- `test_directories.py` - Tests for directory management
- `test_features.py` - Tests for feature extraction
- `test_data_preparation.py` - Tests for data preparation pipeline
- `test_models.py` - Tests for model registry and loading
- `test_evaluation.py` - Tests for model evaluation metrics
- `test_mlflow_manager.py` - Tests for MLflow integration
- `test_model_selector.py` - Tests for model selection functionality

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest test/test_constants.py
```

### Run specific test class
```bash
pytest test/test_constants.py::TestConstants
```

### Run specific test function
```bash
pytest test/test_constants.py::TestConstants::test_profile_constant
```

### Run with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

### Run only fast tests (exclude slow markers)
```bash
pytest -m "not slow"
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:
- `temp_dir` - Temporary directory for test files
- `sample_dataframe` - Sample pandas DataFrame
- `sample_sensor_data` - Sample sensor data
- `sample_profile_data` - Sample profile data
- `sample_train_test_data` - Sample train/test split
- `sample_model` - Sample trained model
- `sample_metrics` - Sample metrics dictionary
- `sample_config_file` - Sample config YAML file
- `sample_csv_file` - Sample CSV file
- `sample_json_file` - Sample JSON file

## Requirements

All tests require:
- pytest
- pytest-cov (for coverage reports)
- All project dependencies from `requirements.txt`

## Notes

- Tests use mocking extensively to avoid dependencies on external resources
- MLflow tests mock the MLflow API to avoid requiring a running MLflow server
- File I/O tests use temporary directories that are cleaned up automatically
- Some tests may require actual data files to be present in `data/raw/` directory
