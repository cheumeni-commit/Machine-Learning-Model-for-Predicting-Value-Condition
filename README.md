# Valve Condition Predictor

A machine learning system for predictive maintenance of hydraulic systems, designed to predict valve condition (optimal vs non-optimal) based on sensor data from hydraulic system cycles.

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting valve condition in hydraulic systems. It uses sensor data (pressure and flow rate) collected during production cycles to classify whether a valve is operating at optimal condition (100%) or requires maintenance.

**Key Features:**
- 🤖 Multiple ML models with automated hyperparameter tuning
- 📊 MLflow integration for experiment tracking and model versioning
- 🔍 SHAP explainability for model interpretability
- 📈 Data drift monitoring using Evidently AI
- 🚀 Interactive Streamlit web application
- 🎯 Model selection and comparison tools
- 🐳 Docker containerization for easy deployment
- 🔄 CI/CD pipeline with automated testing

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Project Structure](#-project-structure)
- [Model Selection](#-model-selection)
- [Streamlit Application](#-streamlit-application)
- [Training Pipeline](#-training-pipeline)
- [Testing](#-testing)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 📊 Dataset

The project uses the **"Condition Monitoring of Hydraulic Systems"** dataset from the UCI Machine Learning Repository.

- **Total cycles**: 2,205
- **Training cycles**: 2,000
- **Test cycles**: 205
- **Sensors**:
  - **PS2** (Pressure Sensor) - 100 Hz sampling rate
  - **FS1** (Flow Sensor) - 10 Hz sampling rate
- **Target**: Valve condition (Optimal: 100% vs Non-Optimal: <100%)
- **Features**: 18 statistical features extracted from sensor signals

## ✨ Features

### 1. Model Training & Experimentation
- Automated model training with multiple algorithms (Random Forest, XGBoost, etc.)
- Grid search for hyperparameter optimization
- Comprehensive evaluation metrics (accuracy, F1, precision, recall)
- MLflow tracking for all experiments

### 2. Model Selection
- Compare models across different metrics
- Select best model based on custom criteria
- Load models from MLflow artifacts or model registry

### 3. Explainability (SHAP)
- Local explainability (individual predictions)
- Global feature importance
- Force plots and waterfall plots
- Feature dependency analysis

### 4. Drift Monitoring
- Data distribution drift detection
- Feature-level drift analysis
- Outlier detection
- Statistical comparison between training and production data

### 5. Web Application
- Interactive prediction interface
- Real-time model performance visualization
- SHAP explanations integration
- Drift monitoring dashboard
- MLflow experiment browser

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Machine-Learning-Model-for-Predicting-Value-Condition
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

## 🏃 Quick Start

### 1. Prepare the Dataset

Place your raw data files in the `data/raw/` directory:
- `profile.txt` - System profile data
- `PS2.txt` - Pressure sensor data
- `FS1.txt` - Flow sensor data

Then run the data preparation script:
```bash
python src/run_dataset.py
```

### 2. Train a Model

Train models with MLflow tracking:
```bash
python src/run_train.py
```

This will:
- Load and preprocess the data
- Train multiple models with hyperparameter tuning
- Log all experiments to MLflow
- Save the best model artifacts

### 3. Launch the Streamlit App

Start the interactive web application:
```bash
streamlit run src/api/app_streamlit.py
```

Access the app at `http://localhost:8501`

### 4. View MLflow Experiments

Launch the MLflow UI:
```bash
mlflow ui --backend-store-uri file:./src/storage/mlflow_artifacts
```

Access at `http://localhost:5000`

## 🐳 Docker Deployment

The project includes Docker support for easy deployment and containerization.

### Prerequisites

- Docker Engine 20.10 or higher
- Docker Compose 2.0 or higher (optional, for docker-compose)

### Building Docker Images

#### Streamlit Application

Build the Streamlit application image:
```bash
docker build -f Dockerfile -t valve-condition-predictor:latest .
```

#### Training Pipeline

Build the training pipeline image:
```bash
docker build -f Dockerfile.train -t valve-condition-predictor-train:latest .
```

### Running with Docker

#### Streamlit Application

Run the Streamlit app in a container:

**Linux/Mac:**
```bash
docker run -d \
  --name valve-condition-predictor \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/storage:/app/storage \
  valve-condition-predictor:latest
```

**Windows (PowerShell):**
```powershell
docker run -d `
  --name valve-condition-predictor `
  -p 8501:8501 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/mlruns:/app/mlruns `
  -v ${PWD}/storage:/app/storage `
  valve-condition-predictor:latest
```

Access the app at `http://localhost:8501`

#### Training Pipeline

Run the training pipeline in a container:

**Linux/Mac:**
```bash
docker run -it \
  --name valve-condition-train \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/storage:/app/storage \
  valve-condition-predictor-train:latest
```

**Windows (PowerShell):**
```powershell
docker run -it `
  --name valve-condition-train `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/config:/app/config `
  -v ${PWD}/mlruns:/app/mlruns `
  -v ${PWD}/storage:/app/storage `
  valve-condition-predictor-train:latest
```

### Docker Compose

Use Docker Compose for easier management:

#### Start Streamlit Application
```bash
docker-compose up -d streamlit-app
```

#### Start MLflow UI
```bash
docker-compose --profile mlflow up -d mlflow-ui
```

#### Start Both Services
```bash
docker-compose --profile mlflow up -d
```

#### Stop Services
```bash
docker-compose down
```

#### View Logs
```bash
docker-compose logs -f streamlit-app
```

### Volume Mounts

The Docker setup uses volume mounts to persist:
- **Data**: `./data` - Training and test datasets
- **Configuration**: `./config` - Configuration files
- **MLflow Runs**: `./src/storage/mlflow_artifacts` - MLflow experiment tracking data
- **Storage**: `./storage` - Model artifacts and storage

### Health Checks

The Streamlit container includes a health check that monitors the application status. Check container health:
```bash
docker ps
```

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions CI/CD pipeline for automated testing and Docker image building.

### Pipeline Overview

The CI/CD pipeline (`/.github/workflows/ci.yml`) includes:

1. **Test Job**: Runs pytest tests across multiple Python versions (3.8, 3.9, 3.10)
2. **Build Docker Job**: Builds Docker images for both Streamlit app and training pipeline
3. **Lint Job**: Performs code quality checks using flake8 and pylint

### Trigger Events

The pipeline automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### Pipeline Jobs

#### Test Job
- **Matrix Strategy**: Tests across Python 3.8, 3.9, and 3.10
- **Steps**:
  - Checkout code
  - Set up Python environment
  - Install system dependencies (gcc, g++)
  - Install Python dependencies
  - Run pytest test suite

#### Build Docker Job
- **Dependencies**: Runs after successful test completion
- **Steps**:
  - Checkout code
  - Set up Docker Buildx
  - Build Streamlit app image
  - Build training pipeline image
  - Uses GitHub Actions cache for faster builds

#### Lint Job
- **Steps**:
  - Checkout code
  - Set up Python environment
  - Install linting tools (flake8, pylint)
  - Run code quality checks

### Viewing Pipeline Status

1. Go to the **Actions** tab in your GitHub repository
2. Click on a workflow run to see detailed logs
3. Each job shows individual test results and build status

### Local Testing

Before pushing, you can run the same checks locally:

```bash
# Run tests
pytest -v --tb=short

# Run linting
flake8 src/ test/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 src/ test/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Build Docker images
docker build -f Dockerfile -t valve-condition-predictor:latest .
docker build -f Dockerfile.train -t valve-condition-predictor-train:latest .
```

## 📖 Usage

### Model Selection

#### Quick Usage

```python
from src.training.model_selector import select_best_model

# Load the best model based on test_accuracy
best_model = select_best_model(metric="test_accuracy")
```

#### Advanced Usage

```python
from src.training.model_selector import ModelSelector

# Create a selector instance
selector = ModelSelector()

# Get summary of all available models
summary = selector.get_model_summary()
print(f"Available metrics: {summary['available_metrics']}")

# Compare all models by a metric
sorted_runs = selector.compare_runs(metric="test_accuracy")
for run in sorted_runs:
    print(f"Run {run['run_id']}: {run['metrics']['test_accuracy']:.4f}")

# Get the best run
best_run = selector.get_best_run(metric="test_accuracy")
print(f"Best run ID: {best_run['run_id']}")

# Load the best model
best_model = selector.load_best_model(metric="test_accuracy")

# Or load from model registry
model = selector.load_model_from_registry(
    model_name="valve_condition_model",
    version=None  # Latest version
)
```

### Available Metrics

Common metrics available for model selection:
- `test_accuracy` - Test set accuracy
- `test_f1` - Test set F1 score
- `test_precision` - Test set precision
- `test_recall` - Test set recall
- `train_accuracy` - Training set accuracy

### Making Predictions

```python
import joblib
import pandas as pd

# Load a trained model
model = joblib.load('path/to/model.pkl')

# Prepare your feature vector (18 features)
features = pd.DataFrame([[...]])  # Your feature values

# Make prediction
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"Prediction: {'Optimal' if prediction[0] == 1 else 'Non-Optimal'}")
print(f"Confidence: {probability[0].max():.2%}")
```

## 📁 Project Structure

```
Machine-Learning-Model-for-Predicting-Value-Condition/
│
├── config/                 # Configuration files
│   └── prod.yml           # Production configuration
│
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   │   ├── profile.txt
│   │   ├── PS2.txt
│   │   └── FS1.txt
│   └── intermediate/      # Processed data
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── src/                    # Source code
│   ├── api/               # API and web application
│   │   ├── app_streamlit.py      # Main Streamlit app
│   │   ├── shap_explainer.py     # SHAP integration
│   │   ├── drift_monitor.py      # Drift monitoring
│   │   └── streamlit_mlflow_page.py
│   │
│   ├── config/            # Configuration modules
│   │   ├── config.py
│   │   └── directories.py
│   │
│   ├── training/          # Training pipeline
│   │   ├── data_preparation.py
│   │   ├── features.py
│   │   ├── models.py
│   │   ├── evaluation.py
│   │   ├── mlflow_manager.py
│   │   └── model_selector.py
│   │
│   ├── storage/           # Model storage
│   │   └── mlflow_artifacts/
│   │
│   ├── constants.py       # Project constants
│   ├── read_write.py      # Data I/O utilities
│   ├── run_dataset.py     # Data preparation script
│   ├── run_train.py       # Training script
│   └── select_best_model.py  # Model selection example
│
├── test/                  # Test suite
│   ├── conftest.py        # Pytest configuration and fixtures
│   ├── test_config.py
│   ├── test_constants.py
│   ├── test_data_preparation.py
│   ├── test_directories.py
│   ├── test_evaluation.py
│   ├── test_features.py
│   ├── test_mlflow_manager.py
│   ├── test_model_selector.py
│   ├── test_models.py
│   └── test_read_write.py
│
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── pytest.ini            # Pytest configuration
├── Dockerfile             # Docker image for Streamlit app
├── Dockerfile.train       # Docker image for training pipeline
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker ignore patterns
├── .github/               # GitHub configuration
│   └── workflows/
│       └── ci.yml         # CI/CD pipeline definition
├── LICENSE                # Apache 2.0 License
└── README.md             # This file
```

## 🎯 Model Selection

The project includes a comprehensive model selection system that allows you to:

1. **Compare Models**: View all trained models sorted by any metric
2. **Select Best Model**: Automatically load the best model based on your criteria
3. **Model Registry**: Use MLflow model registry for production deployments
4. **Metrics Comparison**: Compare models across multiple metrics simultaneously

See `src/select_best_model.py` for a complete example.

## 🌐 Streamlit Application

The Streamlit application provides a user-friendly interface for:

- **🔮 Prediction**: Make predictions on new data with real-time results
- **📊 Analysis**: View model performance metrics and visualizations
- **🔍 SHAP Explainability**: Understand why the model makes specific predictions
- **📈 Drift Monitoring**: Monitor data quality and detect distribution shifts
- **🧪 MLflow**: Browse and compare MLflow experiments
- **ℹ️ About**: Project documentation and information

## 🔧 Training Pipeline

The training pipeline (`src/run_train.py`) includes:

1. **Data Loading**: Loads preprocessed train/test splits
2. **Model Training**: Trains multiple models with hyperparameter tuning
3. **Evaluation**: Comprehensive evaluation on test set
4. **MLflow Logging**: Logs all metrics, parameters, and artifacts
5. **Model Saving**: Saves trained models for deployment

## 🧪 Testing

The project includes a comprehensive test suite using pytest. Tests are organized by module and cover:

- Configuration and constants
- Data preparation and feature engineering
- Model training and evaluation
- MLflow integration
- Model selection functionality
- Data I/O operations

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest test/test_features.py
```

Run tests by marker (e.g., only unit tests):
```bash
pytest -m unit
```

Skip slow tests:
```bash
pytest -m "not slow"
```

### Test Configuration

Test configuration is defined in `pytest.ini`:
- Test discovery patterns
- Markers for categorizing tests (unit, integration, slow)
- Output formatting options

## 📦 Requirements

Key dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting
- `mlflow` - Experiment tracking
- `streamlit` - Web application framework
- `shap` - Model explainability
- `evidently` - Data drift monitoring
- `plotly` - Interactive visualizations

See `requirements.txt` for the complete list with versions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## 👤 Author

**Jean-Michel Cheumeni**

- Email: cheumenijean@yahoo.fr

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the "Condition Monitoring of Hydraulic Systems" dataset
- MLflow team for the excellent experiment tracking framework
- SHAP contributors for model explainability tools
- Evidently AI for drift monitoring capabilities

---

**Note**: This project is designed for industrial predictive maintenance applications. Ensure proper validation and testing before deploying to production environments.
