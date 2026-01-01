import logging
import joblib  # or pickle.
import json

import pandas as pd

from src.constants import (
                           c_PROFILE,
                           c_PS2,
                           c_FS1,
                           c_DATA_PROFILE,
                           c_DATA_PS2,
                           c_DATA_FS1
                          )
from src.config.directories import directories as dirs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def get_data():
    try:    
        logger.info("Loading data profile, ps2 and fs1")
        data_profile = load_dataset(str(dirs.inputs / c_PROFILE))
        data_ps2 = load_dataset(str(dirs.inputs / c_PS2))
        data_fs1 = load_dataset(str(dirs.inputs / c_FS1))
        logger.info("Data catalog loaded. ✅")
    except Exception as e:
        logger.error(f"Error in get_data: {e}")
        raise e from e 

    return {c_DATA_PROFILE: data_profile,
            c_DATA_PS2: data_ps2,
            c_DATA_FS1: data_fs1,
    }   
  
  
def save_dataset(dataset, *, path):
    try:
        dataset.to_csv(path, index=False)
        logger.info(f"Dataset saved at {path.relative_to(dirs.root_dir)}")
    except Exception as e:
        logger.error(f"Error in save_dataset: {e}")
        raise e from e


def load_data_train_test():
    try:
        logger.info("Loading data train and test...")
        # Load CSV files with comma separator and headers (default pandas behavior)
        X_train = pd.read_csv(str(dirs.intermediate / 'X_train.csv'), encoding='utf-8', encoding_errors='replace')
        X_test = pd.read_csv(str(dirs.intermediate / 'X_test.csv'), encoding='utf-8', encoding_errors='replace')
        y_train_df = pd.read_csv(str(dirs.intermediate / 'y_train.csv'), encoding='utf-8', encoding_errors='replace')
        y_test_df = pd.read_csv(str(dirs.intermediate / 'y_test.csv'), encoding='utf-8', encoding_errors='replace')
        
        # Reset indices to ensure alignment
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train_df = y_train_df.reset_index(drop=True)
        y_test_df = y_test_df.reset_index(drop=True)
        
        # Extract Series from single-column DataFrames
        y_train = y_train_df.iloc[:, 0] if len(y_train_df.columns) == 1 else y_train_df['Target']
        y_test = y_test_df.iloc[:, 0] if len(y_test_df.columns) == 1 else y_test_df['Target']
        
        # Ensure y_train and y_test are Series with proper indices
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]
        
        # Reset indices to ensure they match X_train and X_test
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Validate shapes match
        if len(X_train) != len(y_train):
            raise ValueError(f"Mismatch: X_train has {len(X_train)} rows, y_train has {len(y_train)} rows")
        if len(X_test) != len(y_test):
            raise ValueError(f"Mismatch: X_test has {len(X_test)} rows, y_test has {len(y_test)} rows")
        
        logger.info("Data train and test loaded. ✅")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in load_data_train_test: {e}")
        raise e from e 


def load_dataset(path):
    try:
        return pd.read_csv(path, sep='\t', header=None, encoding='utf-8', encoding_errors='replace')
    except Exception as e:
        logger.error(f"Error in load_dataset: {e}")
        raise e from e


def save_model(model, *, path):
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved at {path}")
    except Exception as e:
        logger.error(f"Error in save_model: {e}")
        raise e from e


def save_metrics(metrics, *, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved at {path}")
    except Exception as e:
        logger.error(f"Error in save_metrics: {e}")
        raise e from e
        

def save_lexique(lexique, *, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(lexique, f, indent=2)
        logger.info(f"Lexique saved at {path}")
    except Exception as e:
        logger.error(f"Error in save_lexique: {e}")
        raise e from e


def save_metrics_per_class(metrics, *, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics per class saved at {path}")
    except Exception as e:
        logger.error(f"Error in save_metrics_per_class: {e}")
        raise e from e


def load_model(path, default=None):
    try:
        return joblib.load(path)
    except ValueError:
        if default is None:
             default = {}
        logger.info(f"Model not found at {path}, using default: {default}")
        return default
    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        raise e from e
    

def load_json_file(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception as e:
        logger.error(f"Error in load_json_file: {e}")
        raise e from e
    except ValueError:
        if default is None:
             default = {}
        return default


def save_prediction(predictions,*, path):
    #https://docs.python.org/3/library/json.html
    try:
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(predictions, fp, indent=2)
    except Exception as e:
        logger.error(f"Error in save_prediction: {e}")
        raise e from e