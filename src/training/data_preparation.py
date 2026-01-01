import pandas as pd
import logging
from sklearn.utils import shuffle

from src.config.directories import directories as dirs
from src.read_write import get_data, save_dataset
from src.training.features import extract_features
from src.constants import (c_PROFILE_COLUMNS, 
                           c_DATA_PROFILE, 
                           c_DATA_PS2, 
                           c_DATA_FS1, 
                           c_DATA_PS2_FEATURES, 
                           c_DATA_FS1_FEATURES, 
                           c_TRAIN_CYCLES,
                           c_VALVE_CONDITION
                           )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def data_preparation():
    try:
        data = get_data()
        profile_df = data.get(c_DATA_PROFILE)
        profile_df.columns = c_PROFILE_COLUMNS
        profile_df['Target'] = (profile_df[c_VALVE_CONDITION] == 100).astype(int)
        ps2_features = extract_features(data.get(c_DATA_PS2), c_DATA_PS2_FEATURES, "PS2")
        fs1_features = extract_features(data.get(c_DATA_FS1), c_DATA_FS1_FEATURES, "FS1")
        data_df = pd.concat([profile_df, ps2_features, fs1_features], axis=1)
        return data_df
    except Exception as e:
        logger.error(f"Error in data_preparation: {e}")
        raise e from e


def data_preparation_train_test():
    try:
        data_df = data_preparation()
        # data_df = shuffle(data_df)
        train_cycles = c_TRAIN_CYCLES
        X = data_df.drop(columns=c_PROFILE_COLUMNS + ['Target'])

        y = data_df['Target']
        X_train = X.iloc[:train_cycles]
        X_test = X.iloc[train_cycles:]
        y_train = y.iloc[:train_cycles]
        y_test = y.iloc[train_cycles:]
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of X_test: {X_test.shape}")
        logger.info(f"Shape of y_train: {y_train.shape}")
        logger.info(f"Shape of y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in data_preparation_train_test: {e}")
        raise e from e


def save_data_train_test():
    try:

        X_train, X_test, y_train, y_test = data_preparation_train_test()

        save_dataset(X_train, path=dirs.intermediate / 'X_train.csv')
        save_dataset(X_test, path=dirs.intermediate / 'X_test.csv')
        save_dataset(y_train, path=dirs.intermediate / 'y_train.csv')
        save_dataset(y_test, path=dirs.intermediate / 'y_test.csv')
        logger.info("Data train and test saved. ✅")
    except Exception as e:
        logger.error(f"Error in save_data_train_test: {e}")
        raise e from e
