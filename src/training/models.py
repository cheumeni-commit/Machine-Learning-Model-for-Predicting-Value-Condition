import logging
from typing import List

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC

from src.config.config import get_config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Models registry
_MODELS_REGISTRY_ = {'RandomForestClassifier': RandomForestClassifier,
                     'SGDClassifier': SGDClassifier,
                     'SVC': SVC,
                     "LogisticRegressor" : LogisticRegression,
                     'xgb_Classifier' : xgb.XGBClassifier
                     }


def _load_models()-> List:
    models = []
    params = []

    for k, v in _MODELS_REGISTRY_.items():
        models.append(v)
        params.append(get_config().get(k).get('params'))
    return models, params


def get_models():
    """ Load Models """
    try:
        models, params = _load_models()
    except Exception as e:
        logger.error(f"Error in get_model: {e}")
        raise e from e
        logger.info("The models are not available ")

    return models, params