import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def extract_features(data, sensor_name):
    """Charge un fichier de capteur et calcule des caractéristiques statistiques par cycle."""
    try:
        
        # Calculer les caractéristiques statistiques pour chaque ligne (cycle)
        features = {
            f'{sensor_name}_mean': data.mean(axis=1),
            f'{sensor_name}_std': data.std(axis=1),
            f'{sensor_name}_min': data.min(axis=1),
            f'{sensor_name}_max': data.max(axis=1),
            f'{sensor_name}_median': data.median(axis=1),
            f'{sensor_name}_q25': data.quantile(0.25, axis=1),
            f'{sensor_name}_q75': data.quantile(0.75, axis=1),
            f'{sensor_name}_range': data.max(axis=1) - data.min(axis=1),
            f'{sensor_name}_rms': np.sqrt(np.mean(data**2, axis=1)) # Root Mean Square
        }
        
        features_df = pd.DataFrame(features)
        logger.info(f"Shape of {sensor_name} features: {features_df.shape}")
        return features_df
    except Exception as e:
        logger.error(f"Error in extract_features: {e}")
        raise e from e