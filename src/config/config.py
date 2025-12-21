import logging
import yaml

from src.constants import c_PROD 
from src.config.directories import directories as dirs

logger = logging.getLogger(__name__)


def load_param(file_name: str) -> dict:
    """Load parameters from a YAML file."""

    try:
        with open(file_name, 'r', encoding='utf-8') as fp:
            read_data = yaml.safe_load(fp)
    except Exception as e:
        logger.error(f"Error in load_param: {e} - File: {file_name}")
        raise e from e
    return read_data


def load_config_file() -> dict:
    """Load configuration file."""
    try:
        read_data = load_param(str(dirs.config / c_PROD))
        return read_data
    except Exception as e:
        logger.error(f"Error in load_config_file: {e} - File: {str(dirs.config / c_PROD)}")
        raise e from e
    


def get_config() -> dict:
    """Get configuration file."""
    try:
        return load_config_file()
    except Exception as e:
        logger.error(f"Error in get_config: {e}")
        raise e from e