from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class _Directories:

    def __init__(self) -> None:

        self.root_dir = Path(__file__).resolve(strict=True).parents[2]
        
        self.data_dir = self.root_dir / "data"
        self.inputs = self.data_dir / "raw"
        self.intermediate = self.data_dir / "intermediate"
        self.config = self.root_dir / "config"
        self.dir = self.root_dir / "src"
        self.dir_storage = self.dir / "storage"
        self.raw_store_dir = self.dir_storage / 'mlflow_artifacts'
        self.test_dir = self.root_dir / "test"


        for dir_path in vars(self).values():
            try:
                dir_path.mkdir(exist_ok=True, parents=True)
            except:
                logger.info("Error when we are build a {} directory".format(dir_path))

directories = _Directories()