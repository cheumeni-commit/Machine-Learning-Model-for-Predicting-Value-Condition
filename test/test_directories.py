"""
Unit tests for directories module.
"""
from pathlib import Path
from src.config.directories import directories, _Directories


class TestDirectories:
    """Test cases for directories module."""
    
    def test_directories_instance(self):
        """Test that directories is an instance of _Directories."""
        assert isinstance(directories, _Directories)
    
    def test_root_dir_exists(self):
        """Test that root_dir is defined and exists."""
        assert hasattr(directories, 'root_dir')
        assert isinstance(directories.root_dir, Path)
        # The root_dir should be the project root (2 levels up from this file)
        assert directories.root_dir.exists()
    
    def test_data_dir(self):
        """Test that data_dir is defined correctly."""
        assert hasattr(directories, 'data_dir')
        assert directories.data_dir == directories.root_dir / "data"
    
    def test_inputs_dir(self):
        """Test that inputs directory is defined correctly."""
        assert hasattr(directories, 'inputs')
        assert directories.inputs == directories.data_dir / "raw"
    
    def test_intermediate_dir(self):
        """Test that intermediate directory is defined correctly."""
        assert hasattr(directories, 'intermediate')
        assert directories.intermediate == directories.data_dir / "intermediate"
    
    def test_config_dir(self):
        """Test that config directory is defined correctly."""
        assert hasattr(directories, 'config')
        assert directories.config == directories.root_dir / "config"
    
    def test_dir_storage(self):
        """Test that storage directory is defined correctly."""
        assert hasattr(directories, 'dir_storage')
        assert directories.dir_storage == directories.dir / "storage"
    
    def test_raw_store_dir(self):
        """Test that raw_store_dir is defined correctly."""
        assert hasattr(directories, 'raw_store_dir')
        assert directories.raw_store_dir == directories.dir_storage / 'mlflow_artifacts'
    
    def test_test_dir(self):
        """Test that test_dir is defined correctly."""
        assert hasattr(directories, 'test_dir')
        assert directories.test_dir == directories.root_dir / "test"
    
    def test_directories_are_paths(self):
        """Test that all directory attributes are Path objects."""
        for attr_name in ['root_dir', 'data_dir', 'inputs', 'intermediate', 
                         'config', 'dir', 'dir_storage', 'raw_store_dir', 'test_dir']:
            attr = getattr(directories, attr_name)
            assert isinstance(attr, Path), f"{attr_name} should be a Path object"
