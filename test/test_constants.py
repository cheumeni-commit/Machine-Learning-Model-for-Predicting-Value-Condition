"""
Unit tests for constants module.
"""
from src import constants


class TestConstants:
    """Test cases for constants module."""
    
    def test_profile_constant(self):
        """Test that profile constant is defined."""
        assert hasattr(constants, 'c_PROFILE')
        assert constants.c_PROFILE == 'profile.txt'
    
    def test_ps2_constant(self):
        """Test that PS2 constant is defined."""
        assert hasattr(constants, 'c_PS2')
        assert constants.c_PS2 == 'PS2.txt'
    
    def test_fs1_constant(self):
        """Test that FS1 constant is defined."""
        assert hasattr(constants, 'c_FS1')
        assert constants.c_FS1 == 'FS1.txt'
    
    def test_data_keys(self):
        """Test that data keys are defined."""
        assert hasattr(constants, 'c_DATA_PROFILE')
        assert hasattr(constants, 'c_DATA_PS2')
        assert hasattr(constants, 'c_DATA_FS1')
        assert hasattr(constants, 'c_DATA_PS2_FEATURES')
        assert hasattr(constants, 'c_DATA_FS1_FEATURES')
    
    def test_profile_columns(self):
        """Test that profile columns are defined correctly."""
        assert hasattr(constants, 'c_PROFILE_COLUMNS')
        expected_columns = ['Cooler_Condition', 'Valve_Condition', 
                          'Internal_Pump_Leakage', 'Hydraulic_Accumulator', 'Stable_Flag']
        assert constants.c_PROFILE_COLUMNS == expected_columns
    
    def test_classes(self):
        """Test that classes are defined."""
        assert hasattr(constants, 'c_CLASSES')
        assert len(constants.c_CLASSES) == 2
        assert 'Non-Optimal (0)' in constants.c_CLASSES
        assert 'Optimal (1)' in constants.c_CLASSES
    
    def test_train_cycles(self):
        """Test that train cycles constant is defined."""
        assert hasattr(constants, 'c_TRAIN_CYCLES')
        assert constants.c_TRAIN_CYCLES == 2000
        assert isinstance(constants.c_TRAIN_CYCLES, int)
    
    def test_prod_constant(self):
        """Test that prod constant is defined."""
        assert hasattr(constants, 'c_PROD')
        assert constants.c_PROD == 'prod.yml'
