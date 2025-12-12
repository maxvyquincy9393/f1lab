# -*- coding: utf-8 -*-
"""
test_loader.py
~~~~~~~~~~~~~~
Unit tests for the data loader module.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_data, clean_data


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_valid_file(self):
        """Test loading a valid CSV file."""
        data_path = Path(__file__).parent.parent / 'data' / 'Formula1_2025Season_RaceResults.csv'
        if data_path.exists():
            df = load_data(str(data_path))
            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_load_data_invalid_file(self):
        """Test loading a non-existent file returns None."""
        df = load_data('non_existent_file.csv')
        assert df is None


class TestCleanData:
    """Tests for clean_data function."""
    
    def test_clean_data_creates_finished_column(self):
        """Test that clean_data creates a Finished column."""
        # Create sample data
        sample_data = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Points': ['25', '18'],
            'Position': ['1', '2'],
            'Starting Grid': ['1', '3'],
            'Laps': ['57', '57'],
            'Time/Retired': ['+0.000', '+5.123']
        })
        
        cleaned = clean_data(sample_data)
        
        assert 'Finished' in cleaned.columns
        assert cleaned['Points'].dtype in ['int64', 'float64']
        assert cleaned['Position'].dtype in ['int64', 'float64']
    
    def test_clean_data_handles_dnf(self):
        """Test that clean_data handles DNF entries."""
        sample_data = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Points': ['0', '0'],
            'Position': [None, '20'],
            'Starting Grid': ['1', '3'],
            'Laps': ['10', '57'],
            'Time/Retired': ['DNS', '+1 Lap']
        })
        
        cleaned = clean_data(sample_data)
        
        # DNS should not be marked as finished
        assert cleaned.loc[0, 'Finished'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
