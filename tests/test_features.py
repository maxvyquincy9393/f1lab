# -*- coding: utf-8 -*-
"""
test_features.py
~~~~~~~~~~~~~~~~
Unit tests for the feature engineering module.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import prepare_features


@pytest.fixture
def sample_race_data():
    """Create sample race data for testing."""
    return pd.DataFrame({
        'Driver': ['Driver A', 'Driver B', 'Driver C', 'Driver A', 'Driver B', 'Driver C'],
        'Team': ['Team X', 'Team Y', 'Team Z', 'Team X', 'Team Y', 'Team Z'],
        'Track': ['Race 1', 'Race 1', 'Race 1', 'Race 2', 'Race 2', 'Race 2'],
        'Position': [1, 2, 3, 2, 1, 3],
        'Starting Grid': [1, 2, 3, 3, 1, 2],
        'Finished': [True, True, True, True, True, False]
    })


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / 'models'
    model_dir.mkdir()
    
    # Change to temp directory for tests
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Restore original directory
    os.chdir(original_dir)


class TestPrepareFeatures:
    """Tests for prepare_features function."""
    
    def test_returns_correct_columns(self, sample_race_data, temp_model_dir):
        """Test that correct feature columns are returned."""
        X, y, encoders = prepare_features(sample_race_data, train_mode=True)
        
        expected_cols = ['Starting Grid', 'Driver', 'Team', 'Track']
        assert list(X.columns) == expected_cols
    
    def test_filters_finished_races_in_train_mode(self, sample_race_data, temp_model_dir):
        """Test that unfinished races are filtered in train mode."""
        X, y, encoders = prepare_features(sample_race_data, train_mode=True)
        
        # One row has Finished=False, so should be 5 rows
        assert len(X) == 5
    
    def test_returns_target_in_train_mode(self, sample_race_data, temp_model_dir):
        """Test that target variable is returned in train mode."""
        X, y, encoders = prepare_features(sample_race_data, train_mode=True)
        
        assert y is not None
        assert len(y) == 5
    
    def test_encodes_categorical_columns(self, sample_race_data, temp_model_dir):
        """Test that categorical columns are label encoded."""
        X, y, encoders = prepare_features(sample_race_data, train_mode=True)
        
        # After encoding, columns should be numeric
        assert X['Driver'].dtype in [np.int32, np.int64]
        assert X['Team'].dtype in [np.int32, np.int64]
        assert X['Track'].dtype in [np.int32, np.int64]
    
    def test_creates_encoders(self, sample_race_data, temp_model_dir):
        """Test that encoders are created and returned."""
        X, y, encoders = prepare_features(sample_race_data, train_mode=True)
        
        assert 'Driver' in encoders
        assert 'Team' in encoders
        assert 'Track' in encoders
    
    def test_saves_encoders_to_file(self, sample_race_data, temp_model_dir):
        """Test that encoder files are saved."""
        X, y, encoders = prepare_features(sample_race_data, train_mode=True)
        
        model_dir = temp_model_dir / 'models'
        assert (model_dir / 'Driver_encoder.pkl').exists()
        assert (model_dir / 'Team_encoder.pkl').exists()
        assert (model_dir / 'Track_encoder.pkl').exists()
    
    def test_inference_mode_uses_saved_encoders(self, sample_race_data, temp_model_dir):
        """Test that inference mode loads saved encoders."""
        # First train to create encoders
        X_train, _, _ = prepare_features(sample_race_data, train_mode=True)
        
        # Then inference with new data
        new_data = pd.DataFrame({
            'Driver': ['Driver A'],
            'Team': ['Team X'],
            'Track': ['Race 1'],
            'Starting Grid': [1],
            'Finished': [True]
        })
        
        X_test, y_test, encoders = prepare_features(new_data, train_mode=False)
        
        assert len(X_test) == 1
        assert 'Driver' in encoders


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, temp_model_dir):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame(columns=['Driver', 'Team', 'Track', 'Position', 'Starting Grid', 'Finished'])
        
        X, y, encoders = prepare_features(empty_df, train_mode=True)
        
        assert len(X) == 0
    
    def test_all_finished_false(self, temp_model_dir):
        """Test when all races are DNF."""
        dnf_df = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Team': ['Team X', 'Team Y'],
            'Track': ['Race 1', 'Race 1'],
            'Position': [1, 2],
            'Starting Grid': [1, 2],
            'Finished': [False, False]
        })
        
        X, y, encoders = prepare_features(dnf_df, train_mode=True)
        
        assert len(X) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
