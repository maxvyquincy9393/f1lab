"""
Tests for the model module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import train_model, load_trained_model


@pytest.fixture
def sample_race_data():
    """Create sample race data for model training."""
    np.random.seed(42)
    n_samples = 100
    
    drivers = ['NOR', 'VER', 'PIA', 'RUS', 'LEC', 'HAM']
    teams = ['McLaren', 'Red Bull', 'Mercedes', 'Ferrari']
    tracks = ['Australia', 'Bahrain', 'Monaco', 'Silverstone']
    
    data = {
        'Driver': np.random.choice(drivers, n_samples),
        'Team': np.random.choice(teams, n_samples),
        'Track': np.random.choice(tracks, n_samples),
        'Starting Grid': np.random.randint(1, 21, n_samples),
        'Position': np.random.randint(1, 21, n_samples),
        'Finished': [True] * n_samples
    }
    
    return pd.DataFrame(data)


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


class TestTrainModel:
    """Tests for train_model function."""
    
    def test_returns_model_and_test_data(self, sample_race_data, temp_model_dir):
        """Test that train_model returns model and test data."""
        model, X_test, y_test = train_model(sample_race_data)
        
        assert model is not None
        assert len(X_test) > 0
        assert len(y_test) > 0
    
    def test_saves_model_file(self, sample_race_data, temp_model_dir):
        """Test that model is saved to file."""
        model, X_test, y_test = train_model(sample_race_data)
        
        model_path = temp_model_dir / 'models' / 'f1_model.pkl'
        assert model_path.exists()
    
    def test_model_can_predict(self, sample_race_data, temp_model_dir):
        """Test that trained model can make predictions."""
        model, X_test, y_test = train_model(sample_race_data)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_test_size_is_correct(self, sample_race_data, temp_model_dir):
        """Test that test split is approximately 20%."""
        model, X_test, y_test = train_model(sample_race_data)
        
        # Should be about 20% of data (allowing for rounding)
        expected_test_size = len(sample_race_data) * 0.2
        assert abs(len(X_test) - expected_test_size) <= 2


class TestLoadTrainedModel:
    """Tests for load_trained_model function."""
    
    def test_returns_none_if_no_model(self, temp_model_dir):
        """Test that None is returned if no model exists."""
        model = load_trained_model()
        
        assert model is None
    
    def test_loads_saved_model(self, sample_race_data, temp_model_dir):
        """Test that saved model can be loaded."""
        # First train and save
        trained_model, _, _ = train_model(sample_race_data)
        
        # Then load
        loaded_model = load_trained_model()
        
        assert loaded_model is not None
    
    def test_loaded_model_can_predict(self, sample_race_data, temp_model_dir):
        """Test that loaded model can make predictions."""
        # Train and save
        trained_model, X_test, _ = train_model(sample_race_data)
        
        # Load and predict
        loaded_model = load_trained_model()
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == len(X_test)


class TestModelPerformance:
    """Tests for model performance characteristics."""
    
    def test_predictions_in_valid_range(self, sample_race_data, temp_model_dir):
        """Test that predictions are within reasonable position range."""
        model, X_test, y_test = train_model(sample_race_data)
        predictions = model.predict(X_test)
        
        # Positions should be roughly between 1 and 20
        assert all(0 < p < 25 for p in predictions)
    
    def test_feature_importances_exist(self, sample_race_data, temp_model_dir):
        """Test that model has feature importances."""
        model, _, _ = train_model(sample_race_data)
        
        importances = model.feature_importances_
        
        assert len(importances) > 0
        assert sum(importances) == pytest.approx(1.0, abs=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
