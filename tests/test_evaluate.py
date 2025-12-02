"""
Tests for the evaluate module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate_model


@pytest.fixture
def trained_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = np.random.randint(1, 21, 100)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model


@pytest.fixture
def test_data():
    """Create test data for evaluation."""
    np.random.seed(42)
    X_test = pd.DataFrame(
        np.random.rand(20, 4),
        columns=['Starting Grid', 'Driver', 'Team', 'Track']
    )
    y_test = pd.Series(np.random.randint(1, 21, 20))
    
    return X_test, y_test


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_returns_metrics_dict(self, trained_model, test_data):
        """Test that metrics dictionary is returned."""
        X_test, y_test = test_data
        metrics, results_df = evaluate_model(trained_model, X_test, y_test)
        
        assert isinstance(metrics, dict)
    
    def test_metrics_contains_mae(self, trained_model, test_data):
        """Test that MAE is in metrics."""
        X_test, y_test = test_data
        metrics, _ = evaluate_model(trained_model, X_test, y_test)
        
        assert 'MAE' in metrics
        assert isinstance(metrics['MAE'], float)
        assert metrics['MAE'] >= 0
    
    def test_metrics_contains_mse(self, trained_model, test_data):
        """Test that MSE is in metrics."""
        X_test, y_test = test_data
        metrics, _ = evaluate_model(trained_model, X_test, y_test)
        
        assert 'MSE' in metrics
        assert isinstance(metrics['MSE'], float)
        assert metrics['MSE'] >= 0
    
    def test_metrics_contains_r2(self, trained_model, test_data):
        """Test that R2 is in metrics."""
        X_test, y_test = test_data
        metrics, _ = evaluate_model(trained_model, X_test, y_test)
        
        assert 'R2' in metrics
        assert isinstance(metrics['R2'], float)
    
    def test_returns_results_dataframe(self, trained_model, test_data):
        """Test that results DataFrame is returned."""
        X_test, y_test = test_data
        metrics, results_df = evaluate_model(trained_model, X_test, y_test)
        
        assert isinstance(results_df, pd.DataFrame)
        assert 'Actual' in results_df.columns
        assert 'Predicted' in results_df.columns
    
    def test_results_df_has_correct_length(self, trained_model, test_data):
        """Test that results DataFrame has correct number of rows."""
        X_test, y_test = test_data
        metrics, results_df = evaluate_model(trained_model, X_test, y_test)
        
        assert len(results_df) == len(y_test)
    
    def test_actual_values_match_y_test(self, trained_model, test_data):
        """Test that Actual column matches y_test."""
        X_test, y_test = test_data
        metrics, results_df = evaluate_model(trained_model, X_test, y_test)
        
        np.testing.assert_array_equal(results_df['Actual'].values, y_test.values)


class TestMetricsCalculations:
    """Tests for correct metric calculations."""
    
    def test_perfect_predictions_have_zero_error(self):
        """Test that perfect predictions result in zero MAE/MSE."""
        # Create model that predicts perfectly (by fitting on same data)
        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([1, 2, 3, 4, 5])
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        metrics, _ = evaluate_model(model, X, y)
        
        # Should have very low error (near zero)
        assert metrics['MAE'] < 0.5
    
    def test_r2_between_minus_inf_and_one(self, trained_model, test_data):
        """Test that R2 is in valid range."""
        X_test, y_test = test_data
        metrics, _ = evaluate_model(trained_model, X_test, y_test)
        
        assert metrics['R2'] <= 1.0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_sample(self, trained_model):
        """Test evaluation with single sample."""
        X_test = pd.DataFrame({'a': [0.5], 'b': [0.5], 'c': [0.5], 'd': [0.5]})
        X_test.columns = ['Starting Grid', 'Driver', 'Team', 'Track']
        y_test = pd.Series([10])
        
        metrics, results_df = evaluate_model(trained_model, X_test, y_test)
        
        assert len(results_df) == 1
        assert 'MAE' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
