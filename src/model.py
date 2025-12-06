"""
F1 Machine Learning Model Module.

Provides model training and loading functions for race position prediction.
Uses Random Forest Regressor as the base model.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
import os
import pickle
from typing import Optional, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from src.features import prepare_features
except ImportError:
    from features import prepare_features

# Configure module logger
logger = logging.getLogger('F1.Model')

# Model configuration
MODEL_PATH = 'models/f1_model.pkl'
N_ESTIMATORS = 100 # Represents max_iter for GBM
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_model(
    df: pd.DataFrame
) -> Tuple[HistGradientBoostingRegressor, pd.DataFrame, pd.Series]:
    """
    Train a Gradient Boosting model (SOTA for Tabular Data).
    
    Uses HistGradientBoostingRegressor for O(n) efficiency and native NaN handling.
    """
    logger.info("Starting model training...")
    
    # Prepare features
    logger.info("Preparing features...")
    X, y, _ = prepare_features(df, train_mode=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    logger.info(f"Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=N_ESTIMATORS,
        learning_rate=0.1,
        max_depth=10,
        random_state=RANDOM_STATE,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Log feature importances
    feature_names = X.columns.tolist()
    importances = dict(zip(feature_names, model.feature_importances_))
    logger.info(f"Feature importances: {importances}")
    
    return model, X_test, y_test


def load_trained_model() -> Optional[HistGradientBoostingRegressor]:
    """Load model from disk (Legacy support)."""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None


class RaceStrategySimulator:
    """
    God Mode Simulation Engine.
    Simulates race pace using physics-based decay curves (Tyre Deg + Fuel Burn).
    """
    
    def __init__(self, base_lap_time: float = 90.0, total_laps: int = 57):
        self.base_lap_time = base_lap_time
        self.total_laps = total_laps
        self.pit_loss = 22.0 # Avg pit loss in seconds
        
    def predict_strategy(self, driver: str, start_tire: str, current_lap: int = 0) -> dict:
        """
        Predict optimal strategy and finish time.
        """
        # Try 1-stop
        t_1stop = self._simulate_stint(start_tire, current_lap, self.total_laps)
        
        # Try 2-stop (simplified split)
        stop_lap = current_lap + (self.total_laps - current_lap) // 2
        t_2stop_s1 = self._simulate_stint(start_tire, current_lap, stop_lap)
        t_2stop_s2 = self._simulate_stint('MEDIUM', stop_lap, self.total_laps) # Assume switch to Med
        t_2stop = t_2stop_s1 + t_2stop_s2 + self.pit_loss
        
        return {
            '1_stop_time': t_1stop,
            '2_stop_time': t_2stop,
            'recommended': '1 Stop' if t_1stop < t_2stop else '2 Stops',
            'delta': abs(t_1stop - t_2stop)
        }

    def _simulate_stint(self, compound: str, start_lap: int, end_lap: int) -> float:
        """Simulate total time for a stint."""
        from features import calculate_degradation_curve, calculate_fuel_correction
        
        total_time = 0.0
        deg_factor = calculate_degradation_curve(compound)
        
        for lap in range(start_lap, end_lap):
            # Physics: Base + Deg - Fuel
            tyre_age = lap - start_lap
            deg_penalty = tyre_age * deg_factor
            fuel_gain = calculate_fuel_correction(lap, self.total_laps)
            
            lap_time = self.base_lap_time + deg_penalty + fuel_gain
            total_time += lap_time
            
        return total_time

    def catch_up_prediction(self, chaser_gap: float, chaser_tire: str, leader_tire: str, laps_remaining: int) -> int:
        """
        Predict lap when chaser catches leader.
        Returns lap number or -1 if never.
        """
        from features import calculate_degradation_curve
        
        deg_chaser = calculate_degradation_curve(chaser_tire)
        deg_leader = calculate_degradation_curve(leader_tire)
        
        current_gap = chaser_gap
        
        for i in range(laps_remaining):
            # Relative pace (neg = chaser faster)
            # Simplified: Assume chaser has fresher tires (-0.5s avg advantage)
            pace_delta = -0.5 + (deg_chaser * i) - (deg_leader * i)
            current_gap += pace_delta
            
            if current_gap <= 0:
                return i # Laps to catch
                
        return -1
