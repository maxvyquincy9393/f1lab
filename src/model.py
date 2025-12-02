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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

try:
    from src.features import prepare_features
except ImportError:
    from features import prepare_features

# Configure module logger
logger = logging.getLogger('F1.Model')

# Model configuration
MODEL_PATH = 'models/f1_model.pkl'
N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_model(
    df: pd.DataFrame
) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.Series]:
    """
    Train a Random Forest model for race position prediction.
    
    The model is trained on historical race data to predict
    finishing position based on starting grid, driver, team, and track.
    
    Args:
        df: Cleaned race DataFrame with Position, Starting Grid,
            Driver, Team, Track, and Finished columns.
        
    Returns:
        Tuple of (trained model, X_test, y_test) for evaluation.
        
    Example:
        >>> model, X_test, y_test = train_model(df_race)
        >>> predictions = model.predict(X_test)
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
    logger.info(f"Training RandomForestRegressor (n_estimators={N_ESTIMATORS})...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, 
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all CPU cores
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


def load_trained_model() -> Optional[RandomForestRegressor]:
    """
    Load a previously trained model from disk.
    
    Returns:
        RandomForestRegressor or None: Trained model or None if not found.
        
    Example:
        >>> model = load_trained_model()
        >>> if model is not None:
        ...     prediction = model.predict(X_new)
    """
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.warning(f"Model not found at {MODEL_PATH}")
        return None
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        return None
