"""
F1 Feature Engineering Module.

Provides feature preparation and encoding for machine learning models.
Handles categorical encoding for Driver, Team, and Track features.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configure module logger
logger = logging.getLogger('F1.Features')


def prepare_features(
    df: pd.DataFrame, 
    train_mode: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, LabelEncoder]]:
    """
    Prepare features for ML model training or prediction with error handling.
    
    Features used:
    - Starting Grid (numeric)
    - Driver (encoded)
    - Team (encoded)
    - Track (encoded)
    
    In training mode:
    - Filters to only finished races
    - Fits and saves label encoders
    - Returns target variable (Position)
    
    In prediction mode:
    - Loads saved encoders
    - Returns None for target
    
    Args:
        df: Input DataFrame with race data.
        train_mode: If True, fit encoders and return target.
                   If False, load encoders for prediction.
        
    Returns:
        Tuple of (X features, y target, encoders dict).
        y is None if train_mode is False.
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> X, y, encoders = prepare_features(df_train, train_mode=True)
        >>> print(f"Features shape: {X.shape}")
    """
    logger.info(f"Preparing features (train_mode={train_mode})...")
    
    try:
        # Validate input
        if df is None or df.empty:
            logger.error("Cannot prepare features from empty DataFrame")
            raise ValueError("DataFrame is None or empty")
        
        # Filter to finished races in training mode
        if train_mode:
            if 'Finished' not in df.columns:
                logger.error("'Finished' column required for training mode")
                raise ValueError("Missing 'Finished' column")
            df = df[df['Finished'] == True].copy()
            logger.info(f"Filtered to {len(df)} finished races")
            if df.empty:
                logger.error("No finished races found in data")
                raise ValueError("No finished races in data")
        
        # Define features and target
        features = ['Starting Grid', 'Driver', 'Team', 'Track']
        target = 'Position'
        
        # Validate required columns exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing feature columns: {missing_features}")
            raise ValueError(f"Missing columns: {missing_features}")
        
        if train_mode and target not in df.columns:
            logger.error(f"Target column '{target}' not found")
            raise ValueError(f"Missing target column: {target}")
        
        X = df[features].copy()
        y = df[target] if train_mode and target in df.columns else None
        
        # Ensure models directory exists
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created directory: {model_dir}")
        
        # Encode categorical columns
        encoders: Dict[str, LabelEncoder] = {}
        categorical_cols = ['Driver', 'Team', 'Track']
        
        for col in categorical_cols:
            le = LabelEncoder()
            encoder_path = os.path.join(model_dir, f'{col}_encoder.pkl')
            
            if train_mode:
                # Fit encoder and save
                try:
                    X[col] = le.fit_transform(X[col])
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(le, f)
                    encoders[col] = le
                    logger.debug(f"Fitted and saved encoder for {col} ({len(le.classes_)} classes)")
                except Exception as e:
                    logger.error(f"Error encoding {col}: {e}")
                    raise
            else:
                # Load existing encoder
                try:
                    with open(encoder_path, 'rb') as f:
                        le = pickle.load(f)
                    
                    # Handle unknown categories gracefully
                    unknown_values = set(X[col].unique()) - set(le.classes_)
                    if unknown_values:
                        logger.warning(f"Unknown values in {col}: {unknown_values}")
                    
                    X[col] = X[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    encoders[col] = le
                    logger.debug(f"Loaded encoder for {col} ({len(le.classes_)} classes)")
                except FileNotFoundError:
                    logger.error(f"Encoder for {col} not found at {encoder_path}")
                    logger.error("Please train the model first using train_mode=True")
                    raise
                except Exception as e:
                    logger.error(f"Error loading encoder for {col}: {e}")
                    raise
        
        logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        if y is not None:
            logger.debug(f"Target range: {y.min()}-{y.max()}")
        
        return X, y, encoders
        
    except Exception as e:
        logger.exception(f"Error preparing features: {e}")
        raise
