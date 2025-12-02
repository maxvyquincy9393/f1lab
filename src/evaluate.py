"""
F1 Model Evaluation Module.

Provides functions for evaluating ML model performance with
various metrics and visualization capabilities.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure module logger
logger = logging.getLogger('F1.Evaluate')


def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate model performance with multiple metrics.
    
    Computes the following metrics:
    - MAE (Mean Absolute Error): Average absolute difference
    - MSE (Mean Squared Error): Average squared difference
    - RMSE (Root Mean Squared Error): Square root of MSE
    - R2 (R-squared): Coefficient of determination
    
    Args:
        model: Trained model with predict method.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        
    Returns:
        Tuple of (metrics dict, results DataFrame with Actual/Predicted).
        
    Example:
        >>> metrics, results = evaluate_model(model, X_test, y_test)
        >>> print(f"MAE: {metrics['MAE']:.2f}")
        >>> print(f"R2: {metrics['R2']:.3f}")
    """
    logger.info("Evaluating model performance...")
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions,
        'Error': y_test.values - predictions,
        'Abs_Error': np.abs(y_test.values - predictions)
    })
    
    # Log metrics
    logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    logger.info(f"Average absolute error: {results_df['Abs_Error'].mean():.2f} positions")
    
    return metrics, results_df


def plot_predictions(
    results_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot actual vs predicted positions.
    
    Creates a scatter plot with ideal line for visual evaluation.
    
    Args:
        results_df: DataFrame with Actual and Predicted columns.
        save_path: Optional path to save the figure.
        
    Returns:
        matplotlib Figure object.
    """
    logger.info("Creating prediction plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(
        results_df['Actual'], 
        results_df['Predicted'],
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Ideal line (y = x)
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    ax.set_xlabel('Actual Position', fontsize=12)
    ax.set_ylabel('Predicted Position', fontsize=12)
    ax.set_title('F1 Position Prediction: Actual vs Predicted', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_error_distribution(
    results_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Plot the distribution of prediction errors.
    
    Args:
        results_df: DataFrame with Error column.
        save_path: Optional path to save the figure.
        
    Returns:
        matplotlib Figure object.
    """
    logger.info("Creating error distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(results_df['Error'], bins=20, kde=True, ax=ax, color='steelblue')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Prediction Errors', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig
