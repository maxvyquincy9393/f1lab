"""
F1 Visualization Package.

Provides data loading, analysis, and visualization for F1 racing data.
"""

try:
    from .loader import load_data, clean_data
    from .analysis import (
        calculate_driver_stats, calculate_team_stats, 
        calculate_race_stats, calculate_combined_standings,
        calculate_combined_constructor_standings, F1_2025_CALENDAR
    )
    from .features import prepare_features
    from .model import train_model, load_trained_model
    from .evaluate import evaluate_model
except ImportError:
    # Fallback for when running from different directories
    try:
        from loader import load_data, clean_data
        from analysis import (
            calculate_driver_stats, calculate_team_stats,
            calculate_race_stats, calculate_combined_standings,
            calculate_combined_constructor_standings, F1_2025_CALENDAR
        )
        from features import prepare_features
        from model import train_model, load_trained_model
        from evaluate import evaluate_model
    except ImportError:
        # If imports still fail, define empty placeholders
        load_data = clean_data = None
        calculate_driver_stats = calculate_team_stats = None
        calculate_race_stats = calculate_combined_standings = None
        calculate_combined_constructor_standings = F1_2025_CALENDAR = None
        prepare_features = train_model = load_trained_model = None
        evaluate_model = None

__version__ = "1.0.0"
__author__ = "maxvy"

__all__ = [
    "load_data",
    "clean_data", 
    "calculate_driver_stats",
    "calculate_team_stats",
    "calculate_race_stats",
    "calculate_combined_standings",
    "calculate_combined_constructor_standings",
    "F1_2025_CALENDAR",
    "prepare_features",
    "train_model",
    "load_trained_model",
    "evaluate_model",
]
