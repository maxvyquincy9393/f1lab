"""
F1 Data Loader Module.

Provides functions for loading and cleaning F1 race data from CSV files.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

# Configure module logger
logger = logging.getLogger('F1.Loader')

# Team name normalization mapping
TEAM_NAME_MAPPING = {
    'McLaren Mercedes': 'McLaren',
    'McLaren': 'McLaren',
    'Red Bull Racing Honda RBPT': 'Red Bull Racing',
    'Red Bull Racing Honda EBPT': 'Red Bull Racing',
    'Red Bull Racing': 'Red Bull Racing',
    'Racing Bulls Honda RBPT': 'RB',
    'RB': 'RB',
    'Kick Sauber Ferrari': 'Kick Sauber',
    'Kick Sauber': 'Kick Sauber',
    'Alpine Renault': 'Alpine',
    'Alpine': 'Alpine',
    'Williams Mercedes': 'Williams',
    'Williams': 'Williams',
    'Aston Martin Aramco Mercedes': 'Aston Martin',
    'Aston Martin': 'Aston Martin',
    'Haas Ferrari': 'Haas',
    'Haas': 'Haas',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
}


def normalize_team_name(team_name: str) -> str:
    """Normalize team name to standard form."""
    return TEAM_NAME_MAPPING.get(team_name, team_name)


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load F1 data from a CSV file with comprehensive error handling.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        pd.DataFrame or None: Loaded data or None if loading fails.
        
    Example:
        >>> df = load_data('data/Formula1_2025Season_RaceResults.csv')
        >>> if df is not None:
        ...     print(f"Loaded {len(df)} rows")
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Try different encodings if UTF-8 fails
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 encoding failed, trying latin-1 for {file_path}")
            df = pd.read_csv(file_path, encoding='latin-1')
        
        # Validate loaded data
        if df.empty:
            logger.warning(f"Loaded empty DataFrame from {file_path}")
            return None
            
        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns from {file_path}")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.error(f"Please ensure the file exists at the specified path")
        return None
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error loading data from {file_path}: {e}")
        return None


def load_combined_data(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load and combine race and sprint data with combined points.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        pd.DataFrame with race data and combined points (race + sprint).
    """
    try:
        data_path = Path(data_dir)
        
        # Load race results
        race_file = data_path / 'Formula1_2025Season_RaceResults.csv'
        race_df = load_data(str(race_file))
        if race_df is None:
            return None
        
        # Load sprint results
        sprint_file = data_path / 'Formula1_2025Season_SprintResults.csv'
        sprint_df = None
        if sprint_file.exists():
            sprint_df = load_data(str(sprint_file))
        
        # Calculate combined points per driver
        if sprint_df is not None and not sprint_df.empty:
            sprint_points = sprint_df.groupby('Driver')['Points'].sum().to_dict()
            
            # Add sprint points to race data (distribute across races)
            race_df['Sprint_Points'] = race_df['Driver'].map(sprint_points).fillna(0)
            
            # Calculate total points per driver across all races
            race_df['Race_Points'] = race_df['Points']
            
            logger.info(f"Combined race and sprint data. Sprint races: {sprint_df['Track'].nunique()}")
        else:
            race_df['Sprint_Points'] = 0
            race_df['Race_Points'] = race_df['Points']
        
        return race_df
        
    except Exception as e:
        logger.exception(f"Error loading combined data: {e}")
        return None


def get_driver_total_points(data_dir: str) -> dict:
    """
    Calculate total points (race + sprint) per driver.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        Dictionary mapping driver name to total points.
    """
    try:
        data_path = Path(data_dir)
        
        # Load race results
        race_file = data_path / 'Formula1_2025Season_RaceResults.csv'
        race_df = load_data(str(race_file))
        if race_df is None:
            return {}
        
        race_points = race_df.groupby('Driver')['Points'].sum()
        
        # Load sprint results
        sprint_file = data_path / 'Formula1_2025Season_SprintResults.csv'
        sprint_points = pd.Series(dtype=float)
        if sprint_file.exists():
            sprint_df = load_data(str(sprint_file))
            if sprint_df is not None:
                sprint_points = sprint_df.groupby('Driver')['Points'].sum()
        
        # Combine points
        total_points = race_points.add(sprint_points, fill_value=0)
        
        return total_points.to_dict()
        
    except Exception as e:
        logger.exception(f"Error calculating total points: {e}")
        return {}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess F1 race data with comprehensive error handling.
    
    Performs the following cleaning operations:
    - Validates input DataFrame
    - Converts Points to numeric (fills NaN with 0)
    - Converts Position to numeric
    - Converts Starting Grid to numeric
    - Converts Laps to numeric
    - Creates Finished flag based on Position and Time/Retired
    
    Args:
        df: Raw DataFrame from CSV.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with proper data types.
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> df_raw = load_data('data/Formula1_2025Season_RaceResults.csv')
        >>> df_clean = clean_data(df_raw)
        >>> print(df_clean.dtypes)
    """
    logger.info("Cleaning data...")
    
    try:
        # Validate input
        if df is None or df.empty:
            logger.error("Cannot clean None or empty DataFrame")
            raise ValueError("DataFrame is None or empty")
        
        # Check for required columns
        required_columns = ['Points', 'Position', 'Starting Grid', 'Laps', 'Time/Retired']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.warning(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Normalize team names
        if 'Team' in df.columns:
            df['Team'] = df['Team'].apply(normalize_team_name)
            logger.info(f"Normalized team names to {df['Team'].nunique()} unique teams")
        
        # Convert numeric columns with error handling
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        df['Starting Grid'] = pd.to_numeric(df['Starting Grid'], errors='coerce')
        df['Laps'] = pd.to_numeric(df['Laps'], errors='coerce')
        
        # Create finished flag - must have valid position and not DNS/DSQ/DNF
        try:
            df['Finished'] = (
                df['Position'].notna() & 
                ~df['Time/Retired'].str.contains('DNS|DSQ|DNF', na=False, regex=True)
            )
        except Exception as e:
            logger.warning(f"Error creating Finished flag: {e}. Using position-only check.")
            df['Finished'] = df['Position'].notna()
        
        # Log cleaning summary
        finishers = df['Finished'].sum()
        logger.info(f"Data cleaned: {len(df)} rows, {finishers} finishers ({finishers/len(df)*100:.1f}%)")
        logger.debug(f"Points range: {df['Points'].min()}-{df['Points'].max()}")
        logger.debug(f"Position range: {df['Position'].min()}-{df['Position'].max()}")
        
        return df
        
    except Exception as e:
        logger.exception(f"Error cleaning data: {e}")
        raise
