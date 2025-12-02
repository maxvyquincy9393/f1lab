"""
F1 Data Analysis Module.

Provides statistical analysis functions for F1 race and sprint data.
Includes driver standings, team standings, and race statistics.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

# Configure module logger
logger = logging.getLogger('F1.Analysis')

# F1 2025 Calendar Order (official schedule)
F1_2025_CALENDAR = [
    'Australia', 'China', 'Japan', 'Bahrain', 'Saudi Arabia', 'Miami',
    'Emilia-Romagna', 'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain',
    'Belgium', 'Hungary', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore',
    'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

# Team name normalization mapping - consolidate variations to official names
TEAM_NAME_MAPPING = {
    'McLaren': 'McLaren',
    'McLaren Mercedes': 'McLaren',
    'Red Bull Racing Honda RBPT': 'Red Bull Racing',
    'Red Bull Racing Honda EBPT': 'Red Bull Racing',
    'Red Bull Racing': 'Red Bull Racing',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
    'Aston Martin Aramco Mercedes': 'Aston Martin',
    'Aston Martin': 'Aston Martin',
    'Alpine Renault': 'Alpine',
    'Alpine': 'Alpine',
    'Williams Mercedes': 'Williams',
    'Williams': 'Williams',
    'Racing Bulls Honda RBPT': 'RB',
    'Racing Bulls': 'RB',
    'RB': 'RB',
    'Haas Ferrari': 'Haas',
    'Haas': 'Haas',
    'Kick Sauber Ferrari': 'Kick Sauber',
    'Kick Sauber': 'Kick Sauber',
    'Sauber': 'Kick Sauber',
}


def normalize_team_name(team: str) -> str:
    """Normalize team name to official short name."""
    return TEAM_NAME_MAPPING.get(team, team)


def calculate_driver_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive driver statistics from race data with error handling.
    
    Computes the following metrics per driver:
    - Total Points, Average Points, Race Count
    - Average/Best/Worst Position
    - Finishes count, Fastest Laps
    - Wins, Podiums, Win Rate, Finish Rate
    
    Args:
        df: Cleaned race DataFrame with Position, Points, Finished columns.
        
    Returns:
        pd.DataFrame: Driver statistics indexed by driver name.
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
        
    Example:
        >>> df = load_data('data/Formula1_2025Season_RaceResults.csv')
        >>> df = clean_data(df)
        >>> driver_stats = calculate_driver_stats(df)
        >>> print(driver_stats.loc['Max Verstappen'])
    """
    logger.info("Calculating driver statistics...")
    
    try:
        # Validate input
        if df is None or df.empty:
            logger.error("Cannot calculate driver stats from empty DataFrame")
            return pd.DataFrame()
        
        required_cols = ['Driver', 'Points', 'Position', 'Finished']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Aggregate driver statistics
        driver_stats = df.groupby('Driver').agg({
            'Points': ['sum', 'mean', 'count'],
            'Position': ['mean', 'min', 'max'],
            'Finished': 'sum',
            'Set Fastest Lap': lambda x: (x == 'Yes').sum() if 'Set Fastest Lap' in df.columns else 0
        }).round(2)
        
        driver_stats.columns = ['Total_Points', 'Avg_Points', 'Races', 'Avg_Position', 'Best_Position', 'Worst_Position', 'Finishes', 'Fastest_Laps']
        
        # Add sprint points if available
        if 'Sprint_Points' in df.columns:
            sprint_points = df.groupby('Driver')['Sprint_Points'].first()
            driver_stats['Total_Points'] = driver_stats['Total_Points'] + sprint_points.fillna(0)
            logger.info("Added sprint points to driver totals")
        
        # Calculate wins and podiums
        wins = df[df['Position'] == 1].groupby('Driver').size()
        podiums = df[df['Position'].between(1, 3)].groupby('Driver').size()
        
        driver_stats['Wins'] = wins
        driver_stats['Podium'] = podiums
        driver_stats = driver_stats.fillna(0)
        
        # Calculate percentages with division by zero protection
        driver_stats['Win_Rate'] = np.where(
            driver_stats['Races'] > 0,
            (driver_stats['Wins'] / driver_stats['Races'] * 100).round(1),
            0
        )
        driver_stats['Finish_Rate'] = np.where(
            driver_stats['Races'] > 0,
            (driver_stats['Finishes'] / driver_stats['Races'] * 100).round(1),
            0
        )
        
        logger.info(f"Calculated stats for {len(driver_stats)} drivers")
        return driver_stats
        
    except Exception as e:
        logger.exception(f"Error calculating driver statistics: {e}")
        return pd.DataFrame()


def calculate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive team/constructor statistics.
    Normalizes team names to consolidate variations.
    
    Computes the following metrics per team:
    - Total Points, Average Points per Entry
    - Number of Drivers
    - Average Position, Average Grid Position
    - Finishes, Fastest Laps
    - Points per Driver, Finish Rate
    
    Args:
        df: Cleaned race DataFrame.
        
    Returns:
        pd.DataFrame: Team statistics indexed by normalized team name.
    """
    logger.info("Calculating team statistics...")
    
    # Normalize team names first
    df_norm = df.copy()
    df_norm['Team'] = df_norm['Team'].apply(normalize_team_name)
    
    team_agg = df_norm.groupby('Team').agg({
        'Points': ['sum', 'mean'],
        'Driver': 'nunique',
        'Position': 'mean',
        'Finished': 'sum',
        'Set Fastest Lap': lambda x: (x == 'Yes').sum() if 'Set Fastest Lap' in df_norm.columns else 0,
        'Starting Grid': 'mean'
    }).round(2)
    
    team_agg.columns = ['_'.join(col).strip() for col in team_agg.columns.values]
    
    team_analysis = team_agg.rename(columns={
        'Points_sum': 'Total_Points',
        'Points_mean': 'Avg_Points_Per_Entry',
        'Driver_nunique': 'Drivers',
        'Position_mean': 'Avg_Position',
        'Finished_sum': 'Finishes',
        'Set Fastest Lap_<lambda>': 'Fastest_Laps',
        'Starting Grid_mean': 'Avg_Grid'
    })
    
    # Note: Sprint points are already included in Points column for combined data
    # No need to add them separately
    
    team_analysis['Points_Per_Driver'] = (team_analysis['Total_Points'] / team_analysis['Drivers']).round(1)
    
    races_count = df_norm['Track'].nunique()
    team_analysis['Total_Entries'] = team_analysis['Drivers'] * races_count
    team_analysis['Finish_Rate'] = (team_analysis['Finishes'] / team_analysis['Total_Entries'] * 100).round(1)
    
    logger.info(f"Calculated stats for {len(team_analysis)} teams")
    return team_analysis


def calculate_race_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics per race/track.
    
    Args:
        df: Cleaned race DataFrame.
        
    Returns:
        pd.DataFrame: Race statistics indexed by track name.
    """
    logger.info("Calculating race statistics...")
    race_stats = df.groupby('Track').agg({
        'Driver': 'count',
        'Finished': 'sum',
        'Points': 'sum',
    }).rename(columns={'Driver': 'Entries', 'Finished': 'Finishers'})
    
    race_stats['Finish_Rate'] = (race_stats['Finishers'] / race_stats['Entries'] * 100).round(1)
    
    logger.info(f"Calculated stats for {len(race_stats)} races")
    return race_stats


def calculate_combined_standings(
    df_race: pd.DataFrame, 
    df_sprint: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate combined driver standings with Race + Sprint points.
    
    This matches the official F1 championship standings calculation
    where total points = race points + sprint race points.
    
    Args:
        df_race: Race results DataFrame.
        df_sprint: Sprint results DataFrame (can be None).
        
    Returns:
        pd.DataFrame: Combined standings with Position, Team, Race_Points,
                      Sprint_Points, Total_Points, Wins columns.
                      
    Example:
        >>> standings = calculate_combined_standings(df_race, df_sprint)
        >>> print(standings.head())
    """
    logger.info("Calculating combined driver standings...")
    # Ensure Points column is numeric and fill NaN with 0
    df_sprint = df_sprint.copy()
    df_sprint['Points'] = pd.to_numeric(df_sprint['Points'], errors='coerce').fillna(0)
    
    # Calculate race points per driver
    race_points = df_race.groupby('Driver').agg({
        'Points': 'sum',
        'Team': 'first'
    }).rename(columns={'Points': 'Race_Points'})
    
    # Calculate sprint points per driver
    sprint_points = df_sprint.groupby('Driver')['Points'].sum().rename('Sprint_Points')
    
    # Combine standings
    standings = race_points.join(sprint_points, how='outer').fillna(0)
    standings['Total_Points'] = standings['Race_Points'] + standings['Sprint_Points']
    
    # Calculate wins
    race_wins = df_race[df_race['Position'] == 1].groupby('Driver').size().rename('Race_Wins')
    sprint_wins = df_sprint[df_sprint['Position'] == 1].groupby('Driver').size().rename('Sprint_Wins')
    
    standings = standings.join(race_wins, how='left').fillna(0)
    standings = standings.join(sprint_wins, how='left').fillna(0)
    standings['Total_Wins'] = standings['Race_Wins'] + standings['Sprint_Wins']
    
    # Sort by total points (tiebreaker: race wins)
    standings = standings.sort_values(['Total_Points', 'Race_Wins'], ascending=[False, False])
    standings['Position'] = range(1, len(standings) + 1)
    
    standings = standings[['Position', 'Team', 'Race_Points', 'Sprint_Points', 'Total_Points', 
                           'Race_Wins', 'Sprint_Wins', 'Total_Wins']]
    
    logger.info(f"Combined standings calculated: {len(standings)} drivers")
    return standings


def calculate_combined_constructor_standings(
    df_race: pd.DataFrame, 
    df_sprint: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate combined constructor standings with Race + Sprint points.
    Normalizes team names to consolidate variations (e.g., McLaren Mercedes -> McLaren).
    
    Args:
        df_race: Race results DataFrame.
        df_sprint: Sprint results DataFrame (can be None).
        
    Returns:
        pd.DataFrame: Constructor standings indexed by normalized team name.
    """
    logger.info("Calculating combined constructor standings...")
    
    # Copy dataframes and normalize team names
    df_race_norm = df_race.copy()
    df_sprint_norm = df_sprint.copy()
    
    df_race_norm['Team'] = df_race_norm['Team'].apply(normalize_team_name)
    df_sprint_norm['Team'] = df_sprint_norm['Team'].apply(normalize_team_name)
    
    # Ensure Points column is numeric and fill NaN with 0
    df_sprint_norm['Points'] = pd.to_numeric(df_sprint_norm['Points'], errors='coerce').fillna(0)
    
    race_points = df_race_norm.groupby('Team')['Points'].sum().rename('Race_Points')
    sprint_points = df_sprint_norm.groupby('Team')['Points'].sum().rename('Sprint_Points')
    
    standings = pd.DataFrame({'Race_Points': race_points, 'Sprint_Points': sprint_points}).fillna(0)
    standings['Total_Points'] = standings['Race_Points'] + standings['Sprint_Points']
    
    race_wins = df_race_norm[df_race_norm['Position'] == 1].groupby('Team').size().rename('Race_Wins')
    sprint_wins = df_sprint_norm[df_sprint_norm['Position'] == 1].groupby('Team').size().rename('Sprint_Wins')
    
    standings = standings.join(race_wins, how='left').fillna(0)
    standings = standings.join(sprint_wins, how='left').fillna(0)
    
    standings = standings.sort_values('Total_Points', ascending=False)
    standings['Position'] = range(1, len(standings) + 1)
    
    standings = standings[['Position', 'Race_Points', 'Sprint_Points', 'Total_Points', 'Race_Wins', 'Sprint_Wins']]
    
    logger.info(f"Constructor standings calculated: {len(standings)} teams")
    return standings
