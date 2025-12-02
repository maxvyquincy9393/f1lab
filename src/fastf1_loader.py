"""
FastF1 Loader Module

This module provides wrapper functions for the FastF1 library,
with proper error handling, logging, and retry mechanisms.
"""

import fastf1
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import wraps

# Setup logging
logger = logging.getLogger('f1_visualization.fastf1_loader')

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Default retry configuration
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0


# ============================================================
# DECORATORS
# ============================================================

def retry_on_error(
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    delay: float = DEFAULT_RETRY_DELAY
):
    """
    Decorator to retry function on error.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            
            logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            raise last_exception
        return wrapper
    return decorator


# ============================================================
# SESSION LOADING
# ============================================================

@retry_on_error()
def get_session(
    year: int,
    race: Union[str, int],
    session_type: str = 'R'
) -> Optional[Any]:
    """
    Load a FastF1 session with error handling and retry.
    
    Args:
        year: Season year (e.g., 2025)
        race: Race name, location, or round number
        session_type: Session type - 'R' (Race), 'Q' (Qualifying), 
                     'FP1', 'FP2', 'FP3', 'S' (Sprint), 'SQ' (Sprint Qualifying)
    
    Returns:
        FastF1 Session object or None if loading fails
    """
    logger.info(f"Loading session: {year} {race} - {session_type}")
    
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load()
        logger.info(f"Session loaded: {session.event['EventName']} - {session.name}")
        return session
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        return None


# ============================================================
# RESULTS AND LAP DATA
# ============================================================

def get_race_results(session: Any) -> pd.DataFrame:
    """Extract race results from session."""
    if session is None:
        logger.warning("No session provided for race results")
        return pd.DataFrame()
    
    try:
        results = session.results.copy()
        columns = [
            'DriverNumber', 'Abbreviation', 'FullName', 'TeamName',
            'Position', 'GridPosition', 'Points', 'Status', 'Time'
        ]
        available_cols = [col for col in columns if col in results.columns]
        logger.debug(f"Extracted {len(results)} driver results")
        return results[available_cols]
    except Exception as e:
        logger.error(f"Error getting race results: {e}")
        return pd.DataFrame()


def get_lap_data(
    session: Any,
    driver: Optional[str] = None
) -> pd.DataFrame:
    """Extract lap data from session."""
    if session is None:
        logger.warning("No session provided for lap data")
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        if driver:
            laps = laps[laps['Driver'] == driver]
            logger.debug(f"Filtered laps for driver: {driver}")
        logger.debug(f"Extracted {len(laps)} laps")
        return laps
    except Exception as e:
        logger.error(f"Error getting lap data: {e}")
        return pd.DataFrame()


# ============================================================
# TELEMETRY
# ============================================================

def get_telemetry(
    lap_or_session: Any,
    driver: Optional[str] = None,
    lap: Optional[int] = None
) -> pd.DataFrame:
    """Get telemetry data for a lap or driver."""
    if hasattr(lap_or_session, 'get_telemetry'):
        try:
            telemetry = lap_or_session.get_telemetry()
            logger.debug(f"Extracted telemetry with {len(telemetry)} data points")
            return telemetry
        except Exception as e:
            logger.error(f"Error getting telemetry from lap: {e}")
            return pd.DataFrame()
    
    session = lap_or_session
    if session is None:
        logger.warning("No session provided for telemetry")
        return pd.DataFrame()
    
    try:
        if driver is None:
            logger.error("Driver code required when passing session")
            return pd.DataFrame()
        
        driver_laps = session.laps.pick_driver(driver)
        
        if lap:
            target_lap = driver_laps[driver_laps['LapNumber'] == lap].iloc[0]
        else:
            target_lap = driver_laps.pick_fastest()
        
        telemetry = target_lap.get_telemetry()
        logger.debug(f"Extracted telemetry for {driver}: {len(telemetry)} points")
        return telemetry
    except Exception as e:
        logger.error(f"Error getting telemetry: {e}")
        return pd.DataFrame()


# ============================================================
# WEATHER AND SCHEDULE
# ============================================================

def get_weather_data(session: Any) -> pd.DataFrame:
    """Extract weather data from session."""
    if session is None:
        logger.warning("No session provided for weather data")
        return pd.DataFrame()
    
    try:
        weather = session.weather_data.copy()
        logger.debug(f"Extracted {len(weather)} weather records")
        return weather
    except Exception as e:
        logger.error(f"Error getting weather data: {e}")
        return pd.DataFrame()


@retry_on_error()
def get_schedule(year: int) -> pd.DataFrame:
    """Get F1 event schedule for a season."""
    logger.info(f"Loading {year} F1 schedule")
    
    try:
        schedule = fastf1.get_event_schedule(year)
        logger.debug(f"Loaded schedule with {len(schedule)} events")
        return schedule
    except Exception as e:
        logger.error(f"Error getting schedule: {e}")
        return pd.DataFrame()


# ============================================================
# DRIVER COMPARISON
# ============================================================

def compare_drivers_lap_times(
    session: Any,
    drivers: List[str]
) -> pd.DataFrame:
    """Compare fastest lap data between multiple drivers."""
    if session is None:
        logger.warning("No session provided for driver comparison")
        return pd.DataFrame()
    
    comparison_data = []
    
    for driver in drivers:
        try:
            driver_laps = session.laps.pick_driver(driver)
            fastest = driver_laps.pick_fastest()
            
            comparison_data.append({
                'Driver': driver,
                'FastestLap': fastest['LapTime'],
                'Sector1': fastest['Sector1Time'],
                'Sector2': fastest['Sector2Time'],
                'Sector3': fastest['Sector3Time'],
                'Compound': fastest['Compound']
            })
            logger.debug(f"Added comparison data for {driver}")
        except Exception as e:
            logger.warning(f"Could not get data for driver {driver}: {e}")
    
    return pd.DataFrame(comparison_data)


# ============================================================
# ANIMATION AND VISUALIZATION HELPERS
# ============================================================

def get_track_layout(session: Any) -> Dict[str, List]:
    """Get track layout coordinates for visualization."""
    if session is None:
        logger.warning("No session provided for track layout")
        return {'x': [], 'y': [], 'speed': []}
    
    try:
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry()
        
        return {
            'x': tel['X'].tolist(),
            'y': tel['Y'].tolist(),
            'speed': tel['Speed'].tolist()
        }
    except Exception as e:
        logger.error(f"Error getting track layout: {e}")
        return {'x': [], 'y': [], 'speed': []}


def get_driver_telemetry_data(
    session: Any,
    driver_code: str
) -> Dict[str, Any]:
    """Get comprehensive telemetry data for a driver."""
    if session is None:
        logger.warning("No session provided for driver telemetry")
        return {}
    
    try:
        lap = session.laps.pick_driver(driver_code).pick_fastest()
        tel = lap.get_telemetry().add_distance()
        
        return {
            'driver': driver_code,
            'lap_time': str(lap['LapTime']),
            'distance': tel['Distance'].tolist(),
            'speed': tel['Speed'].tolist(),
            'throttle': tel['Throttle'].tolist(),
            'brake': tel['Brake'].tolist(),
            'gear': tel['nGear'].tolist(),
            'drs': tel['DRS'].tolist() if 'DRS' in tel.columns else [],
            'x': tel['X'].tolist(),
            'y': tel['Y'].tolist()
        }
    except Exception as e:
        logger.error(f"Error getting telemetry for {driver_code}: {e}")
        return {}


def generate_animation_frames(
    session: Any,
    driver_code: str,
    num_frames: int = 100
) -> List[Dict]:
    """Generate frame data for animation."""
    if session is None:
        logger.warning("No session provided for animation frames")
        return []
    
    try:
        lap = session.laps.pick_driver(driver_code).pick_fastest()
        tel = lap.get_telemetry()
        
        x = tel['X'].values
        y = tel['Y'].values
        speed = tel['Speed'].values
        
        step = max(1, len(x) // num_frames)
        frames = []
        
        for i in range(0, len(x), step):
            trail_start = max(0, i - 30)
            frames.append({
                'frame': len(frames),
                'x': float(x[i]),
                'y': float(y[i]),
                'speed': float(speed[i]),
                'trail_x': x[trail_start:i+1].tolist(),
                'trail_y': y[trail_start:i+1].tolist()
            })
        
        logger.debug(f"Generated {len(frames)} animation frames for {driver_code}")
        return frames
    except Exception as e:
        logger.error(f"Error generating animation frames: {e}")
        return []


def get_delta_time_data(
    session: Any,
    driver1: str,
    driver2: str
) -> Dict[str, Any]:
    """Calculate delta time between two drivers."""
    if session is None:
        logger.warning("No session provided for delta calculation")
        return {}
    
    try:
        lap1 = session.laps.pick_driver(driver1).pick_fastest()
        lap2 = session.laps.pick_driver(driver2).pick_fastest()
        
        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()
        
        max_dist = min(tel1['Distance'].max(), tel2['Distance'].max())
        common_distance = np.linspace(0, max_dist, 500)
        
        time1 = np.interp(
            common_distance,
            tel1['Distance'],
            tel1['Time'].dt.total_seconds()
        )
        time2 = np.interp(
            common_distance,
            tel2['Distance'],
            tel2['Time'].dt.total_seconds()
        )
        delta = time2 - time1
        
        logger.debug(f"Calculated delta: {driver1} vs {driver2}")
        return {
            'driver1': driver1,
            'driver2': driver2,
            'distance': common_distance.tolist(),
            'delta': delta.tolist(),
            'final_delta': float(delta[-1])
        }
    except Exception as e:
        logger.error(f"Error calculating delta: {e}")
        return {}


def get_battle_animation_data(
    session: Any,
    driver1: str,
    driver2: str,
    num_frames: int = 100
) -> Dict[str, Any]:
    """Generate battle animation data for two drivers."""
    if session is None:
        logger.warning("No session provided for battle animation")
        return {}
    
    try:
        lap1 = session.laps.pick_driver(driver1).pick_fastest()
        lap2 = session.laps.pick_driver(driver2).pick_fastest()
        
        tel1 = lap1.get_telemetry()
        tel2 = lap2.get_telemetry()
        
        length = min(len(tel1), len(tel2))
        step = max(1, length // num_frames)
        
        frames = []
        for i in range(0, length, step):
            trail_start = max(0, i - 30)
            frames.append({
                'frame': len(frames),
                'car1': {
                    'x': float(tel1['X'].iloc[i]),
                    'y': float(tel1['Y'].iloc[i]),
                    'trail_x': tel1['X'].iloc[trail_start:i+1].tolist(),
                    'trail_y': tel1['Y'].iloc[trail_start:i+1].tolist()
                },
                'car2': {
                    'x': float(tel2['X'].iloc[i]),
                    'y': float(tel2['Y'].iloc[i]),
                    'trail_x': tel2['X'].iloc[trail_start:i+1].tolist(),
                    'trail_y': tel2['Y'].iloc[trail_start:i+1].tolist()
                }
            })
        
        logger.debug(f"Generated battle animation: {driver1} vs {driver2}")
        return {
            'driver1': driver1,
            'driver2': driver2,
            'team1': lap1['Team'],
            'team2': lap2['Team'],
            'frames': frames
        }
    except Exception as e:
        logger.error(f"Error generating battle animation: {e}")
        return {}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_available_sessions(year: int, race: Union[str, int]) -> List[str]:
    """Get list of available session types for a race."""
    session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R', 'S', 'SQ']
    available = []
    
    for session_type in session_types:
        try:
            session = fastf1.get_session(year, race, session_type)
            available.append(session_type)
        except:
            pass
    
    return available


def get_driver_codes(session: Any) -> List[str]:
    """Get list of driver codes from session."""
    if session is None:
        return []
    
    try:
        return session.results['Abbreviation'].tolist()
    except:
        return []
