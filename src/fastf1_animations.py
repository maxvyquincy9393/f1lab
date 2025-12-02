"""
FastF1 Advanced Animation Module.

Provides comprehensive animation and visualization functions for F1 telemetry data.
Includes position tracking, gear changes, DRS zones, tire strategy, and more.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logger = logging.getLogger('f1_visualization.fastf1_animations')


def get_position_animation_data(
    session: Any,
    lap_number: Optional[int] = None,
    drivers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate track position animation data with speed heatmap.
    
    Args:
        session: FastF1 Session object
        lap_number: Specific lap number (None for fastest laps)
        drivers: List of driver codes (None for all)
        
    Returns:
        Dictionary with animation frames for each driver
    """
    try:
        logger.info("Generating position animation data...")
        
        if session is None:
            logger.warning("No session provided")
            return {}
        
        if drivers is None:
            drivers = session.results['Abbreviation'].tolist()[:10]  # Top 10
        
        animation_data = {
            'drivers': [],
            'frames': [],
            'layout': {'x_range': [], 'y_range': []}
        }
        
        for driver_code in drivers:
            try:
                driver_laps = session.laps.pick_driver(driver_code)
                if lap_number:
                    lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
                else:
                    lap = driver_laps.pick_fastest()
                
                tel = lap.get_telemetry()
                
                animation_data['drivers'].append({
                    'code': driver_code,
                    'team': lap['Team'],
                    'x': tel['X'].tolist(),
                    'y': tel['Y'].tolist(),
                    'speed': tel['Speed'].tolist(),
                    'distance': tel['Distance'].tolist() if 'Distance' in tel.columns else []
                })
                
                logger.debug(f"Added position data for {driver_code}")
                
            except Exception as e:
                logger.warning(f"Could not get position data for {driver_code}: {e}")
        
        logger.info(f"Generated position animation for {len(animation_data['drivers'])} drivers")
        return animation_data
        
    except Exception as e:
        logger.exception(f"Error generating position animation: {e}")
        return {}


def get_gear_animation_data(
    session: Any,
    driver_code: str,
    num_frames: int = 200
) -> Dict[str, Any]:
    """
    Generate gear changes animation data.
    
    Args:
        session: FastF1 Session object
        driver_code: Driver abbreviation
        num_frames: Number of animation frames
        
    Returns:
        Dictionary with gear change data by track position
    """
    try:
        logger.info(f"Generating gear animation for {driver_code}...")
        
        if session is None:
            return {}
        
        lap = session.laps.pick_driver(driver_code).pick_fastest()
        tel = lap.get_telemetry()
        
        # Extract gear data
        gear_data = {
            'driver': driver_code,
            'x': tel['X'].tolist(),
            'y': tel['Y'].tolist(),
            'gear': tel['nGear'].tolist(),
            'speed': tel['Speed'].tolist(),
            'distance': tel['Distance'].tolist() if 'Distance' in tel.columns else [],
            'rpm': tel['RPM'].tolist() if 'RPM' in tel.columns else []
        }
        
        # Calculate gear change points
        gear_changes = []
        prev_gear = tel['nGear'].iloc[0]
        for i, gear in enumerate(tel['nGear']):
            if gear != prev_gear:
                gear_changes.append({
                    'index': i,
                    'from_gear': int(prev_gear),
                    'to_gear': int(gear),
                    'x': float(tel['X'].iloc[i]),
                    'y': float(tel['Y'].iloc[i])
                })
                prev_gear = gear
        
        gear_data['changes'] = gear_changes
        
        logger.info(f"Generated gear animation with {len(gear_changes)} gear changes")
        return gear_data
        
    except Exception as e:
        logger.exception(f"Error generating gear animation: {e}")
        return {}


def get_drs_zones_animation(
    session: Any,
    driver_code: str
) -> Dict[str, Any]:
    """
    Generate DRS activation zones animation.
    
    Args:
        session: FastF1 Session object
        driver_code: Driver abbreviation
        
    Returns:
        Dictionary with DRS activation data
    """
    try:
        logger.info(f"Generating DRS zones for {driver_code}...")
        
        if session is None:
            return {}
        
        lap = session.laps.pick_driver(driver_code).pick_fastest()
        tel = lap.get_telemetry()
        
        if 'DRS' not in tel.columns:
            logger.warning("DRS data not available")
            return {}
        
        drs_data = {
            'driver': driver_code,
            'x': tel['X'].tolist(),
            'y': tel['Y'].tolist(),
            'drs_status': tel['DRS'].tolist(),
            'speed': tel['Speed'].tolist()
        }
        
        # Identify DRS zones (consecutive DRS=1 segments)
        drs_zones = []
        in_zone = False
        zone_start = 0
        
        for i, drs_val in enumerate(tel['DRS']):
            if drs_val > 0 and not in_zone:
                # Entering DRS zone
                in_zone = True
                zone_start = i
            elif drs_val == 0 and in_zone:
                # Exiting DRS zone
                in_zone = False
                drs_zones.append({
                    'start_idx': zone_start,
                    'end_idx': i,
                    'x_start': float(tel['X'].iloc[zone_start]),
                    'y_start': float(tel['Y'].iloc[zone_start]),
                    'x_end': float(tel['X'].iloc[i]),
                    'y_end': float(tel['Y'].iloc[i])
                })
        
        drs_data['zones'] = drs_zones
        
        logger.info(f"Identified {len(drs_zones)} DRS zones")
        return drs_data
        
    except Exception as e:
        logger.exception(f"Error generating DRS zones: {e}")
        return {}


def get_tire_strategy_animation(
    session: Any,
    driver_code: str
) -> Dict[str, Any]:
    """
    Generate tire compound and strategy animation.
    
    Args:
        session: FastF1 Session object
        driver_code: Driver abbreviation
        
    Returns:
        Dictionary with tire strategy data
    """
    try:
        logger.info(f"Generating tire strategy for {driver_code}...")
        
        if session is None:
            return {}
        
        driver_laps = session.laps.pick_driver(driver_code)
        
        tire_stints = []
        current_compound = None
        stint_start_lap = 0
        
        for idx, lap in driver_laps.iterrows():
            lap_num = lap['LapNumber']
            compound = lap['Compound']
            
            if compound != current_compound:
                if current_compound is not None:
                    # Save previous stint
                    tire_stints.append({
                        'compound': current_compound,
                        'start_lap': stint_start_lap,
                        'end_lap': lap_num - 1,
                        'num_laps': lap_num - stint_start_lap
                    })
                
                current_compound = compound
                stint_start_lap = lap_num
        
        # Add final stint
        if current_compound is not None:
            tire_stints.append({
                'compound': current_compound,
                'start_lap': stint_start_lap,
                'end_lap': driver_laps['LapNumber'].max(),
                'num_laps': driver_laps['LapNumber'].max() - stint_start_lap + 1
            })
        
        strategy_data = {
            'driver': driver_code,
            'stints': tire_stints,
            'total_laps': len(driver_laps)
        }
        
        logger.info(f"Generated tire strategy with {len(tire_stints)} stints")
        return strategy_data
        
    except Exception as e:
        logger.exception(f"Error generating tire strategy: {e}")
        return {}


def get_lap_evolution_animation(
    session: Any,
    driver_code: str
) -> Dict[str, Any]:
    """
    Generate lap time evolution animation.
    
    Args:
        session: FastF1 Session object
        driver_code: Driver abbreviation
        
    Returns:
        Dictionary with lap time progression
    """
    try:
        logger.info(f"Generating lap evolution for {driver_code}...")
        
        if session is None:
            return {}
        
        driver_laps = session.laps.pick_driver(driver_code)
        
        lap_times = []
        for idx, lap in driver_laps.iterrows():
            if pd.notna(lap['LapTime']):
                lap_times.append({
                    'lap_number': lap['LapNumber'],
                    'lap_time': lap['LapTime'].total_seconds(),
                    'compound': lap['Compound'],
                    'stint': lap.get('Stint', 0)
                })
        
        evolution_data = {
            'driver': driver_code,
            'laps': lap_times,
            'fastest_lap': min([lt['lap_time'] for lt in lap_times]) if lap_times else None
        }
        
        logger.info(f"Generated lap evolution with {len(lap_times)} laps")
        return evolution_data
        
    except Exception as e:
        logger.exception(f"Error generating lap evolution: {e}")
        return {}


def get_sector_comparison_animation(
    session: Any,
    drivers: List[str]
) -> Dict[str, Any]:
    """
    Generate sector time comparison animation.
    
    Args:
        session: FastF1 Session object
        drivers: List of driver codes to compare
        
    Returns:
        Dictionary with sector comparison data
    """
    try:
        logger.info(f"Generating sector comparison for {len(drivers)} drivers...")
        
        if session is None:
            return {}
        
        sector_data = []
        
        for driver_code in drivers:
            try:
                lap = session.laps.pick_driver(driver_code).pick_fastest()
                
                sector_data.append({
                    'driver': driver_code,
                    'team': lap['Team'],
                    'sector1': lap['Sector1Time'].total_seconds() if pd.notna(lap['Sector1Time']) else None,
                    'sector2': lap['Sector2Time'].total_seconds() if pd.notna(lap['Sector2Time']) else None,
                    'sector3': lap['Sector3Time'].total_seconds() if pd.notna(lap['Sector3Time']) else None,
                    'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None
                })
                
            except Exception as e:
                logger.warning(f"Could not get sector data for {driver_code}: {e}")
        
        comparison = {
            'drivers': sector_data,
            'best_sectors': {
                's1': min([d['sector1'] for d in sector_data if d['sector1']], default=None),
                's2': min([d['sector2'] for d in sector_data if d['sector2']], default=None),
                's3': min([d['sector3'] for d in sector_data if d['sector3']], default=None)
            }
        }
        
        logger.info(f"Generated sector comparison for {len(sector_data)} drivers")
        return comparison
        
    except Exception as e:
        logger.exception(f"Error generating sector comparison: {e}")
        return {}


def get_speed_trap_data(
    session: Any,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Get speed trap data for all drivers.
    
    Args:
        session: FastF1 Session object
        top_n: Number of top speeds to return
        
    Returns:
        Dictionary with speed trap data
    """
    try:
        logger.info("Generating speed trap data...")

        if session is None:
            return {}
        
        speed_data = []
        
        for idx, driver_row in session.results.iterrows():
            driver_code = driver_row['Abbreviation']
            try:
                lap = session.laps.pick_driver(driver_code).pick_fastest()
                tel = lap.get_telemetry()
                
                max_speed = tel['Speed'].max()
                
                speed_data.append({
                    'driver': driver_code,
                    'team': driver_row['TeamName'],
                    'max_speed': float(max_speed),
                    'avg_speed': float(tel['Speed'].mean())
                })
                
            except Exception as e:
                logger.warning(f"Could not get speed data for {driver_code}: {e}")
        
        # Sort by max speed
        speed_data.sort(key=lambda x: x['max_speed'], reverse=True)
        
        trap_data = {
            'speeds': speed_data[:top_n],
            'fastest': speed_data[0] if speed_data else None
        }
        
        logger.info(f"Generated speed trap data for {len(speed_data)} drivers")
        return trap_data
        
    except Exception as e:
        logger.exception(f"Error generating speed trap data: {e}")
        return {}


def get_race_pace_animation(
    session: Any,
    drivers: List[str],
    rolling_window: int = 5
) -> Dict[str, Any]:
    """
    Generate race pace comparison animation with rolling average.
    
    Args:
        session: FastF1 Session object
        drivers: List of driver codes
        rolling_window: Window size for rolling average
        
    Returns:
        Dictionary with race pace data
    """
    try:
        logger.info(f"Generating race pace for {len(drivers)} drivers...")
        
        if session is None:
            return {}
        
        pace_data = []
        
        for driver_code in drivers:
            try:
                driver_laps = session.laps.pick_driver(driver_code)
                
                # Extract lap times
                lap_times = []
                for idx, lap in driver_laps.iterrows():
                    if pd.notna(lap['LapTime']):
                        lap_times.append({
                            'lap': lap['LapNumber'],
                            'time': lap['LapTime'].total_seconds(),
                            'compound': lap['Compound']
                        })
                
                # Calculate rolling average
                if len(lap_times) >= rolling_window:
                    times = [lt['time'] for lt in lap_times]
                    rolling_avg = pd.Series(times).rolling(window=rolling_window).mean().tolist()
                else:
                    rolling_avg = [lt['time'] for lt in lap_times]
                
                pace_data.append({
                    'driver': driver_code,
                    'laps': lap_times,
                    'rolling_avg': rolling_avg
                })
                
            except Exception as e:
                logger.warning(f"Could not get race pace for {driver_code}: {e}")
        
        race_pace = {
            'drivers': pace_data,
            'rolling_window': rolling_window
        }
        
        logger.info(f"Generated race pace for {len(pace_data)} drivers")
        return race_pace
        
    except Exception as e:
        logger.exception(f"Error generating race pace: {e}")
        return {}
