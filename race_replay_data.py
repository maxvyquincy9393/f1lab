# -*- coding: utf-8 -*-
"""
race_replay_data.py
~~~~~~~~~~~~~~~~~~~
Race replay telemetry data processing for animated visualization.

Adapted from f1-race-replay by IAmTomShaw
https://github.com/IAmTomShaw/f1-race-replay

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import os
import pickle
import numpy as np
import pandas as pd
import fastf1
import fastf1.plotting
from pathlib import Path
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Configuration
FPS = 10  # Frames per second for replay (lower = faster processing)
DT = 1 / FPS
CACHE_DIR = Path("./computed_replay_data")


def enable_cache():
    """Enable FastF1 cache."""
    cache_path = Path("./.fastf1-cache")
    cache_path.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def get_tyre_compound_color(compound: str) -> str:
    """Get color for tyre compound."""
    colors = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFDD00', 
        'HARD': '#EEEEEE',
        'INTERMEDIATE': '#39B54A',
        'WET': '#00AAFF',
    }
    return colors.get(str(compound).upper(), '#888888')


def get_tyre_compound_int(compound: str) -> int:
    """Convert tyre compound to integer for processing."""
    mapping = {
        'SOFT': 1,
        'MEDIUM': 2,
        'HARD': 3,
        'INTERMEDIATE': 4,
        'WET': 5,
    }
    return mapping.get(str(compound).upper(), 0)


def get_driver_colors(session) -> dict:
    """Get team colors for all drivers in session."""
    try:
        color_mapping = fastf1.plotting.get_driver_color_mapping(session)
        return {driver: color for driver, color in color_mapping.items()}
    except Exception as e:
        logger.warning(f"Could not get driver colors: {e}")
        return {}


def get_circuit_rotation(session) -> float:
    """Get circuit rotation angle for proper orientation."""
    try:
        circuit = session.get_circuit_info()
        return float(circuit.rotation) if hasattr(circuit, 'rotation') else 0.0
    except Exception as e:
        logger.warning(f"Could not get circuit rotation: {e}")
        return 0.0


def get_track_coordinates(session) -> dict:
    """Extract track layout coordinates from fastest lap."""
    try:
        fastest_lap = session.laps.pick_fastest()
        if fastest_lap is None:
            return None
            
        telemetry = fastest_lap.get_telemetry()
        if telemetry.empty:
            return None
            
        return {
            'x': telemetry['X'].to_numpy(),
            'y': telemetry['Y'].to_numpy(),
        }
    except Exception as e:
        logger.error(f"Could not get track coordinates: {e}")
        return None


def _process_driver_telemetry(driver_no, session, driver_code: str) -> dict:
    """Process telemetry data for a single driver."""
    logger.info(f"Processing telemetry for driver: {driver_code}")
    
    laps = session.laps.pick_drivers(driver_no)
    if laps.empty:
        return None
    
    driver_max_lap = int(laps.LapNumber.max()) if not laps.empty else 0
    
    # Collect all lap data
    t_all, x_all, y_all = [], [], []
    lap_numbers, speed_all, gear_all = [], [], []
    tyre_compounds = []
    
    total_dist_so_far = 0.0
    race_dist_all = []
    
    for _, lap in laps.iterlaps():
        try:
            lap_tel = lap.get_telemetry()
            if lap_tel.empty:
                continue
                
            lap_number = lap.LapNumber
            tyre_int = get_tyre_compound_int(lap.Compound)
            
            t_lap = lap_tel["SessionTime"].dt.total_seconds().to_numpy()
            x_lap = lap_tel["X"].to_numpy()
            y_lap = lap_tel["Y"].to_numpy()
            d_lap = lap_tel["Distance"].to_numpy()
            speed_lap = lap_tel["Speed"].to_numpy()
            gear_lap = lap_tel["nGear"].to_numpy()
            
            race_d_lap = total_dist_so_far + d_lap
            
            t_all.append(t_lap)
            x_all.append(x_lap)
            y_all.append(y_lap)
            race_dist_all.append(race_d_lap)
            lap_numbers.append(np.full_like(t_lap, lap_number))
            tyre_compounds.append(np.full_like(t_lap, tyre_int))
            speed_all.append(speed_lap)
            gear_all.append(gear_lap)
            
            # Update total distance for next lap
            if len(d_lap) > 0:
                total_dist_so_far += d_lap[-1]
                
        except Exception as e:
            logger.warning(f"Error processing lap {lap.LapNumber} for {driver_code}: {e}")
            continue
    
    if not t_all:
        return None
    
    # Concatenate all arrays
    t_all = np.concatenate(t_all)
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    race_dist_all = np.concatenate(race_dist_all)
    lap_numbers = np.concatenate(lap_numbers)
    tyre_compounds = np.concatenate(tyre_compounds)
    speed_all = np.concatenate(speed_all)
    gear_all = np.concatenate(gear_all)
    
    # Sort by time
    order = np.argsort(t_all)
    
    return {
        "code": driver_code,
        "data": {
            "t": t_all[order],
            "x": x_all[order],
            "y": y_all[order],
            "dist": race_dist_all[order],
            "lap": lap_numbers[order],
            "tyre": tyre_compounds[order],
            "speed": speed_all[order],
            "gear": gear_all[order],
        },
        "t_min": t_all.min(),
        "t_max": t_all.max(),
        "max_lap": driver_max_lap
    }


def get_race_replay_frames(session, session_type: str = 'R', use_cache: bool = True) -> dict:
    """
    Generate race replay frames with driver positions at each timestamp.
    
    Returns dict with:
        - frames: list of frame dicts with driver positions
        - track: track coordinate data
        - driver_colors: color mapping for drivers
        - total_laps: max lap count
        - timeline: array of timestamps
    """
    event_name = str(session.event['EventName']).replace(' ', '_')
    cache_suffix = 'sprint' if session_type == 'S' else 'race'
    cache_file = CACHE_DIR / f"{event_name}_{cache_suffix}_replay.pkl"
    
    # Try loading from cache
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded cached replay data for {event_name}")
                return cached_data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    
    # Process fresh data
    drivers = session.drivers
    driver_codes = {
        num: session.get_driver(num)["Abbreviation"]
        for num in drivers
    }
    
    driver_data = {}
    global_t_min = None
    global_t_max = None
    max_lap_number = 0
    
    # Process each driver
    for driver_no in drivers:
        result = _process_driver_telemetry(driver_no, session, driver_codes[driver_no])
        if result is None:
            continue
            
        code = result["code"]
        driver_data[code] = result["data"]
        
        t_min = result["t_min"]
        t_max = result["t_max"]
        max_lap_number = max(max_lap_number, result["max_lap"])
        
        global_t_min = t_min if global_t_min is None else min(global_t_min, t_min)
        global_t_max = t_max if global_t_max is None else max(global_t_max, t_max)
    
    if global_t_min is None or global_t_max is None:
        raise ValueError("No valid telemetry data found")
    
    # Create timeline
    timeline = np.arange(global_t_min, global_t_max, DT) - global_t_min
    
    # Resample each driver's data to common timeline
    resampled_data = {}
    for code, data in driver_data.items():
        t = data["t"] - global_t_min
        order = np.argsort(t)
        t_sorted = t[order]
        
        resampled_data[code] = {
            "t": timeline,
            "x": np.interp(timeline, t_sorted, data["x"][order]),
            "y": np.interp(timeline, t_sorted, data["y"][order]),
            "dist": np.interp(timeline, t_sorted, data["dist"][order]),
            "lap": np.interp(timeline, t_sorted, data["lap"][order]),
            "tyre": np.interp(timeline, t_sorted, data["tyre"][order]),
            "speed": np.interp(timeline, t_sorted, data["speed"][order]),
            "gear": np.interp(timeline, t_sorted, data["gear"][order]),
        }
    
    # Build frames
    frames = []
    driver_codes_list = list(resampled_data.keys())
    
    for i in range(len(timeline)):
        snapshot = []
        for code in driver_codes_list:
            d = resampled_data[code]
            snapshot.append({
                "code": code,
                "x": float(d["x"][i]),
                "y": float(d["y"][i]),
                "dist": float(d["dist"][i]),
                "lap": int(round(d["lap"][i])),
                "tyre": int(d["tyre"][i]),
                "speed": float(d["speed"][i]),
                "gear": int(d["gear"][i]),
            })
        
        # Sort by distance (leader first)
        snapshot.sort(key=lambda r: r["dist"], reverse=True)
        
        # Assign positions
        frame_data = {}
        for idx, car in enumerate(snapshot):
            code = car["code"]
            frame_data[code] = {
                **car,
                "position": idx + 1,
            }
        
        leader = snapshot[0] if snapshot else None
        frames.append({
            "time": float(timeline[i]),
            "drivers": frame_data,
            "leader_lap": leader["lap"] if leader else 0,
        })
    
    # Get additional data
    track_coords = get_track_coordinates(session)
    driver_colors = get_driver_colors(session)
    
    result = {
        "frames": frames,
        "track": track_coords,
        "driver_colors": driver_colors,
        "total_laps": max_lap_number,
        "timeline": timeline,
        "event_name": session.event['EventName'],
    }
    
    # Cache for later use
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Cached replay data for {event_name}")
    except Exception as e:
        logger.warning(f"Could not cache data: {e}")
    
    return result


def get_frame_at_time(frames: list, target_time: float) -> dict:
    """Get the frame closest to the target time."""
    if not frames:
        return None
    
    # Binary search for efficiency
    left, right = 0, len(frames) - 1
    while left < right:
        mid = (left + right) // 2
        if frames[mid]["time"] < target_time:
            left = mid + 1
        else:
            right = mid
    
    return frames[left]


def format_race_time(seconds: float) -> str:
    """Format seconds to race timing format (H:MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
