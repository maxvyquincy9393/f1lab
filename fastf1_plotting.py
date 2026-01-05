# -*- coding: utf-8 -*-
"""
fastf1_plotting.py
~~~~~~~~~~~~~~~~~~
Matplotlib track and telemetry plots.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)

def setup_f1_plotting():
    """Initialize FastF1 plotting style."""
    try:
        fastf1.plotting.setup_mpl(misc_mpl_mods=False)
        plt.style.use('dark_background')
    except Exception as e:
        logger.error(f"Error setting up plotting: {e}")

def plot_circuit_with_corners(session):
    """
    Generate a matplotlib figure of the circuit with corners and DRS zones.
    Returns: matplotlib Figure object
    """
    setup_f1_plotting()
    
    try:
        circuit_info = session.get_circuit_info()
        lap = session.laps.pick_fastest()
        pos = lap.get_telemetry().add_distance()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Track line
        ax.plot(pos['X'], pos['Y'], color='white', linewidth=2, label='Track')
        
        # Corners
        if circuit_info and circuit_info.corners is not None:
            # Draw corner markers
            # Note: circuit_info.corners provides distance, need to map to X,Y
            # This is an approximation as direct mapping requires interpolation
            pass 
            
        # FastF1 usually provides a rotated track map. 
        # For the standard plotting, let's stick to the simple track map 
        # overlaid with corner numbers if possible.
        
        # Improved approach: Use FastF1's native ability if available, 
        # otherwise manually annotate.
        # Since mapping corner distance to X,Y accurately needs interpolation:
        
        import numpy as np
        
        track = pos[['X', 'Y', 'Distance']].copy()
        
        if circuit_info:
            # Annotate corners
            for _, corner in circuit_info.corners.iterrows():
                dist = corner['Distance']
                # Find closest point
                idx = (np.abs(track['Distance'] - dist)).argmin()
                x, y = track.iloc[idx]['X'], track.iloc[idx]['Y']
                
                ax.text(x, y, str(corner['Number']) + corner['Letter'], 
                        color='yellow', fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='black', edgecolor='yellow', boxstyle='circle,pad=0.2'))
        
        ax.set_aspect('equal')
        ax.set_title(f"{session.event.year} {session.event.EventName} - {session.name}", color='white')
        ax.axis('off')
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting circuit: {e}")
        return None

def plot_team_pace_comparison(session):
    """
    Generate a team pace comparison violin/box plot.
    """
    setup_f1_plotting()
    try:
        laps = session.laps.pick_quicklaps()
        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        
        teams = transformed_laps["Team"].unique()
        team_palette = {team: fastf1.plotting.get_team_color(team, session=session) for team in teams}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a list of data for each team
        data = []
        labels = []
        colors = []
        
        sorted_teams = sorted(teams, key=lambda t: transformed_laps[transformed_laps["Team"] == t]["LapTimeSeconds"].median())
        
        for team in sorted_teams:
            team_laps = transformed_laps[transformed_laps["Team"] == team]["LapTimeSeconds"].dropna()
            if len(team_laps) > 0:
                data.append(team_laps.values)
                labels.append(team)
                colors.append(team_palette.get(team, '#FFFFFF'))
        
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('white')
            pc.set_alpha(0.7)
            
        # Customize
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Lap Time (s)")
        ax.set_title("Team Pace Comparison", color='white')
        ax.grid(True, axis='y', alpha=0.1)
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting team pace: {e}")
        return None

def plot_tyre_strategy_summary(session):
    """
    Generate a tyre strategy summary plot.
    """
    setup_f1_plotting()
    try:
        laps = session.laps
        drivers = session.drivers
        drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]
        
        fig, ax = plt.subplots(figsize=(10, len(drivers)/2))
        
        driver_positions = {driver: i for i, driver in enumerate(drivers)}
        
        for driver in drivers:
            driver_laps = laps.pick_driver(driver)
            
            for _, stint in driver_laps.groupby("Stint"):
                compound = stint["Compound"].iloc[0]
                color = fastf1.plotting.get_compound_color(compound, session=session)
                
                start = stint["LapNumber"].min()
                end = stint["LapNumber"].max()
                
                ax.barh(driver, end-start, left=start, color=color, edgecolor="black", height=0.8)
        
        ax.set_xlabel("Lap Number")
        ax.invert_yaxis()
        ax.set_title("Tyre Strategy", color='white')
        
        # Legend
        from matplotlib.patches import Patch
        compounds = laps["Compound"].unique()
        legend_elements = [Patch(facecolor=fastf1.plotting.get_compound_color(c, session=session), 
                               label=c, edgecolor='black') for c in compounds if c]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting tyre strategy: {e}")
        return None

def plot_gear_shift_on_track(session, driver):
    """
    Generate a track map colored by gear usage for a specific driver.
    """
    setup_f1_plotting()
    try:
        lap = session.laps.pick_driver(driver).pick_fastest()
        tel = lap.get_telemetry()
        
        x = tel['X'].values
        y = tel['Y'].values
        gear = tel['nGear'].values
        
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (num_lines) x (points_per_line) x 2 (for x and y)
        import numpy as np
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a continuous norm to map from data points to colors
        cmap = plt.get_cmap('Paired')
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(1, 9))
        lc.set_array(gear)
        lc.set_linewidth(4)
        
        ax.add_collection(lc)
        ax.set_aspect('equal')
        ax.set_title(f"Gear Shift - {driver} ({session.event.EventName})", color='white')
        
        # Hide axes
        ax.axis('off')
        ax.set_xlim(x.min() - 1000, x.max() + 1000)
        ax.set_ylim(y.min() - 1000, y.max() + 1000)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax, ticks=np.arange(1, 9))
        cbar.ax.set_yticklabels(np.arange(1, 9))
        cbar.set_label("Gear")
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting gear shift: {e}")
        return None

def plot_speed_on_track(session, driver):
    """
    Generate a track map colored by speed for a specific driver.
    """
    setup_f1_plotting()
    try:
        lap = session.laps.pick_driver(driver).pick_fastest()
        tel = lap.get_telemetry()
        
        x = tel['X'].values
        y = tel['Y'].values
        speed = tel['Speed'].values
        
        import numpy as np
        from matplotlib.collections import LineCollection
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a continuous norm to map from data points to colors
        cmap = plt.get_cmap('inferno')
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(speed.min(), speed.max()))
        lc.set_array(speed)
        lc.set_linewidth(4)
        
        ax.add_collection(lc)
        ax.set_aspect('equal')
        ax.set_title(f"Speed Visualization - {driver} ({session.event.EventName})", color='white')
        
        # Hide axes
        ax.axis('off')
        ax.set_xlim(x.min() - 1000, x.max() + 1000)
        ax.set_ylim(y.min() - 1000, y.max() + 1000)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label("Speed (km/h)")
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting speed map: {e}")
        return None
