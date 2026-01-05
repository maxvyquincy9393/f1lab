# -*- coding: utf-8 -*-
"""
race_replay_viz.py
~~~~~~~~~~~~~~~~~~
Race replay visualization components for Streamlit using Plotly.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Team colors fallback
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'Racing Bulls': '#6692FF',
    'Kick Sauber': '#52E252',
    'Haas': '#B6BABD',
}


def create_track_figure(track_coords: dict, rotation: float = 0.0) -> go.Figure:
    """Create base track layout figure."""
    if track_coords is None:
        return None
    
    x = track_coords['x']
    y = track_coords['y']
    
    # Apply rotation if needed
    if rotation != 0:
        angle_rad = np.radians(rotation)
        cos_r, sin_r = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = x * cos_r - y * sin_r
        y_rot = x * sin_r + y * cos_r
        x, y = x_rot, y_rot
    
    fig = go.Figure()
    
    # Track outline
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#444444', width=25),
        hoverinfo='skip',
        name='Track Edge'
    ))
    
    # Track surface
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#333333', width=20),
        hoverinfo='skip',
        name='Track'
    ))
    
    # Center line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#555555', width=1, dash='dot'),
        hoverinfo='skip',
        name='Center Line'
    ))
    
    return fig


def create_replay_animation(
    frames: list,
    track_coords: dict,
    driver_colors: dict,
    total_laps: int,
    rotation: float = 0.0,
    frame_skip: int = 5
) -> go.Figure:
    """
    Create animated race replay figure.
    
    Args:
        frames: List of frame dicts with driver positions
        track_coords: Track X/Y coordinates
        driver_colors: Color mapping for drivers
        total_laps: Total race laps
        rotation: Track rotation angle
        frame_skip: Skip frames for performance (1 = all frames)
    """
    if not frames or track_coords is None:
        return None
    
    # Apply rotation to track
    x_track = track_coords['x']
    y_track = track_coords['y']
    
    if rotation != 0:
        angle_rad = np.radians(rotation)
        cos_r, sin_r = np.cos(angle_rad), np.sin(angle_rad)
        x_track = track_coords['x'] * cos_r - track_coords['y'] * sin_r
        y_track = track_coords['x'] * sin_r + track_coords['y'] * cos_r
    
    # Subsample frames for performance
    sampled_frames = frames[::frame_skip]
    
    # Create base figure with track
    fig = go.Figure()
    
    # Track background
    fig.add_trace(go.Scatter(
        x=x_track, y=y_track,
        mode='lines',
        line=dict(color='#333333', width=20),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Get initial frame data
    initial_frame = sampled_frames[0] if sampled_frames else None
    if initial_frame is None:
        return fig
    
    # Extract driver data for initial frame
    drivers_data = initial_frame.get('drivers', {})
    
    x_drivers = []
    y_drivers = []
    colors = []
    texts = []
    
    for code, data in drivers_data.items():
        x_pos = data['x']
        y_pos = data['y']
        
        if rotation != 0:
            x_pos = data['x'] * cos_r - data['y'] * sin_r
            y_pos = data['x'] * sin_r + data['y'] * cos_r
        
        x_drivers.append(x_pos)
        y_drivers.append(y_pos)
        colors.append(driver_colors.get(code, '#FFFFFF'))
        texts.append(code)
    
    # Add initial driver positions
    fig.add_trace(go.Scatter(
        x=x_drivers,
        y=y_drivers,
        mode='markers+text',
        marker=dict(
            size=20,
            color=colors,
            line=dict(width=2, color='white')
        ),
        text=texts,
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    # Create animation frames
    animation_frames = []
    
    for idx, frame in enumerate(sampled_frames):
        drivers_data = frame.get('drivers', {})
        
        x_anim = []
        y_anim = []
        colors_anim = []
        texts_anim = []
        hovers = []
        
        for code, data in drivers_data.items():
            x_pos = data['x']
            y_pos = data['y']
            
            if rotation != 0:
                x_pos = data['x'] * cos_r - data['y'] * sin_r
                y_pos = data['x'] * sin_r + data['y'] * cos_r
            
            x_anim.append(x_pos)
            y_anim.append(y_pos)
            colors_anim.append(driver_colors.get(code, '#FFFFFF'))
            texts_anim.append(code)
            hovers.append(f"P{data['position']}: {code}<br>Lap {data['lap']}<br>{data['speed']:.0f} km/h")
        
        leader_lap = frame.get('leader_lap', 0)
        race_time = frame.get('time', 0)
        mins = int(race_time // 60)
        secs = int(race_time % 60)
        
        animation_frames.append(go.Frame(
            data=[
                go.Scatter(x=x_track, y=y_track),  # Track (unchanged)
                go.Scatter(
                    x=x_anim,
                    y=y_anim,
                    marker=dict(size=20, color=colors_anim, line=dict(width=2, color='white')),
                    text=texts_anim,
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hovers
                )
            ],
            name=str(idx),
            layout=go.Layout(
                title=dict(
                    text=f"Lap {leader_lap}/{total_laps} | {mins:02d}:{secs:02d}",
                    font=dict(size=16)
                )
            )
        ))
    
    fig.frames = animation_frames
    
    # Add animation controls
    fig.update_layout(
        title=dict(
            text=f"Lap 1/{total_laps} | 00:00",
            font=dict(size=16, color='white'),
            x=0.5,
            xanchor='center'
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                 "label": str(i * frame_skip),
                 "method": "animate"}
                for i in range(len(sampled_frames))
            ]
        }],
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        height=700,
        margin=dict(l=20, r=20, t=60, b=60)
    )
    
    return fig


def create_leaderboard_table(frame_data: dict, driver_colors: dict) -> go.Figure:
    """Create a leaderboard table for current frame."""
    if not frame_data:
        return None
    
    # Sort by position
    sorted_drivers = sorted(frame_data.items(), key=lambda x: x[1]['position'])
    
    positions = []
    drivers = []
    laps = []
    speeds = []
    colors = []
    
    for code, data in sorted_drivers[:10]:  # Top 10
        positions.append(f"P{data['position']}")
        drivers.append(code)
        laps.append(f"L{data['lap']}")
        speeds.append(f"{data['speed']:.0f}")
        colors.append(driver_colors.get(code, '#FFFFFF'))
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Pos', 'Driver', 'Lap', 'Speed'],
            fill_color='#1a1a2e',
            font=dict(color='white', size=12),
            align='center',
            height=30
        ),
        cells=dict(
            values=[positions, drivers, laps, speeds],
            fill_color='#0E1117',
            font=dict(color='white', size=11),
            align='center',
            height=25
        )
    )])
    
    fig.update_layout(
        paper_bgcolor='#0E1117',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    
    return fig


def create_telemetry_gauges(driver_data: dict, driver_color: str = '#E10600') -> go.Figure:
    """Create speed and gear gauges for selected driver."""
    speed = driver_data.get('speed', 0)
    gear = driver_data.get('gear', 0)
    position = driver_data.get('position', 0)
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        column_widths=[0.4, 0.3, 0.3]
    )
    
    # Speed gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=speed,
        title={'text': "Speed (km/h)", 'font': {'size': 12, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 360], 'tickcolor': 'white'},
            'bar': {'color': driver_color},
            'bgcolor': '#1a1a2e',
            'borderwidth': 2,
            'bordercolor': 'white',
        },
        number={'font': {'color': 'white', 'size': 24}}
    ), row=1, col=1)
    
    # Gear indicator
    fig.add_trace(go.Indicator(
        mode="number",
        value=gear,
        title={'text': "Gear", 'font': {'size': 12, 'color': 'white'}},
        number={'font': {'color': driver_color, 'size': 48}}
    ), row=1, col=2)
    
    # Position indicator
    fig.add_trace(go.Indicator(
        mode="number",
        value=position,
        title={'text': "Position", 'font': {'size': 12, 'color': 'white'}},
        number={'font': {'color': 'white', 'size': 48}, 'prefix': 'P'}
    ), row=1, col=3)
    
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_static_replay_frame(
    frame: dict,
    track_coords: dict,
    driver_colors: dict,
    total_laps: int,
    rotation: float = 0.0
) -> go.Figure:
    """Create a single static frame for manual stepping."""
    if not frame or track_coords is None:
        return None
    
    # Apply rotation
    x_track = track_coords['x']
    y_track = track_coords['y']
    
    cos_r, sin_r = 1, 0
    if rotation != 0:
        angle_rad = np.radians(rotation)
        cos_r, sin_r = np.cos(angle_rad), np.sin(angle_rad)
        x_track = track_coords['x'] * cos_r - track_coords['y'] * sin_r
        y_track = track_coords['x'] * sin_r + track_coords['y'] * cos_r
    
    fig = go.Figure()
    
    # Track
    fig.add_trace(go.Scatter(
        x=x_track, y=y_track,
        mode='lines',
        line=dict(color='#333333', width=20),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Drivers
    drivers_data = frame.get('drivers', {})
    
    for code, data in drivers_data.items():
        x_pos = data['x']
        y_pos = data['y']
        
        if rotation != 0:
            x_pos = data['x'] * cos_r - data['y'] * sin_r
            y_pos = data['x'] * sin_r + data['y'] * cos_r
        
        color = driver_colors.get(code, '#FFFFFF')
        
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[y_pos],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(width=2, color='white')),
            text=[code],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            hovertemplate=f"P{data['position']}: {code}<br>Lap {data['lap']}<br>{data['speed']:.0f} km/h<extra></extra>",
            showlegend=False
        ))
    
    leader_lap = frame.get('leader_lap', 0)
    race_time = frame.get('time', 0)
    mins = int(race_time // 60)
    secs = int(race_time % 60)
    
    fig.update_layout(
        title=dict(
            text=f"Lap {leader_lap}/{total_laps} | {mins:02d}:{secs:02d}",
            font=dict(size=18, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='white'),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig
