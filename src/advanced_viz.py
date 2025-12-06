import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
import numpy as np
import pandas as pd
import logging

# Setup logging
logger = logging.getLogger('f1_visualization.advanced_viz')

def get_driver_color(driver, session):
    """Helper to get team color for a driver."""
    try:
        # Try getting from session info
        if hasattr(session, 'get_driver'):
            drv_info = session.get_driver(driver)
            team_name = drv_info['TeamName']
            # You might want to import TEAM_COLORS here or use fastf1
            return fastf1.plotting.get_team_color(team_name, session=session)
    except:
        pass
    return '#888888' # Default gray

def plot_telemetry_comparison(session, driver1, driver2, lap_number=None):
    """
    Compare telemetry (Speed, Throttle, Brake, Gear, RPM) between two drivers.
    Returns a Plotly Figure.
    """
    try:
        laps = session.laps
        
        # Select laps
        if lap_number:
            lap1 = laps.pick_driver(driver1).pick_lap(lap_number)
            lap2 = laps.pick_driver(driver2).pick_lap(lap_number)
            title_suffix = f"Lap {lap_number}"
        else:
            lap1 = laps.pick_driver(driver1).pick_fastest()
            lap2 = laps.pick_driver(driver2).pick_fastest()
            title_suffix = "Fastest Lap"
            
        if lap1.empty or lap2.empty:
            return None

        # Get telemetry
        tel1 = lap1.get_car_data().add_distance()
        tel2 = lap2.get_car_data().add_distance()
        
        # Colors
        color1 = get_driver_color(driver1, session)
        color2 = get_driver_color(driver2, session)
        
        # Create subplots
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Speed", "Throttle", "Brake", "Gear", "RPM"),
            row_heights=[0.3, 0.15, 0.15, 0.15, 0.25]
        )
        
        # 1. Speed
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Speed'],
            name=driver1, line=dict(color=color1), legendgroup=driver1
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Speed'],
            name=driver2, line=dict(color=color2), legendgroup=driver2
        ), row=1, col=1)
        
        # 2. Throttle
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Throttle'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Throttle'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2
        ), row=2, col=1)
        
        # 3. Brake
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Brake'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1,
            fill='tozeroy'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Brake'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2,
            fill='tozeroy', opacity=0.5
        ), row=3, col=1)
        
        # 4. Gear
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['nGear'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1,
            mode='lines' # Step line might be better but standard line is okay for now
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['nGear'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2
        ), row=4, col=1)
        
        # 5. RPM
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['RPM'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1
        ), row=5, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['RPM'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2
        ), row=5, col=1)

        # Delta time calculation (Approximation)
        # This is complex because distances don't match perfectly. 
        # We skip delta time for this iteration to keep it robust.

        fig.update_layout(
            title=f"Telemetry Comparison: {driver1} vs {driver2} ({title_suffix})",
            height=900,
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        fig.update_yaxes(title_text="km/h", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1, range=[0, 105])
        fig.update_yaxes(title_text="On/Off", row=3, col=1, range=[0, 1.1])
        fig.update_yaxes(title_text="Gear", row=4, col=1)
        fig.update_yaxes(title_text="RPM", row=5, col=1)
        fig.update_xaxes(title_text="Distance (m)", row=5, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting telemetry: {e}")
        return None

def plot_track_3d(session, driver=None):
    """
    3D visualization of the track using telemetry position data.
    Color-coded by speed.
    """
    try:
        laps = session.laps
        if driver:
            lap = laps.pick_driver(driver).pick_fastest()
        else:
            lap = laps.pick_fastest()
            
        if lap.empty:
            return None
            
        tel = lap.get_telemetry()
        x = tel['X']
        y = tel['Y']
        z = tel['Z']
        speed = tel['Speed']
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color=speed,
                colorscale='Viridis',
                width=5,
                colorbar=dict(title="Speed (km/h)")
            ),
            hovertext=[f"Speed: {s:.0f} km/h<br>Elev: {e:.1f}m" for s, e in zip(speed, z)],
            hoverinfo="text"
        )])
        
        fig.update_layout(
            title=f"3D Track Map - {session.event.EventName}",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=True, title="Elevation"),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.5)
                )
            ),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting 3D track: {e}")
        return None

def plot_gear_shift_trace(session, driver=None):
    """
    Gear shift distribution and shift points.
    """
    try:
        laps = session.laps
        if driver:
            lap = laps.pick_driver(driver).pick_fastest()
        else:
            lap = laps.pick_fastest()
            
        tel = lap.get_telemetry()
        
        # Calculate gear shifts
        tel['GearChange'] = tel['nGear'].diff().fillna(0)
        shifts = tel[tel['GearChange'] != 0]
        
        fig = go.Figure()
        
        # Speed trace background
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['Speed'],
            name='Speed',
            line=dict(color='gray', width=1),
            opacity=0.5
        ))
        
        # Upshifts
        upshifts = shifts[shifts['GearChange'] > 0]
        fig.add_trace(go.Scatter(
            x=upshifts['Distance'], y=upshifts['Speed'],
            mode='markers',
            name='Upshift',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
        
        # Downshifts
        downshifts = shifts[shifts['GearChange'] < 0]
        fig.add_trace(go.Scatter(
            x=downshifts['Distance'], y=downshifts['Speed'],
            mode='markers',
            name='Downshift',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
        
        fig.update_layout(
            title=f"Gear Shift Points - {lap['Driver']}",
            xaxis_title="Distance (m)",
            yaxis_title="Speed (km/h)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting gear shifts: {e}")
        return None
