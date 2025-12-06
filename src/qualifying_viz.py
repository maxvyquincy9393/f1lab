import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('f1_visualization.qualifying')

def plot_qualifying_evolution(session):
    """
    Visualize lap time evolution throughout Q1, Q2, and Q3.
    """
    try:
        laps = session.laps
        
        # Filter for valid flying laps
        flying_laps = laps.pick_quicklaps().reset_index()
        
        if flying_laps.empty:
            return None
            
        # Create figure
        fig = go.Figure()
        
        # Q1, Q2, Q3 markers
        # Note: FastF1 splits sessions differently depending on API version/data availability.
        # Usually 'Split' column or time breaks define it. 
        # Simple approach: Plot all laps over time
        
        # Convert session time to minutes
        flying_laps['SessionTimeMins'] = flying_laps['LapStartTime'].dt.total_seconds() / 60
        
        # Color by team
        for team in flying_laps['Team'].unique():
            team_laps = flying_laps[flying_laps['Team'] == team]
            # Get team color (needs session or hardcoded map if not passed)
            # We'll assume standard plotting style handles colors or pass default
            
            fig.add_trace(go.Scatter(
                x=team_laps['SessionTimeMins'],
                y=team_laps['LapTime'].dt.total_seconds(),
                mode='markers',
                name=team,
                marker=dict(size=8, opacity=0.7),
                text=team_laps['Driver'],
                hovertemplate="<b>%{text}</b><br>Time: %{y:.3f}s<br>Session Time: %{x:.1f}m<extra></extra>"
            ))
            
        # Add Trendline (Track Evolution)
        # Fit a simple polynomial or moving average to show track improvement
        z = np.polyfit(flying_laps['SessionTimeMins'], flying_laps['LapTime'].dt.total_seconds(), 2)
        p = np.poly1d(z)
        x_trend = np.linspace(flying_laps['SessionTimeMins'].min(), flying_laps['SessionTimeMins'].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Track Evolution Trend',
            line=dict(color='white', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"Qualifying Lap Time Evolution - {session.event.EventName}",
            xaxis_title="Session Time (minutes)",
            yaxis_title="Lap Time (s)",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(autorange="reversed") # Faster times at top
        )
        
        return fig
    except Exception as e:
        logger.error(f"Qualifying evolution plot error: {e}")
        return None

def plot_qualifying_gap(session):
    """
    Bar chart of gaps to Pole Position.
    """
    try:
        drivers = pd.unique(session.laps['Driver'])
        list_fastest_laps = list()
        
        for drv in drivers:
            drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
            if not pd.isna(drvs_fastest_lap['LapTime']):
                list_fastest_laps.append(drvs_fastest_lap)
                
        fastest_laps = pd.DataFrame(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)
        
        pole_lap = fastest_laps.iloc[0]
        fastest_laps['Gap'] = fastest_laps['LapTime'] - pole_lap['LapTime']
        fastest_laps['GapSeconds'] = fastest_laps['Gap'].dt.total_seconds()
        
        fig = go.Figure()
        
        # Add bars
        colors = [] # logic to get colors
        # Simplification: Use simple color scale or team colors if available
        
        fig.add_trace(go.Bar(
            x=fastest_laps['GapSeconds'],
            y=fastest_laps['Driver'],
            orientation='h',
            text=[f"+{g:.3f}s" if g > 0 else "POLE" for g in fastest_laps['GapSeconds']],
            textposition='outside',
            marker_color='#E10600' # Default F1 Red
        ))
        
        fig.update_layout(
            title="Gap to Pole",
            xaxis_title="Gap (seconds)",
            height=max(500, len(fastest_laps)*25),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(autorange="reversed") 
        )
        
        return fig
    except Exception as e:
        logger.error(f"Qualifying gap plot error: {e}")
        return None

def plot_sector_dominance(session):
    """
    Mini-sectors or sector dominance map.
    Who was fastest in S1, S2, S3?
    """
    # For now, simple bar chart of sector times for top 10
    try:
        laps = session.laps.pick_quicklaps()
        drivers = pd.unique(laps['Driver'])
        
        sector_data = []
        for drv in drivers:
            best = laps.pick_driver(drv).pick_fastest()
            if not pd.isna(best['LapTime']):
                sector_data.append({
                    'Driver': drv,
                    'S1': best['Sector1Time'].total_seconds() if pd.notna(best['Sector1Time']) else 0,
                    'S2': best['Sector2Time'].total_seconds() if pd.notna(best['Sector2Time']) else 0,
                    'S3': best['Sector3Time'].total_seconds() if pd.notna(best['Sector3Time']) else 0,
                    'Total': best['LapTime'].total_seconds()
                })
                
        df = pd.DataFrame(sector_data).sort_values('Total').head(10)
        
        if df.empty:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Sector 1', x=df['Driver'], y=df['S1'], marker_color='#FF3333'))
        fig.add_trace(go.Bar(name='Sector 2', x=df['Driver'], y=df['S2'], marker_color='#00FF00'))
        fig.add_trace(go.Bar(name='Sector 3', x=df['Driver'], y=df['S3'], marker_color='#3333FF'))
        
        fig.update_layout(
            title="Top 10 - Sector Times Breakdown",
            barmode='stack',
            yaxis_title="Time (s)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    except Exception as e:
        logger.error(f"Sector dominance plot error: {e}")
        return None