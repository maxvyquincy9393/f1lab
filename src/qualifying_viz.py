"""
Qualifying Visualization Module.

Provides comprehensive qualifying analysis and visualization functions
with proper spacing and team color integration.

Author: F1 Analytics Team
Version: 2.0.0
"""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, List

try:
    from config import TEAM_COLORS, get_team_color
except ImportError:
    from src.config import TEAM_COLORS, get_team_color

# Configure module logger
logger = logging.getLogger('F1.QualifyingViz')


def calculate_qualifying_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time gaps from pole position in qualifying.
    
    Args:
        df: Qualifying results DataFrame with Position and Time columns.
        
    Returns:
        pd.DataFrame: Results with Gap_To_Pole column added.
    """
    try:
        logger.info("Calculating qualifying gaps...")
        
        if df is None or df.empty:
            logger.warning("Empty qualifying data")
            return pd.DataFrame()
        
        df = df.copy()
        
        # Convert position to numeric
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        
        # Sort by position
        df = df.sort_values('Position').reset_index(drop=True)
        
        # Get pole time (P1)
        pole_position = df[df['Position'] == 1]
        if pole_position.empty:
            logger.warning("No pole position found")
            return df
        
        # Calculate gaps to pole
        # This will be implemented based on actual time columns available
        df['Gap_To_Pole'] = 0.0  # Placeholder
        
        logger.info(f"Calculated gaps for {len(df)} drivers")
        return df
        
    except Exception as e:
        logger.exception(f"Error calculating qualifying gaps: {e}")
        return pd.DataFrame()


def plot_qualifying_gap_analysis(
    df: pd.DataFrame,
    session_name: str = "Qualifying"
) -> go.Figure:
    """
    Create qualifying gap analysis visualization with proper spacing.
    
    Args:
        df: Qualifying results DataFrame.
        session_name: Name of the qualifying session.
        
    Returns:
        Plotly Figure object.
    """
    try:
        logger.info(f"Creating qualifying gap analysis for {session_name}...")
        
        if df is None or df.empty:
            logger.warning("No data for qualifying gap analysis")
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No qualifying data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        # Sort by position
        df_plot = df.sort_values('Position').head(20)  # Top 20
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each driver with team colors
        for idx, row in df_plot.iterrows():
            team_color = get_team_color(row.get('Team', ''))
            
            fig.add_trace(go.Bar(
                x=[row.get('Gap_To_Pole', 0)],
                y=[row.get('Driver', f"P{row['Position']}")],
                orientation='h',
                marker=dict(
                    color=team_color,
                    line=dict(color='white', width=1)
                ),
                name=row.get('Driver', ''),
                showlegend=False,
                text=f"+{row.get('Gap_To_Pole', 0):.3f}s" if row.get('Gap_To_Pole', 0) > 0 else "POLE",
                textposition='outside'
            ))
        
        # Update layout with proper spacing
        fig.update_layout(
            title=dict(
                text=f"<b>{session_name} - Gap to Pole Position</b>",
                font=dict(size=20, color='white'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Gap to Pole (seconds)",
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                title="Driver",
                categoryorder='total ascending',
                tickfont=dict(size=12)
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            height=max(600, len(df_plot) * 30),  # Dynamic height with minimum
            margin=dict(l=150, r=150, t=80, b=80),  # Proper margins  
            hovermode='y unified'
        )
        
        logger.info(f"Created qualifying gap chart for {len(df_plot)} drivers")
        return fig
        
    except Exception as e:
        logger.exception(f"Error creating qualifying gap analysis: {e}")
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig


def plot_q1_q2_q3_progression(df: pd.DataFrame) -> go.Figure:
    """
    Create Q1/Q2/Q3 progression chart showing driver advancement.
    
    Args:
        df: Qualifying results with Q1, Q2, Q3 times.
        
    Returns:
        Plotly Figure object.
    """
    try:
        logger.info("Creating Q1/Q2/Q3 progression chart...")
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Q1 Results', 'Q2 Results', 'Q3 Results'),
            horizontal_spacing=0.12  # Proper spacing between panels
        )
        
        # Add Q1, Q2, Q3 data (placeholder - actual implementation depends on data structure)
        # This is a template that will be filled with actual data
        
        fig.update_layout(
            title=dict(
                text="<b>Qualifying Session Progression</b>",
                font=dict(size=20, color='white'),
                x=0.5,
                xanchor='center'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            showlegend=True,
            margin=dict(l=80, r=80, t=100, b=80)  # Proper spacing
        )
        
        logger.info("Created Q1/Q2/Q3 progression chart")
        return fig
        
    except Exception as e:
        logger.exception(f"Error creating Q1/Q2/Q3 progression: {e}")
        return go.Figure()


def plot_sector_time_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Create sector time comparison heatmap for qualifying.
    
    Args:
        df: Qualifying results with sector times.
        
    Returns:
        Plotly Figure object with proper spacing.
    """
    try:
        logger.info("Creating sector time comparison...")
        
        # Create heatmap or grouped bar chart
        fig = go.Figure()
        
        # Placeholder implementation - will be filled based on actual data structure
        
        fig.update_layout(
            title=dict(
                text="<b>Sector Time Comparison</b>",
                font=dict(size=20, color='white'),
                x=0.5,
                xanchor='center'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            margin=dict(l=100, r=100, t=80, b=80)  # Proper spacing
        )
        
        logger.info("Created sector time comparison")
        return fig
        
    except Exception as e:
        logger.exception(f"Error creating sector comparison: {e}")
        return go.Figure()


def create_qualifying_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted summary table for qualifying results.
    
    Args:
        df: Qualifying results DataFrame.
        
    Returns:
        Formatted DataFrame for display.
    """
    try:
        logger.info("Creating qualifying summary table...")
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Select key columns for display
        display_cols = ['Position', 'Driver', 'Team', 'Gap_To_Pole']
        available_cols = [col for col in display_cols if col in df.columns]
        
        summary = df[available_cols].copy()
        summary = summary.sort_values('Position').reset_index(drop=True)
        
        logger.info(f"Created summary table with {len(summary)} entries")
        return summary
        
    except Exception as e:
        logger.exception(f"Error creating summary table: {e}")
        return pd.DataFrame()
