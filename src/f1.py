# -*- coding: utf-8 -*-
"""
app.py
~~~~~~
Streamlit dashboard entry point.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from config import TEAM_COLORS, DRIVER_PROFILES, F1_2025_COMPLETED_RACES, F1_2025_CALENDAR, F1_2025_RACE_NAMES, STREAMLIT_CONFIG, SOCIAL_MEDIA_CONFIG
from loader import load_data as load_csv_data, clean_data, load_combined_data
from analysis import calculate_driver_stats, calculate_team_stats, calculate_combined_constructor_standings, calculate_teammate_comparison
from config import DATA_FILES
from fastf1_extended import (
    get_session_info, get_weather_summary, get_tyre_stints,
    get_pit_stops, get_sector_times, get_speed_data,
    get_track_status, get_car_data, get_position_data,
    get_position_changes, get_flag_events,
    get_race_results, get_session_schedule, get_gaps_to_leader,
    get_tyre_degradation, get_best_sectors, get_top_speeds,
    export_session_to_csv, get_circuit_layout_info,
    get_race_control_messages, get_detailed_pit_analysis
)
from model import load_trained_model, RaceStrategySimulator
from features import prepare_features
from fastf1_plotting import (
    plot_circuit_with_corners, plot_team_pace_comparison, 
    plot_tyre_strategy_summary, plot_gear_shift_on_track,
    plot_speed_on_track
)
from advanced_viz import plot_telemetry_comparison, plot_track_3d, plot_gear_shift_trace, plot_corner_performance, plot_tyre_shape, plot_circuit_context
from qualifying_viz import plot_qualifying_evolution, plot_qualifying_gap, plot_sector_dominance
from home import render_home_tab
from race_replay_data import get_race_replay_frames, get_circuit_rotation, format_race_time
from race_replay_viz import create_replay_animation, create_static_replay_frame, create_leaderboard_table, create_telemetry_gauges
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title=STREAMLIT_CONFIG.page_title,
    page_icon=STREAMLIT_CONFIG.page_icon,
    layout=STREAMLIT_CONFIG.layout,
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None # You can put custom text here or None to hide
    }
)

# Initialize session state for navigation persistence
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0
if 'driver_profile_selection' not in st.session_state:
    st.session_state.driver_profile_selection = None
if 'team_profile_selection' not in st.session_state:
    st.session_state.team_profile_selection = None
if 'race_detail_selection' not in st.session_state:
    st.session_state.race_detail_selection = None

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Aggressive Height Reduction for Header */
    header {visibility: hidden !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    
    /* Hide specific deploy button */
    /* Hide specific deploy button and badges */
    .stDeployButton {display: none !important;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    div[class*="stAppDeployButton"] {display: none !important;}
    
    /* Hide Fork button and GitHub link */
    .viewerBadge_container__1QSob {display: none;}
    .viewerBadge_link__1S137 {display: none;}
    button[title="View source on GitHub"] {display: none;}
    a[href*="github.com"] {display: none !important;}
    
    /* Hide Streamlit logo/branding bottom right */
    .stDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    button[kind="header"] {display: none;}
    .css-18ni7ap {display: none;}
    .css-1dp5vir {display: none;}
    
    /* Hide floating action buttons - ALL VARIANTS */
    .st-emotion-cache-1gulkj5 {display: none !important;}
    .st-emotion-cache-1wmy9hl {display: none !important;}
    [data-testid="stChatActionButtonIcon"] {display: none !important;}
    
    /* Hide bottom right floating buttons */
    div[data-testid="stBottomBlockContainer"] {display: none !important;}
    .stActionButton {display: none !important;}
    button[data-testid="stActionButton"] {display: none !important;}
    
    /* Hide ALL bottom right corner elements */
    div[class*="fixed"] {display: none !important;}
    div[style*="position: fixed"][style*="bottom"] {display: none !important;}
    div[style*="position: fixed"][style*="right"] {display: none !important;}
    
    /* Target the specific buttons in screenshot */
    button[aria-label*="Location"] {display: none !important;}
    button[kind="primaryFormSubmit"] {display: none !important;}
    div.stChatFloatingButtonContainer {display: none !important;}
    
    /* Hide streamlit branding */
    ._profileContainer_gzau3_53 {display: none !important;}
    ._profilePreview_gzau3_63 {display: none !important;}
    
    /* Custom styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E10600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .driver-card {
        background: #15151e;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .driver-photo {
        border-radius: 50%;
        border: 3px solid #E10600;
    }
    .stat-box {
        background: #1f1f2e;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
    .weather-box {
        background: linear-gradient(135deg, #0f3460 0%, #16537e 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    .pit-stop-row {
        background: #1a1a2e;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .gauge-container {
        text-align: center;
        padding: 10px;
    }
    .telemetry-panel {
        background: #0a0a0f;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


def setup_fastf1_cache():
    """Setup FastF1 cache directory."""
    cache_dir = Path("./f1_cache")
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def show_plotly_chart(fig, use_container_width=True, **kwargs):
    """Display Plotly chart with hidden toolbar and logo."""
    config = {
        'displayModeBar': False,  # Hide the entire toolbar
        'displaylogo': False,     # Hide Plotly logo
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 
                                   'autoScale2d', 'resetScale2d', 'toImage'],
        'staticPlot': False       # Keep it interactive
    }
    st.plotly_chart(fig, use_container_width=use_container_width, config=config, **kwargs)


@st.cache_data(ttl=3600)
def load_race_data():
    """Load and cache race data with combined race + sprint points."""
    try:
        # Load combined data (race + sprint) - pass the data directory
        data_dir = str(Path(DATA_FILES.race_results).parent)
        df = load_combined_data(data_dir)
        if df is not None:
            df = clean_data(df)
        return df
    except Exception as e:
        logger.error(f"Error loading combined data: {e}")
        # Fallback to race data only
        try:
            df = load_csv_data(str(DATA_FILES.race_results))
            if df is not None:
                df = clean_data(df)
            return df
        except Exception as e2:
            logger.error(f"Error loading race data: {e2}")
            return None


@st.cache_data(ttl=3600)
def load_sprint_data():
    """Load and cache sprint data."""
    try:
        sprint_file = Path(DATA_FILES.race_results).parent / 'Formula1_2025Season_SprintResults.csv'
        if sprint_file.exists():
            df = load_csv_data(str(sprint_file))
            if df is not None:
                df = clean_data(df)
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading sprint data: {e}")
        return None


def load_fastf1_session(year: int, race: str, session_type: str, load_telemetry: bool = False):
    """Load FastF1 session with caching and optional telemetry."""
    try:
        session = fastf1.get_session(year, race, session_type)
        if load_telemetry:
            session.load()
        else:
            session.load(telemetry=False, laps=True, weather=True)
        return session
    except Exception as e:
        logger.error(f"Error loading FastF1 session: {e}")
        return None


@st.cache_data(ttl=3600)
def get_real_grid_positions(year: int, race: str) -> dict:
    """
    Fetch actual Qualifying grid positions from FastF1.
    """
    try:
        # Try Loading Qualifying
        session = fastf1.get_session(year, race, 'Q')
        # Light load (no telemetry)
        session.load(telemetry=False, laps=False, weather=False)
        
        if session.results is None or session.results.empty:
            return {}
            
        # Create mapping: Driver -> GridPosition
        # Ensure GridPosition is valid (not 0.0 unless pole?)
        # 0.0 usually means pit lane or DQ? No, 0.0 is unclassified?
        # Pole is 1.0. FastF1 uses 0.0 for missing?
        grid_map = {}
        for drv in session.results['Abbreviation']:
            res = session.results.loc[session.results['Abbreviation'] == drv].iloc[0]
            grid = res['GridPosition']
            if grid <= 0:
                # Fallback to Position if Grid is 0 (e.g. penalities applied later, or just use finish position of Q)
                grid = res['Position']
            grid_map[drv] = grid
            
        return grid_map
    except Exception as e:
        logger.warning(f"Could not fetch real grid for {race}: {e}")
        return {}


def create_gauge(value, max_value, title, color="#E10600"):
    """Create a gauge chart for telemetry display."""
    # Safely convert value to float
    try:
        if value is None:
            value = 0.0
        elif isinstance(value, (bool, np.bool_)):
            value = 100.0 if value else 0.0
        elif hasattr(value, 'item'):
            value = float(value.item())
        elif isinstance(value, (np.integer, np.floating)):
            value = float(value)
        else:
            value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    
    # Ensure max_value is valid
    try:
        max_value = float(max_value) if max_value and max_value > 0 else 100.0
    except:
        max_value = 100.0
    
    # Clamp value to valid range
    value = max(0.0, min(float(value), float(max_value)))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "gray",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, max_value * 0.6], 'color': '#1a1a2e'},
                {'range': [max_value * 0.6, max_value * 0.85], 'color': '#2a2a4e'},
                {'range': [max_value * 0.85, max_value], 'color': '#3a3a6e'}
            ],
        },
        number={'font': {'color': 'white'}}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def format_f1_time(td: pd.Timedelta) -> str:
    """Formats a pandas Timedelta object into an F1-style lap time string (M:SS.mmm or H:MM:SS.mmm)."""
    if pd.isna(td):
        return "N/A"
    
    total_seconds = td.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes:d}:{seconds:06.3f}"


def render_race_replay_tab():
    """Race Replay visualization tab with animated track and leaderboard."""
    st.subheader("Race Replay")
    st.markdown("Watch the race unfold with animated driver positions on track.")
    
    # Initialize session state for replay
    if 'replay_data' not in st.session_state:
        st.session_state.replay_data = None
    if 'replay_frame_idx' not in st.session_state:
        st.session_state.replay_frame_idx = 0
    if 'selected_replay_driver' not in st.session_state:
        st.session_state.selected_replay_driver = None
    
    # Race selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_race = st.selectbox(
            "Select Race",
            F1_2025_COMPLETED_RACES,
            key="replay_race_select"
        )
    
    with col2:
        session_type = st.selectbox(
            "Session",
            ["Race", "Sprint"],
            key="replay_session_type"
        )
    
    with col3:
        frame_skip = st.slider(
            "Quality",
            min_value=1,
            max_value=20,
            value=5,
            help="Lower = more frames, smoother animation but slower"
        )
    
    # Load buttons - side by side
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("Load Web Replay", type="primary", key="load_replay_btn"):
            with st.spinner("Loading telemetry data... This may take 1-2 minutes for first load."):
                try:
                    setup_fastf1_cache()
                    session_code = 'S' if session_type == "Sprint" else 'R'
                    session = load_fastf1_session(2025, selected_race, session_code, load_telemetry=True)
                    
                    if session is not None:
                        replay_data = get_race_replay_frames(session, session_type=session_code)
                        st.session_state.replay_data = replay_data
                        st.session_state.replay_frame_idx = 0
                        st.success(f"Loaded {len(replay_data['frames'])} frames for {selected_race}")
                    else:
                        st.error("Could not load session data. Make sure the race has been completed.")
                except Exception as e:
                    st.error(f"Error loading replay: {e}")
                    logger.exception("Replay load error")
    
    with btn_col2:
        if st.button("🖥️ Launch Desktop Replay", type="secondary", key="launch_desktop_btn"):
            import subprocess
            import sys
            
            session_code = 'S' if session_type == "Sprint" else 'R'
            script_path = Path(__file__).parent / "arcade_replay_window.py"
            
            try:
                # Spawn desktop replay window as separate process
                subprocess.Popen([
                    sys.executable,
                    str(script_path),
                    "--year", "2025",
                    "--race", selected_race,
                    "--session", session_code
                ], cwd=str(Path(__file__).parent.parent))
                
                st.success("Desktop replay window launching... (Check your taskbar)")
                st.info("Controls: SPACE=Play/Pause, ←→=Seek, ↑↓=Speed, 1-4=Quick Speed, R=Reset, ESC=Close")
            except Exception as e:
                st.error(f"Could not launch desktop replay: {e}")
    
    # Display replay if data is loaded
    if st.session_state.replay_data is not None:
        data = st.session_state.replay_data
        frames = data['frames']
        track = data['track']
        colors = data['driver_colors']
        total_laps = data['total_laps']
        
        st.divider()
        
        # Mode selection
        replay_mode = st.radio(
            "Replay Mode",
            ["Animated", "Manual Step"],
            horizontal=True,
            key="replay_mode"
        )
        
        if replay_mode == "Animated":
            # Animated replay with Plotly
            st.markdown("**Use the Play/Pause buttons and slider below the track to control playback.**")
            
            try:
                rotation = get_circuit_rotation(load_fastf1_session(2025, selected_race, 'R', load_telemetry=False)) if track else 0
            except:
                rotation = 0
            
            fig = create_replay_animation(
                frames=frames,
                track_coords=track,
                driver_colors=colors,
                total_laps=total_laps,
                rotation=rotation,
                frame_skip=frame_skip
            )
            
            if fig:
                config = {
                    'displayModeBar': False,
                    'displaylogo': False,
                }
                st.plotly_chart(fig, use_container_width=True, config=config)
            else:
                st.warning("Could not create animation. Track data may be unavailable.")
        
        else:
            # Manual stepping mode
            col_track, col_info = st.columns([2, 1])
            
            with col_track:
                # Frame slider
                frame_idx = st.slider(
                    "Race Progress",
                    min_value=0,
                    max_value=len(frames) - 1,
                    value=st.session_state.replay_frame_idx,
                    key="frame_slider"
                )
                st.session_state.replay_frame_idx = frame_idx
                
                current_frame = frames[frame_idx]
                
                try:
                    rotation = get_circuit_rotation(load_fastf1_session(2025, selected_race, 'R', load_telemetry=False)) if track else 0
                except:
                    rotation = 0
                
                fig = create_static_replay_frame(
                    frame=current_frame,
                    track_coords=track,
                    driver_colors=colors,
                    total_laps=total_laps,
                    rotation=rotation
                )
                
                if fig:
                    config = {'displayModeBar': False, 'displaylogo': False}
                    st.plotly_chart(fig, use_container_width=True, config=config)
            
            with col_info:
                # Leaderboard
                st.markdown("**Live Standings**")
                drivers_data = current_frame.get('drivers', {})
                
                if drivers_data:
                    leaderboard_fig = create_leaderboard_table(drivers_data, colors)
                    if leaderboard_fig:
                        st.plotly_chart(leaderboard_fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Driver selection for telemetry
                    st.markdown("**Driver Telemetry**")
                    driver_codes = sorted(drivers_data.keys(), key=lambda d: drivers_data[d]['position'])
                    selected_driver = st.selectbox(
                        "Select Driver",
                        driver_codes,
                        key="telemetry_driver_select"
                    )
                    
                    if selected_driver and selected_driver in drivers_data:
                        driver_data = drivers_data[selected_driver]
                        color = colors.get(selected_driver, '#E10600')
                        
                        telemetry_fig = create_telemetry_gauges(driver_data, color)
                        if telemetry_fig:
                            st.plotly_chart(telemetry_fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Select a race and click 'Load Race Replay' to start.")


def render_header():
    """Render dashboard header."""
    st.markdown('<h1 class="main-header">F1 2025 Season Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analytics | Telemetry | Predictions</p>', unsafe_allow_html=True)


def render_overview_tab(df, total_points_combined=None):
    """Season Overview tab content."""
    st.header("2025 Season Overview")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Calculate total laps from race data
    total_laps = df['Laps'].sum() if 'Laps' in df.columns else 0
    total_all_points = sum(total_points_combined.values()) if total_points_combined else df['Points'].sum()
    
    # Season metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_races = df['Track'].nunique()
    total_drivers = df['Driver'].nunique()
    total_teams = df['Team'].nunique()
    
    with col1:
        st.metric("Races Completed", total_races)
    with col2:
        st.metric("Drivers", total_drivers)
    with col3:
        st.metric("Teams", total_teams)
    with col4:
        st.metric("Total Race Laps", f"{total_laps:,}")
    with col5:
        st.metric("Total Points", f"{int(total_all_points):,}")
    
    st.divider()
    
    # Championship standings with custom display options
    st.subheader("Championship Standings")
    
    # Display options
    col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 2])
    with col_opt1:
        num_drivers_display = st.slider("Drivers to show", min_value=5, max_value=20, value=10, key="num_drivers")
    with col_opt2:
        num_teams_display = st.slider("Teams to show", min_value=5, max_value=10, value=10, key="num_teams")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Drivers Championship**")
        driver_stats = calculate_driver_stats(df)
        if driver_stats is not None and not driver_stats.empty:
            # Reset index to get Driver as column
            driver_stats_display = driver_stats.reset_index()
            top_drivers = driver_stats_display.nlargest(num_drivers_display, 'Total_Points')
            
            fig = go.Figure()
            # Get team for each driver from original df
            driver_teams = df.groupby('Driver')['Team'].first().to_dict()
            colors = [TEAM_COLORS.get(driver_teams.get(drv, ''), '#666666') for drv in top_drivers['Driver']]
            
            fig.add_trace(go.Bar(
                x=top_drivers['Total_Points'],
                y=top_drivers['Driver'],
                orientation='h',
                marker_color=colors,
                text=top_drivers['Total_Points'].astype(int),
                textposition='outside'
            ))
            fig.update_layout(
                height=max(350, num_drivers_display * 35),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Points",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=10, r=120, t=10, b=40)
            )
            show_plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Constructors Championship**")
        # Load sprint data and calculate combined standings
        df_sprint = load_sprint_data()
        if df_sprint is not None:
            constructor_standings = calculate_combined_constructor_standings(df, df_sprint)
        else:
            # Fallback to race-only stats
            constructor_standings = calculate_team_stats(df)
        
        if constructor_standings is not None and not constructor_standings.empty:
            # Reset index to get Team as column
            team_stats_display = constructor_standings.reset_index()
            team_stats_display = team_stats_display.sort_values('Total_Points', ascending=False).head(num_teams_display)
            
            fig = go.Figure()
            colors = [TEAM_COLORS.get(team, '#666666') for team in team_stats_display['Team']]
            
            fig.add_trace(go.Bar(
                x=team_stats_display['Total_Points'],
                y=team_stats_display['Team'],
                orientation='h',
                marker_color=colors,
                text=team_stats_display['Total_Points'].astype(int),
                textposition='outside'
            ))
            fig.update_layout(
                height=max(350, num_teams_display * 40),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Points",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=10, r=120, t=10, b=40)
            )
            show_plotly_chart(fig, use_container_width=True)
    
    # Points progression
    st.subheader("Championship Points Progression")
    
    races = df['Track'].unique()
    # Use driver_stats with reset index
    top_5_drivers = driver_stats_display.nlargest(5, 'Total_Points')['Driver'].tolist() if driver_stats is not None else []
    
    if top_5_drivers:
        fig = go.Figure()
        
        for driver in top_5_drivers:
            driver_data = df[df['Driver'] == driver].copy()
            driver_data = driver_data.sort_values('Track')
            cumsum_points = driver_data['Points'].cumsum()
            team = driver_data['Team'].iloc[0] if not driver_data.empty else ''
            color = TEAM_COLORS.get(team, '#666666')
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumsum_points) + 1)),
                y=cumsum_points,
                name=driver,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            height=400,
            xaxis_title="Race Number",
            yaxis_title="Cumulative Points",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        show_plotly_chart(fig, use_container_width=True)


def render_drivers_tab(df, total_points_combined=None):
    """Driver Profiles tab content."""
    st.header("Driver Profiles")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    drivers_2025 = sorted([d for d in df['Driver'].unique().tolist() if d in DRIVER_PROFILES])
    
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_driver = st.selectbox("Select Driver", drivers_2025, key="driver_profile_selector")
    
    if selected_driver:
        profile = DRIVER_PROFILES.get(selected_driver, {})
        driver_df = df[df['Driver'] == selected_driver]
        team = driver_df['Team'].iloc[0] if not driver_df.empty else "Unknown"
        team_color = TEAM_COLORS.get(team, '#666666')
        
        # --- Hero Section ---
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {team_color} 0%, #0E1117 100%); padding: 2px; border-radius: 10px; margin-bottom: 20px;">
            <div style="background: #0E1117; border-radius: 8px; padding: 20px;">
                <h1 style="color: white; margin: 0; font-size: 3rem; text-transform: uppercase; letter-spacing: 2px;">
                    <span style="color: {team_color};">{profile.get('number', '')}</span> {selected_driver}
                </h1>
                <h3 style="color: #888; margin: 0; font-weight: 300;">{team}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Profile Content ---
        col_profile, col_stats = st.columns([1, 2])
        
        with col_profile:
            if profile.get('image_url'):
                st.image(profile['image_url'], use_container_width=True)
            else:
                st.warning("Driver image not available")
                
            # Biographical Data Table
            st.markdown("### Biography")
            st.markdown(f"""
            **Nationality:** {profile.get('country', 'N/A')}  
            **Debut:** {profile.get('debut', 'N/A')}  
            **Seasons:** {2025 - profile.get('debut', 2025) if isinstance(profile.get('debut'), int) else 'N/A'}
            """)
            st.info(profile.get('bio', ''))

            # Social Media
            st.markdown("### Connect")
            cols_social = st.columns(len(SOCIAL_MEDIA_CONFIG))
            for idx, (platform, conf) in enumerate(SOCIAL_MEDIA_CONFIG.items()):
                handle = profile.get(platform)
                if handle:
                    url = f"{conf['url_prefix']}{handle.replace('@', '')}"
                    with cols_social[idx]:
                         st.markdown(f"[{platform.capitalize()}]({url})")
            
            # Career Stats
            if 'titles' in profile:
                st.markdown("### Career Highlights")
                c_data = {
                    "Metric": ["World Titles", "Grand Prix Wins", "Pole Positions", "Podiums"],
                    "Value": [
                        profile.get('titles', 0),
                        profile.get('wins', 0),
                        profile.get('poles', 0),
                        profile.get('podiums', 0)
                    ]
                }
                st.dataframe(pd.DataFrame(c_data), hide_index=True, use_container_width=True)

        with col_stats:
            # 2025 Season Performance
            st.subheader("2025 Season Performance")
            
            # Stats Calculation
            race_points = driver_df['Points'].sum()
            total_points = total_points_combined.get(selected_driver, race_points) if total_points_combined else race_points
            wins = len(driver_df[driver_df['Position'] == 1])
            podiums = len(driver_df[driver_df['Position'] <= 3])
            avg_pos = driver_df['Position'].mean()
            best_finish = driver_df['Position'].min() if not driver_df.empty else 0
            
            # Custom Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"<h2 style='text-align:center; color:{team_color}'>{int(total_points)}</h2><p style='text-align:center'>POINTS</p>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<h2 style='text-align:center; color:white'>{wins}</h2><p style='text-align:center'>WINS</p>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<h2 style='text-align:center; color:white'>{podiums}</h2><p style='text-align:center'>PODIUMS</p>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<h2 style='text-align:center; color:white'>P{int(best_finish) if not pd.isna(best_finish) else '-'}</h2><p style='text-align:center'>BEST</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Recent Form Chart
            st.subheader("Recent Form")
            
            fig = go.Figure()
            
            # Position Trend
            fig.add_trace(go.Scatter(
                x=driver_df['Track'],
                y=driver_df['Position'],
                mode='lines+markers',
                name='Finish Position',
                line=dict(color=team_color, width=3),
                marker=dict(size=10, color='white', line=dict(width=2, color=team_color))
            ))
            
            # Add Qualifying comparison if available (simplified)
            if 'Starting Grid' in driver_df.columns:
                fig.add_trace(go.Scatter(
                    x=driver_df['Track'],
                    y=driver_df['Starting Grid'],
                    mode='markers',
                    name='Grid Position',
                    marker=dict(size=8, symbol='diamond', color='#888888')
                ))
            
            fig.update_layout(
                yaxis=dict(autorange='reversed', title='Position', gridcolor='#333'),
                xaxis=dict(showgrid=False),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            show_plotly_chart(fig, use_container_width=True)
            
            # Results Table
            with st.expander("Full 2025 Results"):
                res_cols = ['Track', 'Starting Grid', 'Position', 'Points', 'Laps']
                valid_cols = [c for c in res_cols if c in driver_df.columns]
                st.dataframe(driver_df[valid_cols], hide_index=True, use_container_width=True)


def render_teams_tab(df):
    """Constructor Analysis tab content."""
    st.header("Constructor Analysis")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # 2025 Team Data with car images and specs
    # Keys match CSV team names exactly for proper matching
    TEAM_DATA = {
        'McLaren': {
            'full_name': 'McLaren Formula 1 Team',
            'base': 'Woking, United Kingdom',
            'team_principal': 'Andrea Stella',
            'technical_director': 'Peter Prodromou',
            'chassis': 'MCL39',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 1966,
            'championships': 8,
            'constructor_titles': 8,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mclaren.png',
        },
        'McLaren Mercedes': {
            'full_name': 'McLaren Formula 1 Team',
            'base': 'Woking, United Kingdom',
            'team_principal': 'Andrea Stella',
            'technical_director': 'Peter Prodromou',
            'chassis': 'MCL39',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 1966,
            'championships': 8,
            'constructor_titles': 8,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mclaren.png',
        },
        'Red Bull Racing Honda RBPT': {
            'full_name': 'Oracle Red Bull Racing',
            'base': 'Milton Keynes, United Kingdom',
            'team_principal': 'Christian Horner',
            'technical_director': 'Pierre Wache',
            'chassis': 'RB21',
            'power_unit': 'Honda RBPT',
            'first_entry': 2005,
            'championships': 7,
            'constructor_titles': 6,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/red-bull-racing.png',
        },
        'Red Bull Racing Honda EBPT': {
            'full_name': 'Oracle Red Bull Racing',
            'base': 'Milton Keynes, United Kingdom',
            'team_principal': 'Christian Horner',
            'technical_director': 'Pierre Wache',
            'chassis': 'RB21',
            'power_unit': 'Honda RBPT',
            'first_entry': 2005,
            'championships': 7,
            'constructor_titles': 6,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/red-bull-racing.png',
        },
        'Ferrari': {
            'full_name': 'Scuderia Ferrari HP',
            'base': 'Maranello, Italy',
            'team_principal': 'Frederic Vasseur',
            'technical_director': 'Loic Serra',
            'chassis': 'SF-25',
            'power_unit': 'Ferrari 067',
            'first_entry': 1950,
            'championships': 15,
            'constructor_titles': 16,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/ferrari.png',
        },
        'Mercedes': {
            'full_name': 'Mercedes-AMG Petronas F1 Team',
            'base': 'Brackley, United Kingdom',
            'team_principal': 'Toto Wolff',
            'technical_director': 'James Allison',
            'chassis': 'W16',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 2010,
            'championships': 7,
            'constructor_titles': 8,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mercedes.png',
        },
        'Aston Martin Aramco Mercedes': {
            'full_name': 'Aston Martin Aramco F1 Team',
            'base': 'Silverstone, United Kingdom',
            'team_principal': 'Mike Krack',
            'technical_director': 'Dan Fallows',
            'chassis': 'AMR25',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 2021,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/aston-martin.png',
        },
        'Alpine Renault': {
            'full_name': 'BWT Alpine F1 Team',
            'base': 'Enstone, United Kingdom',
            'team_principal': 'Oliver Oakes',
            'technical_director': 'David Sanchez',
            'chassis': 'A525',
            'power_unit': 'Renault E-Tech RE25',
            'first_entry': 2021,
            'championships': 0,
            'constructor_titles': 2,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/alpine.png',
        },
        'Williams Mercedes': {
            'full_name': 'Williams Racing',
            'base': 'Grove, United Kingdom',
            'team_principal': 'James Vowles',
            'technical_director': 'Pat Fry',
            'chassis': 'FW47',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 1978,
            'championships': 7,
            'constructor_titles': 9,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/williams.png',
        },
        'Racing Bulls Honda RBPT': {
            'full_name': 'Visa Cash App Racing Bulls',
            'base': 'Faenza, Italy',
            'team_principal': 'Laurent Mekies',
            'technical_director': 'Jody Egginton',
            'chassis': 'VCARB 02',
            'power_unit': 'Honda RBPT',
            'first_entry': 2006,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/rb.png',
        },
        'Kick Sauber Ferrari': {
            'full_name': 'Stake F1 Team Kick Sauber',
            'base': 'Hinwil, Switzerland',
            'team_principal': 'Mattia Binotto',
            'technical_director': 'James Key',
            'chassis': 'C45',
            'power_unit': 'Ferrari 067',
            'first_entry': 1993,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/kick-sauber.png',
        },
        'Haas Ferrari': {
            'full_name': 'MoneyGram Haas F1 Team',
            'base': 'Kannapolis, USA',
            'team_principal': 'Ayao Komatsu',
            'technical_director': 'Andrea De Zordo',
            'chassis': 'VF-25',
            'power_unit': 'Ferrari 067',
            'first_entry': 2016,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/haas.png',
        },
    }
    
    # Team selector with session state
    teams = df['Team'].unique().tolist()
    selected_team = st.selectbox(
        "Select Team", 
        sorted(teams),
        key="team_profile_selector"
    )
    
    if selected_team:
        team_df = df[df['Team'] == selected_team]
        team_color = TEAM_COLORS.get(selected_team, '#666666')
        
        # Find matching team data - prioritize exact matches
        team_info = None
        
        # First try exact match
        for key, data in TEAM_DATA.items():
            if key.lower() == selected_team.lower():
                team_info = data
                break
        
        # If no exact match, try to find best partial match
        # But avoid "Ferrari" matching "Haas Ferrari" - use startswith instead
        if team_info is None:
            for key, data in TEAM_DATA.items():
                # Check if selected_team starts with key or key starts with selected_team
                if selected_team.lower().startswith(key.lower()) or key.lower().startswith(selected_team.lower()):
                    team_info = data
                    break
        
        # Final fallback - word-based match
        if team_info is None:
            selected_words = set(selected_team.lower().split())
            for key, data in TEAM_DATA.items():
                key_words = set(key.lower().split())
                # Match if the key's main word is in selected_team
                if key_words & selected_words:
                    team_info = data
                    break
        
        if team_info is None:
            team_info = {}
        
        st.divider()
        
        # Team header with car image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if team_info.get('car_image'):
                st.image(team_info['car_image'], use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; background: linear-gradient(135deg, {team_color}33 0%, {team_color}11 100%); border-radius: 15px; border: 2px solid {team_color};">
                <h2 style="color: {team_color}; margin: 0;">{team_info.get('full_name', selected_team)}</h2>
                <p style="color: #888; margin-top: 5px;">{team_info.get('base', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Team Specifications
        st.subheader("2025 Technical Specifications")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            **Chassis**  
            {team_info.get('chassis', 'N/A')}
            """)
        with col2:
            st.markdown(f"""
            **Power Unit**  
            {team_info.get('power_unit', 'N/A')}
            """)
        with col3:
            st.markdown(f"""
            **Team Principal**  
            {team_info.get('team_principal', 'N/A')}
            """)
        with col4:
            st.markdown(f"""
            **Technical Director**  
            {team_info.get('technical_director', 'N/A')}
            """)
        
        # Team history
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Entry", team_info.get('first_entry', 'N/A'))
        with col2:
            st.metric("Driver Titles", team_info.get('championships', 0))
        with col3:
            st.metric("Constructor Titles", team_info.get('constructor_titles', 0))
        with col4:
            st.metric("Years in F1", 2025 - team_info.get('first_entry', 2025) if team_info.get('first_entry') else 'N/A')
        
        st.divider()
        
        # 2025 Season stats
        st.subheader("2025 Season Performance")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_points = team_df['Points'].sum()
        wins = len(team_df[team_df['Position'] == 1])
        podiums = len(team_df[team_df['Position'] <= 3])
        avg_pos = team_df['Position'].mean()
        drivers = team_df['Driver'].unique()
        
        with col1:
            st.metric("Total Points", int(total_points))
        with col2:
            st.metric("Wins", wins)
        with col3:
            st.metric("Podiums", podiums)
        with col4:
            st.metric("Avg Position", f"{avg_pos:.1f}")
        with col5:
            st.metric("Drivers", len(drivers))
        
        st.divider()
        
        # Driver comparison
        st.subheader("Driver Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Points comparison - fix color issue
            driver_points = team_df.groupby('Driver')['Points'].sum().reset_index()
            
            # Create lighter version of team color for second driver
            def lighten_color(hex_color, factor=0.5):
                hex_color = hex_color.lstrip('#')
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                r = int(r + (255 - r) * factor)
                g = int(g + (255 - g) * factor)
                b = int(b + (255 - b) * factor)
                return f'#{r:02x}{g:02x}{b:02x}'
            
            colors = [team_color, lighten_color(team_color)]
            
            fig = go.Figure(data=[go.Pie(
                labels=driver_points['Driver'],
                values=driver_points['Points'],
                marker_colors=colors[:len(driver_points)],
                hole=0.4
            )])
            fig.update_layout(
                title="Points Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            show_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average position comparison - fix color issue
            driver_avg = team_df.groupby('Driver')['Position'].mean().reset_index()
            colors = [team_color, lighten_color(team_color)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=driver_avg['Driver'],
                y=driver_avg['Position'],
                marker_color=colors[:len(driver_avg)],
                text=[f"{p:.1f}" for p in driver_avg['Position']],
                textposition='outside'
            ))
            fig.update_layout(
                title="Average Race Position",
                yaxis=dict(autorange='reversed', title='Position'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            show_plotly_chart(fig, use_container_width=True)
        
        # Points progression
        st.subheader("Team Points Progression")
        
        fig = go.Figure()
        colors = [team_color, lighten_color(team_color)]
        
        for i, driver in enumerate(drivers):
            driver_data = team_df[team_df['Driver'] == driver].copy()
            driver_data = driver_data.sort_values('Track')
            cumsum = driver_data['Points'].cumsum()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumsum) + 1)),
                y=cumsum,
                name=driver,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            xaxis_title="Race Number",
            yaxis_title="Cumulative Points",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        show_plotly_chart(fig, use_container_width=True)


def render_race_detail_tab(df):
    """Race Weekend Details tab content."""
    st.header("Race Weekend Details")
    
    # Race selector - 2025 only
    available_races = F1_2025_COMPLETED_RACES
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_race = st.selectbox(
            "Select Grand Prix", 
            available_races,
            key="race_detail_gp_selector"
        )
    with col2:
        session_type = st.selectbox(
            "Session", 
            ["Race", "Qualifying", "Sprint", "Practice 1", "Practice 2", "Practice 3"],
            key="race_detail_session_selector"
        )
    
    if selected_race:
        st.divider()
        
        # Load FastF1 data
        with st.spinner("Loading session data..."):
            setup_fastf1_cache()
            session = load_fastf1_session(2025, selected_race, session_type)
        
        if session is None:
            st.error(f"Could not load session data for {selected_race} - {session_type}")
            return
        
        st.divider()
        
        # Weather Panel
        st.subheader("Weather Conditions")
        
        weather = get_weather_summary(session)
        
        if weather and weather.get('available', False):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                air_temp = weather.get('air_temp_avg')
                st.metric("Air Temp", f"{air_temp:.1f}°C" if air_temp is not None else "N/A")
            with col2:
                track_temp = weather.get('track_temp_avg')
                st.metric("Track Temp", f"{track_temp:.1f}°C" if track_temp is not None else "N/A")
            with col3:
                humidity = weather.get('humidity_avg')
                st.metric("Humidity", f"{humidity:.0f}%" if humidity is not None else "N/A")
            with col4:
                wind = weather.get('wind_speed_avg')
                st.metric("Wind Speed", f"{wind:.1f} km/h" if wind is not None else "N/A")
            with col5:
                rain_status = "Yes" if weather.get('rainfall', False) else "No" # Removed emojis
                st.metric("Rainfall", rain_status)
        else:
            st.info("Weather data not available for this session")
        
        st.divider()
        
        # Tabs for different data views - expanded with more FastF1 features
        race_tabs = st.tabs([
            "Results", 
            "Pit Stops", 
            "Tyre Strategy", 
            "Track Status", 
            "Sector Times",
            "Speed Traps",
            "Position Changes",
            "Lap Times"
        ])
        
        with race_tabs[0]:
            # Race Results
            st.subheader("Session Results")
            try:
                results = session.results
                if results is not None and not results.empty:
                    display_results = results[['Position', 'Abbreviation', 'FullName', 'TeamName', 'Time', 'Status']].copy()
                    
                    # Apply global formatter
                    display_results['Time'] = display_results['Time'].apply(format_f1_time)
                    
                    display_results.columns = ['Pos', 'Code', 'Driver', 'Team', 'Time', 'Status']
                    st.dataframe(display_results, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not load results: {e}")
        
        with race_tabs[1]:
            # Pit Stops
            st.subheader("Pit Stop Analysis")
            pit_stops = get_pit_stops(session)
            
            if pit_stops is not None and not pit_stops.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pit Stops", len(pit_stops))
                with col2:
                    avg_time = pit_stops['PitTime'].mean()
                    st.metric("Average Pit Time", f"{avg_time:.1f}s")
                with col3:
                    fastest = pit_stops['PitTime'].min()
                    fastest_driver = pit_stops[pit_stops['PitTime'] == fastest]['Driver'].iloc[0]
                    st.metric("Fastest Stop", f"{fastest:.1f}s ({fastest_driver})")
                with col4:
                    slowest = pit_stops['PitTime'].max()
                    st.metric("Slowest Stop", f"{slowest:.1f}s")
                
                st.divider()
                
                # Pit stop chart - horizontal bar chart grouped by driver
                st.markdown("**Pit Stop Times by Driver**")
                
                # Get unique drivers and sort by first stop lap
                driver_order = pit_stops.groupby('Driver')['Lap'].min().sort_values().index.tolist()
                
                fig = go.Figure()
                
                for driver in driver_order[:15]:  # Top 15 drivers
                    driver_stops = pit_stops[pit_stops['Driver'] == driver].sort_values('Stop')
                    
                    for _, stop in driver_stops.iterrows():
                        fig.add_trace(go.Bar(
                            y=[driver],
                            x=[stop['PitTime']],
                            orientation='h',
                            name=f"Stop {int(stop['Stop'])}",
                            text=f"L{int(stop['Lap'])}: {stop['PitTime']:.1f}s",
                            textposition='auto',
                            marker_color=['#E10600', '#FFD700', '#00FF00', '#3333FF'][int(stop['Stop']-1) % 4],
                            showlegend=False,
                            hovertemplate=f"{driver} - Stop {int(stop['Stop'])}<br>Lap: {int(stop['Lap'])}<br>Time: {stop['PitTime']:.1f}s<extra></extra>"
                        ))
                
                fig.update_layout(
                    title="Pit Stop Duration (seconds)",
                    xaxis_title="Pit Time (seconds)",
                    yaxis_title="Driver",
                    height=max(400, len(driver_order) * 30),
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(range=[0, max(pit_stops['PitTime'].max() * 1.1, 30)])
                )
                show_plotly_chart(fig, use_container_width=True)
                
                # Detailed pit stop table
                st.markdown("**Detailed Pit Stop Data**")
                display_pit = pit_stops.copy()
                display_pit['PitTime'] = display_pit['PitTime'].apply(lambda x: f"{x:.1f}s")
                display_pit = display_pit.rename(columns={'Stop': 'Stop #', 'PitTime': 'Duration'})
                st.dataframe(display_pit, use_container_width=True, hide_index=True)
            else:
                st.info("Pit stop data not available")
        
        with race_tabs[2]:
            # Tyre Strategy
            st.subheader("Tyre Strategy")
            tyre_data = get_tyre_stints(session)
            
            # Check if this is a sprint weekend
            sprint_tracks = ['China', 'Miami', 'Belgium', 'United States', 'Brazil', 'Qatar']
            is_sprint_weekend = selected_race in sprint_tracks
            
            if is_sprint_weekend:
                st.info(f"This is a Sprint Weekend - {selected_race}")
            
            if tyre_data is not None and not tyre_data.empty:
                # Compound Legend with visual - Removed emojis
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown("**SOFT** (Red)")
                with col2:
                    st.markdown("**MEDIUM** (Yellow)")
                with col3:
                    st.markdown("**HARD** (White)")
                with col4:
                    st.markdown("**INTERMEDIATE** (Green)")
                with col5:
                    st.markdown("**WET** (Blue)")
                
                # Stint details table
                with st.expander("View Detailed Stint Data"):
                    display_tyres = tyre_data[['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'Laps']].copy()
                    display_tyres['Stint'] = display_tyres['Stint'] + 1
                    display_tyres = display_tyres.rename(columns={'Stint': 'Stint #', 'StartLap': 'Start', 'EndLap': 'End'})
                    st.dataframe(display_tyres, use_container_width=True, hide_index=True)
            else:
                st.info("Tyre strategy data not available")
        
        with race_tabs[3]:
            # Track Status
            st.subheader("Track Status & Race Control")
            
            track_status = get_track_status(session)
            flag_events = get_flag_events(session)
            
            if track_status is not None and not track_status.empty:
                status_colors = {
                    '1': '#00FF00',  # Green
                    '2': '#FFFF00',  # Yellow
                    '4': '#FF0000',  # Red
                    '5': '#FF6600',  # SC
                    '6': '#FF00FF',  # VSC
                    '7': '#0000FF'   # VSC Ending
                }
                
                fig = go.Figure()
                
                for _, status in track_status.iterrows():
                    status_code = str(status.get('Status', '1'))
                    time_val = status.get('Time', 0)
                    color = status_colors.get(status_code, '#888888')
                    
                    fig.add_trace(go.Scatter(
                        x=[time_val],
                        y=[1],
                        mode='markers',
                        marker=dict(size=15, color=color),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Track Status Timeline",
                    xaxis_title="Session Time",
                    height=200,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                show_plotly_chart(fig, use_container_width=True)
            
            if flag_events is not None and len(flag_events) > 0:
                st.subheader("Flag Events")
                flag_df = pd.DataFrame(flag_events)
                st.dataframe(flag_df, use_container_width=True, hide_index=True)
            else:
                st.info("Flag event data not available")
        
        with race_tabs[4]:
            # Sector Times
            st.subheader("Sector Times Analysis")
            
            # Field Best Sectors
            with st.expander("View Field Best Sectors"):
                best_sectors_df = get_best_sectors(session)
                if not best_sectors_df.empty:
                    st.dataframe(best_sectors_df, use_container_width=True, hide_index=True)
            
            sector_times = get_sector_times(session)
            
            if sector_times is not None and not sector_times.empty and 'Sector1' in sector_times.columns:
                # Select driver for sector analysis
                drivers_list = sector_times['Driver'].unique().tolist()
                selected_driver = st.selectbox("Select Driver for Sector Analysis", drivers_list, key="sector_driver")
                
                driver_sectors = sector_times[sector_times['Driver'] == selected_driver]
                driver_sectors = driver_sectors.dropna(subset=['Sector1', 'Sector2', 'Sector3'], how='all')
                
                if not driver_sectors.empty and len(driver_sectors) > 0:
                    # Check if we have valid data
                    has_s1 = 'Sector1' in driver_sectors.columns and driver_sectors['Sector1'].notna().any()
                    has_s2 = 'Sector2' in driver_sectors.columns and driver_sectors['Sector2'].notna().any()
                    has_s3 = 'Sector3' in driver_sectors.columns and driver_sectors['Sector3'].notna().any()
                    
                    if has_s1 or has_s2 or has_s3:
                        fig = make_subplots(rows=1, cols=3, subplot_titles=('Sector 1', 'Sector 2', 'Sector 3'))
                        
                        colors = ['#FF3333', '#00FF00', '#3333FF']
                        
                        for i, sector in enumerate(['Sector1', 'Sector2', 'Sector3'], 1):
                            if sector in driver_sectors.columns and driver_sectors[sector].notna().any():
                                valid_data = driver_sectors[driver_sectors[sector].notna()]
                                fig.add_trace(
                                    go.Scatter(
                                        x=valid_data['Lap'],
                                        y=valid_data[sector],
                                        mode='lines+markers',
                                        name=f"S{i}",
                                        line=dict(color=colors[i-1], width=2),
                                        marker=dict(size=4)
                                    ),
                                    row=1, col=i
                                )
                        
                        fig.update_layout(
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            showlegend=False
                        )
                        fig.update_yaxes(title_text="Time (s)")
                        fig.update_xaxes(title_text="Lap")
                        show_plotly_chart(fig, use_container_width=True)
                        
                        # Best sectors table
                        st.markdown("**Best Sector Times:**")
                        best_s1 = driver_sectors['Sector1'].min() if 'Sector1' in driver_sectors.columns else None
                        best_s2 = driver_sectors['Sector2'].min() if 'Sector2' in driver_sectors.columns else None
                        best_s3 = driver_sectors['Sector3'].min() if 'Sector3' in driver_sectors.columns else None
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Best S1", f"{best_s1:.3f}s" if best_s1 else "N/A")
                        with col2:
                            st.metric("Best S2", f"{best_s2:.3f}s" if best_s2 else "N/A")
                        with col3:
                            st.metric("Best S3", f"{best_s3:.3f}s" if best_s3 else "N/A")
                        with col4:
                            theoretical = (best_s1 or 0) + (best_s2 or 0) + (best_s3 or 0)
                            st.metric("Theoretical Best", f"{theoretical:.3f}s" if theoretical > 0 else "N/A")
                    else:
                        st.info("No valid sector time data for this driver")
                else:
                    st.info("No sector time data available for this driver")
            else:
                st.info("Sector time data not available for this session")
        
        with race_tabs[5]:
            # Speed Traps
            st.subheader("Speed Trap Analysis")
            top_speeds = get_top_speeds(session)
            
            if top_speeds is not None and not top_speeds.empty:
                st.dataframe(top_speeds, use_container_width=True, hide_index=True)
                
                if 'Max_SpeedST' in top_speeds.columns:
                    fig = go.Figure()
                    # Sort by speed
                    plot_data = top_speeds.sort_values('Max_SpeedST', ascending=True)
                    
                    fig.add_trace(go.Bar(
                        x=plot_data['Max_SpeedST'],
                        y=plot_data['Driver'],
                        orientation='h',
                        marker_color='#E10600',
                        text=plot_data['Max_SpeedST'].apply(lambda x: f"{x:.1f}"),
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Top Speeds (Speed Trap)",
                        xaxis_title="Speed (km/h)",
                        height=max(400, len(plot_data) * 25),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(r=100)
                    )
                    show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Speed trap data not available")

        with race_tabs[6]:
            # Position Changes - Improved visualization
            st.subheader("Position Changes Throughout Race")
            
            position_changes = get_position_changes(session)
            
            if position_changes is not None and not position_changes.empty:
                # Overview metrics
                gainers = position_changes[position_changes['PositionsGained'] > 0]
                losers = position_changes[position_changes['PositionsGained'] < 0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if not gainers.empty:
                        best = gainers.nlargest(1, 'PositionsGained').iloc[0]
                        st.metric(
                            "Best Mover", 
                            best['Driver'],
                            f"+{int(best['PositionsGained'])} positions"
                        )
                    else:
                        st.metric("Best Mover", "N/A", "0")
                
                with col2:
                    if not losers.empty:
                        worst = losers.nsmallest(1, 'PositionsGained').iloc[0]
                        st.metric(
                            "Biggest Drop", 
                            worst['Driver'],
                            f"{int(worst['PositionsGained'])} positions"
                        )
                    else:
                        st.metric("Biggest Drop", "N/A", "0")
                
                with col3:
                    no_change = len(position_changes[position_changes['PositionsGained'] == 0])
                    st.metric("Held Position", f"{no_change} drivers")
                
                st.divider()
                
                # Detailed chart with dual view
                col1, col2 = st.columns(2)
                
                with col1:
                    # Positions Gained/Lost Bar Chart
                    pos_changes_sorted = position_changes.sort_values('PositionsGained', ascending=True)
                    
                    colors = ['rgba(0, 255, 0, 0.8)' if x > 0 else 'rgba(255, 0, 0, 0.8)' if x < 0 else 'rgba(136, 136, 136, 0.8)' 
                              for x in pos_changes_sorted['PositionsGained']]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=pos_changes_sorted['PositionsGained'],
                        y=pos_changes_sorted['Driver'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"+{int(x)}" if x > 0 else str(int(x)) for x in pos_changes_sorted['PositionsGained']],
                        textposition='outside',
                        textfont=dict(color='white', size=12)
                    ))
                    
                    fig.update_layout(
                        title="Positions Gained/Lost",
                        xaxis_title="Positions Change",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(l=10, r=120, t=40, b=40),
                        xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1)
                    )
                    show_plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Grid vs Finish Position comparison
                    pos_sorted = position_changes.sort_values('FinalPosition')
                    
                    fig2 = go.Figure()
                    
                    # Grid position
                    fig2.add_trace(go.Scatter(
                        x=pos_sorted['Driver'],
                        y=pos_sorted['GridPosition'],
                        mode='markers+lines',
                        name='Grid',
                        marker=dict(size=12, color='#FFD700', symbol='diamond'),
                        line=dict(color='#FFD700', dash='dash')
                    ))
                    
                    # Finish position
                    fig2.add_trace(go.Scatter(
                        x=pos_sorted['Driver'],
                        y=pos_sorted['FinalPosition'],
                        mode='markers+lines',
                        name='Finish',
                        marker=dict(size=12, color='#00FF00', symbol='circle'),
                        line=dict(color='#00FF00')
                    ))
                    
                    fig2.update_layout(
                        title="Grid vs Finish Position",
                        yaxis_title="Position",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis=dict(autorange='reversed'),  # P1 at top
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                        xaxis=dict(tickangle=45)
                    )
                    show_plotly_chart(fig2, use_container_width=True)
                
                st.divider()
                
                # Position changes detailed table
                st.markdown("**Complete Position Data:**")
                
                # Format table
                display_df = position_changes.copy()
                
                # Add status column
                def get_status(row):
                    change = row['PositionsGained']
                    if change > 0:
                        return f"UP {int(change)}"
                    elif change < 0:
                        return f"DOWN {int(abs(change))}"
                    else:
                        return "SAME"
                
                display_df['Status'] = display_df.apply(get_status, axis=1)
                
                # Reorder columns
                cols_order = ['Driver', 'GridPosition', 'FinalPosition', 'PositionsGained', 'Status']
                available_cols = [c for c in cols_order if c in display_df.columns]
                display_df = display_df[available_cols].sort_values('FinalPosition')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Position change data not available")
        
        with race_tabs[7]:
            # Lap Times Analysis
            st.subheader("Lap Time Analysis") # Removed emoji
            
            try:
                laps = session.laps
                
                if laps is not None and not laps.empty:
                    # Driver selector for lap times
                    drivers_list = laps['Driver'].unique().tolist()
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        selected_drivers = st.multiselect(
                            "Select Drivers (max 5)", 
                            drivers_list, 
                            default=drivers_list[:3] if len(drivers_list) >= 3 else drivers_list,
                            max_selections=5,
                            key="lap_drivers"
                        )
                    
                    if selected_drivers:
                        fig = go.Figure()
                        
                        for driver in selected_drivers:
                            driver_laps = laps[laps['Driver'] == driver]
                            lap_times = driver_laps['LapTime'].dt.total_seconds()
                            
                            # Get team color
                            team_name = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns and len(driver_laps) > 0 else ''
                            color = TEAM_COLORS.get(team_name, '#888888')
                            
                            fig.add_trace(go.Scatter(
                                x=driver_laps['LapNumber'],
                                y=lap_times,
                                mode='lines+markers',
                                name=driver,
                                line=dict(color=color, width=2),
                                marker=dict(size=4)
                            ))
                        
                        fig.update_layout(
                            title="Lap Time Comparison",
                            xaxis_title="Lap Number",
                            yaxis_title="Lap Time (seconds)",
                            height=450,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02)
                        )
                        show_plotly_chart(fig, use_container_width=True)
                        
                        # Fastest laps table
                        st.markdown("**Fastest Laps:**")
                        fastest_laps = laps.groupby('Driver').apply(
                            lambda x: x.nsmallest(1, 'LapTime')
                        ).reset_index(drop=True)
                        
                        if not fastest_laps.empty:
                            display_cols = ['Driver', 'LapNumber', 'LapTime', 'Compound']
                            available_cols = [c for c in display_cols if c in fastest_laps.columns]
                            fastest_display = fastest_laps[available_cols].sort_values('LapTime') # Show all drivers
                            fastest_display['LapTime'] = fastest_display['LapTime'].apply(format_f1_time) # Apply formatter
                            st.dataframe(fastest_display, use_container_width=True, hide_index=True)
                else:
                    st.info("Lap time data not available")
            except Exception as e:
                st.error(f"Error loading lap times: {e}")


def render_race_analysis_tab(df):
    """Race Analysis tab content."""
    st.header("Race Analysis Center")
    st.markdown("Lap times, Pace comparison, Strategy analysis, Track Visualization") # Removed emojis
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Race selector
    col1, col2 = st.columns([2, 1])
    with col1:
        # Use all races from config, not just completed ones
        all_races = list(F1_2025_RACE_NAMES.keys())
        selected_race = st.selectbox("Select Grand Prix", all_races, key="analysis_race")
    with col2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"], key="analysis_session")
    
    # Check if race has happened or is ongoing
    try:
        schedule = fastf1.get_event_schedule(2025)
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        
        # Find event
        race_event = schedule[schedule['EventName'] == F1_2025_RACE_NAMES.get(selected_race, selected_race)]
        
        if not race_event.empty:
            event = race_event.iloc[0]
            now = pd.Timestamp.now(tz='UTC')
            
            # Check session specific time
            session_key = 'Session1' # Default
            if session_type == "Race": session_key = 'Session5'
            elif session_type == "Qualifying": session_key = 'Session4'
            elif session_type == "Sprint": session_key = 'Session3'
            
            # Map session names loosely if exact key match fails or for standard logic
            # FastF1 schedule keys: Session1...Session5. We need to find which one corresponds to selected type.
            # Actually, easier to just check if date is future.
            
            # Simple check: is event in future?
            if event['EventDate'] > (now + timedelta(hours=48)): # More than 2 days in future
                st.info(f"Analysis not available yet. {selected_race} has not started.")
                return
            
            # More detailed check if we had session times mapping, but simple catch-all exception on load works too.
    except Exception as e:
        pass # Ignore schedule check errors, let load fail naturally if needed

    # Analysis sub-tabs
    analysis_tabs = st.tabs([
        "Lap Analysis", "Pace Comparison", "Stint Analysis", 
        "Gap Chart", "Track Visualization", "Position Chart", "Battle Analysis",
        "Strategy Tools", "Driver Scores", "Telemetry Compare",
        "Race Control", "Engineering"
    ])
    
    # Load session
    with st.spinner("Loading session data..."):
        setup_fastf1_cache()
        # Use mapped name if available
        race_lookup = F1_2025_RACE_NAMES.get(selected_race, selected_race)
        session = load_fastf1_session(2025, race_lookup, session_type)
    
    if session is None:
        st.warning(f"Data not available for {selected_race} - {session_type}. The session might not have started yet.")
        return
    
    # TAB 1: LAP ANALYSIS
    with analysis_tabs[0]:
        st.subheader("Lap Time Analysis")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                selected_drivers = st.multiselect("Select Drivers (max 6)", drivers_list, 
                    default=drivers_list[:3] if len(drivers_list) >= 3 else drivers_list, max_selections=6, key="lap_drv")
                
                if selected_drivers:
                    fig = go.Figure()
                    for driver in selected_drivers:
                        drv_laps = laps[laps['Driver'] == driver]
                        drv_laps = drv_laps[drv_laps['LapTime'].notna()]
                        if not drv_laps.empty:
                            times = drv_laps['LapTime'].dt.total_seconds()
                            team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                            color = TEAM_COLORS.get(team, '#888888')
                            fig.add_trace(go.Scatter(x=drv_laps['LapNumber'], y=times, mode='lines+markers',
                                name=driver, line=dict(color=color, width=2), marker=dict(size=4)))
                    
                    fig.update_layout(title="Lap Time Evolution", xaxis_title="Lap", yaxis_title="Time (s)",
                        height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig, use_container_width=True)
                    
                    # Fastest laps table
                    st.markdown("**Fastest Laps:**")
                    fastest_laps = laps.groupby('Driver').apply(
                        lambda x: x.nsmallest(1, 'LapTime')
                    ).reset_index(drop=True)
                    
                    if not fastest_laps.empty:
                        display_cols = ['Driver', 'LapNumber', 'LapTime', 'Compound']
                        available_cols = [c for c in display_cols if c in fastest_laps.columns]
                        fastest_display = fastest_laps[available_cols].sort_values('LapTime')
                        fastest_display['LapTime'] = fastest_display['LapTime'].apply(format_f1_time)
                        st.dataframe(fastest_display, use_container_width=True, hide_index=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 2: PACE COMPARISON
    with analysis_tabs[1]:
        st.subheader("Pace Comparison")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                c1, c2 = st.columns(2)
                with c1:
                    drv_a = st.selectbox("Driver A", drivers_list, key="pace_a")
                with c2:
                    drv_b = st.selectbox("Driver B", [d for d in drivers_list if d != drv_a], key="pace_b")
                
                if st.button("Compare", type="primary", key="compare_btn"):
                    laps_a = laps[(laps['Driver'] == drv_a) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    laps_b = laps[(laps['Driver'] == drv_b) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    
                    if not laps_a.empty and not laps_b.empty:
                        times_a = laps_a['LapTime'].dt.total_seconds()
                        times_b = laps_b['LapTime'].dt.total_seconds()
                        team_a = laps_a['Team'].iloc[0] if 'Team' in laps_a.columns else ''
                        team_b = laps_b['Team'].iloc[0] if 'Team' in laps_b.columns else ''
                        
                        avg_diff = times_a.mean() - times_b.mean()
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(f"{drv_a} Avg", f"{times_a.mean():.3f}s", f"{avg_diff:+.3f}s")
                        with c2:
                            st.metric(f"{drv_b} Avg", f"{times_b.mean():.3f}s", f"{-avg_diff:+.3f}s")
                        with c3:
                            faster = drv_a if avg_diff < 0 else drv_b
                            st.metric("Faster", faster, f"{abs(avg_diff):.3f}s")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=laps_a['LapNumber'], y=times_a, name=drv_a,
                            line=dict(color=TEAM_COLORS.get(team_a, '#FF0000'), width=2)))
                        fig.add_trace(go.Scatter(x=laps_b['LapNumber'], y=times_b, name=drv_b,
                            line=dict(color=TEAM_COLORS.get(team_b, '#00FF00'), width=2)))
                        fig.update_layout(title=f"{drv_a} vs {drv_b}", xaxis_title="Lap", yaxis_title="Time (s)",
                            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 3: STINT ANALYSIS
    with analysis_tabs[2]:
        st.subheader("Tyre Stint Analysis")
        try:
            tyre_data = get_tyre_stints(session)
            if tyre_data is not None and not tyre_data.empty:
                compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF', 'INTERMEDIATE': '#43B02A', 'WET': '#0067AD'}
                
                try:
                    results = session.results
                    driver_order = results.sort_values('Position')['Abbreviation'].tolist() if results is not None else tyre_data['Driver'].unique().tolist()
                except:
                    driver_order = tyre_data['Driver'].unique().tolist()
                
                drivers = [d for d in driver_order if d in tyre_data['Driver'].values][:20]
                fig = go.Figure()
                
                for driver in drivers:
                    driver_tyres = tyre_data[tyre_data['Driver'] == driver].sort_values('Stint')
                    
                    for _, stint in driver_tyres.iterrows():
                        compound = str(stint.get('Compound', 'MEDIUM')).upper()
                        start = int(stint.get('StartLap', 1))
                        end = int(stint.get('EndLap', start + 10))
                        laps = end - start + 1
                        color = compound_colors.get(compound, '#888888')
                        
                        fig.add_trace(go.Bar(x=[laps], y=[driver], orientation='h', base=start-1,
                            marker_color=color, marker_line_color='#333', marker_line_width=1,
                            showlegend=False, text=compound[0] if compound != 'UNKNOWN' else '?',
                            textposition='inside', textfont=dict(color='black' if compound in ['MEDIUM','HARD'] else 'white', size=10)))
                
                total_laps = tyre_data['EndLap'].max() if 'EndLap' in tyre_data.columns else 50
                fig.update_layout(title="Tyre Strategy", xaxis_title="Lap", xaxis=dict(range=[0, total_laps+2]),
                    height=max(450, len(drivers)*25), barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    yaxis=dict(categoryorder='array', categoryarray=drivers[::-1]))
                show_plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown("**SOFT** (Red)")
                with col2:
                    st.markdown("**MEDIUM** (Yellow)")
                with col3:
                    st.markdown("**HARD** (White)")
                with col4:
                    st.markdown("**INTERMEDIATE** (Green)")
                with col5:
                    st.markdown("**WET** (Blue)")
                
                # Stint details table
                with st.expander("View Detailed Stint Data"):
                    display_tyres = tyre_data[['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'Laps']].copy()
                    display_tyres['Stint'] = display_tyres['Stint'] + 1
                    display_tyres = display_tyres.rename(columns={'Stint': 'Stint #', 'StartLap': 'Start', 'EndLap': 'End'})
                    st.dataframe(display_tyres, use_container_width=True, hide_index=True)
            else:
                st.info("Tyre strategy data not available")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 4: GAP CHART
    with analysis_tabs[3]:
        st.subheader("Gap to Leader")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                gap_drivers = st.multiselect("Select Drivers (max 8)", drivers_list,
                    default=drivers_list[:5] if len(drivers_list) >= 5 else drivers_list, max_selections=8, key="gap_drv")
                
                if gap_drivers:
                    # Find leader
                    leader_data = None
                    min_time = float('inf')
                    for driver in gap_drivers:
                        drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())].sort_values('LapNumber')
                        if not drv_laps.empty:
                            cum = drv_laps['LapTime'].dt.total_seconds().cumsum()
                            total = cum.iloc[-1]
                            if total < min_time:
                                min_time = total
                                leader_data = (driver, drv_laps, cum)
                    
                    if leader_data:
                        leader_name, leader_laps, leader_cum = leader_data
                        leader_dict = dict(zip(leader_laps['LapNumber'], leader_cum))
                        
                        fig = go.Figure()
                        for driver in gap_drivers:
                            drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())].sort_values('LapNumber')
                            if not drv_laps.empty:
                                team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                                cum = drv_laps['LapTime'].dt.total_seconds().cumsum()
                                gaps = [c - leader_dict.get(l, c) for l, c in zip(drv_laps['LapNumber'], cum)]
                                fig.add_trace(go.Scatter(x=drv_laps['LapNumber'], y=gaps, mode='lines',
                                    name=driver, line=dict(color=TEAM_COLORS.get(team, '#888888'), width=2)))
                        
                        fig.update_layout(title=f"Gap to Leader ({leader_name})", xaxis_title="Lap", yaxis_title="Gap (s)",
                            height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 5: TRACK VISUALIZATION
    with analysis_tabs[4]:
        st.subheader("Track Visualization")
        
        with st.expander("3D Track Map Analysis", expanded=True):
            st.markdown("Interactive 3D map colored by speed. Drag to rotate, scroll to zoom.")
            try:
                # Get driver list for specific driver highlight
                if hasattr(session, 'drivers'):
                    drivers = session.drivers
                    drivers_list = [session.get_driver(d)["Abbreviation"] for d in drivers]
                    sel_driver = st.selectbox("Highlight Driver Line", ["Fastest Lap"] + sorted(drivers_list), key="3d_drv_sel")
                    driver_arg = None if sel_driver == "Fastest Lap" else sel_driver
                    
                    if st.button("Generate 3D Map", key="gen_3d_btn"):
                        with st.spinner("Generating 3D Map..."):
                            fig_3d = plot_track_3d(session, driver=driver_arg)
                            if fig_3d:
                                show_plotly_chart(fig_3d, use_container_width=True)
                            else:
                                st.error("Could not generate map (missing telemetry?)")
            except Exception as e:
                st.error(f"Error in 3D Map: {e}")

        st.divider()
        st.subheader("2D Live Race Replay")
        
        try:
            # Speed control
            col_speed, col_lap = st.columns([1, 2])
            with col_speed:
                frame_duration = st.slider("Animation Speed (ms)", 10, 300, 50, 10, 
                    help="Lower = faster animation", key="anim_speed")
            with col_lap:
                selected_lap_anim = st.number_input("Select Lap", min_value=1, value=1, key="anim_lap_select")
            
            if st.button("Generate Animation", type="primary", key="gen_anim_btn"):
                with st.spinner("Loading telemetry data..."):
                    # Get all drivers in session
                    laps = session.laps
                    if laps is None or laps.empty:
                        st.error("No lap data available")
                    else:
                        drivers_in_session = laps['Driver'].unique().tolist()
                        
                        # Collect telemetry for selected lap from all drivers
                        all_tel_data = []
                        driver_colors = {}
                        
                        for driver in drivers_in_session:
                            try:
                                drv_laps = laps.pick_driver(driver)
                                if drv_laps.empty:
                                    continue
                                
                                # Get team color
                                team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                                driver_colors[driver] = TEAM_COLORS.get(team, '#888888')
                                
                                # Get lap telemetry
                                lap_data = drv_laps[drv_laps['LapNumber'] == selected_lap_anim]
                                if lap_data.empty:
                                    lap_data = drv_laps.iloc[[0]]
                                
                                tel = lap_data.iloc[0].get_telemetry()
                                if tel is not None and not tel.empty and 'X' in tel.columns and 'Y' in tel.columns:
                                    tel = tel[['X', 'Y']].copy()
                                    tel['Driver'] = driver
                                    tel['Time_idx'] = range(len(tel))
                                    all_tel_data.append(tel)
                            except:
                                continue
                        
                        if not all_tel_data:
                            st.error("No telemetry data available for this lap")
                        else:
                            # Get track outline from driver with most data points
                            track_tel = max(all_tel_data, key=len)
                            track_x = track_tel['X'].values
                            track_y = track_tel['Y'].values
                            
                            # Get track boundaries
                            x_min, x_max = track_x.min(), track_x.max()
                            y_min, y_max = track_y.min(), track_y.max()
                            x_margin = (x_max - x_min) * 0.08
                            y_margin = (y_max - y_min) * 0.08
                            
                            # Number of animation frames
                            num_points = 150
                            
                            # Interpolate all drivers to same number of time points
                            interpolated_data = {}
                            for tel_df in all_tel_data:
                                driver = tel_df['Driver'].iloc[0]
                                orig_x = tel_df['X'].values
                                orig_y = tel_df['Y'].values
                                
                                if len(orig_x) < 2:
                                    continue
                                
                                # Linear interpolation
                                orig_indices = np.linspace(0, 1, len(orig_x))
                                new_indices = np.linspace(0, 1, num_points)
                                
                                interp_x = np.interp(new_indices, orig_indices, orig_x)
                                interp_y = np.interp(new_indices, orig_indices, orig_y)
                                
                                interpolated_data[driver] = {'X': interp_x, 'Y': interp_y}
                            
                            if not interpolated_data:
                                st.error("Could not interpolate driver data")
                            else:
                                # Create frames - each frame includes track + all car positions
                                frames = []
                                for frame_idx in range(num_points):
                                    frame_data = [
                                        # Track background (must be in every frame)
                                        go.Scatter(
                                            x=track_x, y=track_y,
                                            mode='lines',
                                            line=dict(color='#333333', width=20),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ),
                                        go.Scatter(
                                            x=track_x, y=track_y,
                                            mode='lines',
                                            line=dict(color='#555555', width=1),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        )
                                    ]
                                    
                                    # Add all drivers at current frame position
                                    for driver, data in interpolated_data.items():
                                        frame_data.append(go.Scatter(
                                            x=[data['X'][frame_idx]],
                                            y=[data['Y'][frame_idx]],
                                            mode='markers+text',
                                            marker=dict(size=14, color=driver_colors.get(driver, '#888888'), 
                                                       line=dict(width=2, color='white')),
                                            text=[driver],
                                            textposition='top center',
                                            textfont=dict(size=9, color='white'),
                                            name=driver,
                                            showlegend=False
                                        ))
                                    
                                    frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
                                
                                # Initial figure data (frame 0)
                                initial_data = [
                                    go.Scatter(
                                        x=track_x, y=track_y,
                                        mode='lines',
                                        line=dict(color='#333333', width=20),
                                        name='Track',
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ),
                                    go.Scatter(
                                        x=track_x, y=track_y,
                                        mode='lines',
                                        line=dict(color='#555555', width=1),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    )
                                ]
                                
                                for driver, data in interpolated_data.items():
                                    initial_data.append(go.Scatter(
                                        x=[data['X'][0]],
                                        y=[data['Y'][0]],
                                        mode='markers+text',
                                        marker=dict(size=14, color=driver_colors.get(driver, '#888888'),
                                                   line=dict(width=2, color='white')),
                                        text=[driver],
                                        textposition='top center',
                                        textfont=dict(size=9, color='white'),
                                        name=driver
                                    ))
                                
                                # Build animated figure
                                fig = go.Figure(
                                    data=initial_data,
                                    frames=frames,
                                    layout=go.Layout(
                                        title=dict(text=f"Lap {selected_lap_anim} - {selected_race}", 
                                                  font=dict(color='white', size=16)),
                                        xaxis=dict(range=[x_min - x_margin, x_max + x_margin], 
                                                  visible=False, scaleanchor='y'),
                                        yaxis=dict(range=[y_min - y_margin, y_max + y_margin], visible=False),
                                        height=600,
                                        paper_bgcolor='#0e1117',
                                        plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        showlegend=True,
                                        legend=dict(x=1.02, y=1, bgcolor='rgba(0,0,0,0.5)', font=dict(size=10)),
                                        updatemenus=[dict(
                                            type='buttons',
                                            showactive=False,
                                            y=0,
                                            x=0.1,
                                            xanchor='right',
                                            yanchor='top',
                                            buttons=[
                                                dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(
                                                        frame=dict(duration=frame_duration, redraw=True),
                                                        fromcurrent=True,
                                                        transition=dict(duration=0)
                                                    )]),
                                                dict(label='Pause',
                                                    method='animate',
                                                    args=[[None], dict(
                                                        frame=dict(duration=0, redraw=False),
                                                        mode='immediate',
                                                        transition=dict(duration=0)
                                                    )])
                                            ]
                                        )],
                                        sliders=[dict(
                                            active=0,
                                            yanchor='top',
                                            xanchor='left',
                                            currentvalue=dict(prefix='Progress: ', visible=True, 
                                                             xanchor='right', font=dict(color='white', size=12)),
                                            transition=dict(duration=0),
                                            pad=dict(b=10, t=50),
                                            len=0.9,
                                            x=0.1,
                                            y=0,
                                            steps=[dict(args=[[f.name], dict(
                                                frame=dict(duration=0, redraw=True),
                                                mode='immediate',
                                                transition=dict(duration=0)
                                            )], label='', method='animate') 
                                                for f in frames]
                                        )]
                                    )
                                )
                                
                                show_plotly_chart(fig, use_container_width=True)
                                st.caption(f"Animation: {len(interpolated_data)} drivers, {num_points} frames. Click Play to start.")
            
            # Show Track Dominance Map
            st.markdown("---")
            st.subheader("🗺️ Circuit Dominance Map")
            st.markdown("Color-coded track map showing which team/driver is fastest in each mini-sector.")
            
            if st.button("Generate Dominance Map", type="primary", key="dom_btn"):
                with st.spinner("Calculating mini-sector dominance..."):
                    try:
                        laps = session.laps
                        if laps is not None and not laps.empty:
                            # 1. Get telemetry for all fastest laps per driver
                            drivers = laps['Driver'].unique()
                            telemetry_data = []
                            
                            for d in drivers:
                                dl = laps.pick_driver(d).pick_fastest()
                                if dl is not None:
                                    t = dl.get_telemetry()
                                    if t is not None:
                                        t['Driver'] = d
                                        t['Team'] = dl['Team']
                                        telemetry_data.append(t)
                            
                            if telemetry_data:
                                # 2. Merge all telemetry
                                all_tel = pd.concat(telemetry_data)
                                
                                # 3. Create mini-sectors relative to distance
                                all_tel['DistanceInt'] = (all_tel['Distance'] // 50).astype(int) # 50m chunks
                                
                                # 4. Find fastest driver per chunk (Highest Speed)
                                # Simplified: Using Speed as proxy for 'fastest through sector'
                                sector_dominance = all_tel.loc[all_tel.groupby('DistanceInt')['Speed'].idxmax()]
                                
                                # 5. Plot
                                fig_dom = go.Figure()
                                for team in sector_dominance['Team'].unique():
                                    team_segments = sector_dominance[sector_dominance['Team'] == team]
                                    color = TEAM_COLORS.get(team, '#888')
                                    
                                    fig_dom.add_trace(go.Scatter(
                                        x=team_segments['X'], y=team_segments['Y'],
                                        mode='markers',
                                        marker=dict(size=4, color=color),
                                        name=team
                                    ))
                                
                                fig_dom.update_layout(
                                    height=500,
                                    paper_bgcolor='#0e1117',
                                    plot_bgcolor='#0e1117',
                                    xaxis=dict(visible=False, scaleanchor='y'),
                                    yaxis=dict(visible=False),
                                    title="Top Speed Dominance by Team",
                                    font=dict(color='white')
                                )
                                show_plotly_chart(fig_dom, use_container_width=True)
                            else:
                                st.warning("Not enough telemetry data")
                    except Exception as ex:
                        st.error(f"Dominance Map Error: {ex}")
            
            # Static Preview (Fallback)

            try:
                laps = session.laps
                if laps is not None and not laps.empty:
                    sample_driver = laps['Driver'].iloc[0]
                    drv_laps = laps.pick_driver(sample_driver)
                    if not drv_laps.empty:
                        fastest = drv_laps.pick_fastest()
                        if fastest is not None:
                            tel = fastest.get_telemetry()
                            if tel is not None and 'X' in tel.columns:
                                fig_preview = go.Figure()
                                fig_preview.add_trace(go.Scatter(
                                    x=tel['X'], y=tel['Y'],
                                    mode='lines',
                                    line=dict(color='#E10600', width=3),
                                    showlegend=False
                                ))
                                fig_preview.update_layout(
                                    height=250,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(visible=False, scaleanchor='y'),
                                    yaxis=dict(visible=False),
                                    margin=dict(l=0, r=0, t=0, b=0)
                                )
                                show_plotly_chart(fig_preview, use_container_width=True)
            except:
                pass

            # NEW: Corner Analysis Matrix
            st.divider()
            st.subheader("👑 Corner Mastery Matrix")
            st.markdown("Average speed in **Low (<120)**, **Medium (120-230)**, and **High (>230)** speed zones.")
            
            if st.button("Generate Performance Matrix", key="corn_btn"):
                with st.spinner("Analyzing corner speeds..."):
                    fig_corn = plot_corner_performance(session)
                    if fig_corn:
                        show_plotly_chart(fig_corn, use_container_width=True)
                    else:
                        st.info("Could not generate matrix")
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 6: POSITION CHART
    with analysis_tabs[5]:
        st.subheader("Position Chart")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                pos_data = []
                # Fix: Ensure correct types
                laps = laps.copy()
                laps['Position'] = pd.to_numeric(laps['Position'], errors='coerce')
                
                # Check column existence
                if 'Position' in laps.columns:
                    for lap_num in sorted(laps['LapNumber'].unique()):
                        lap = laps[laps['LapNumber'] == lap_num]
                        for _, row in lap.iterrows():
                            if pd.notna(row['Position']):
                                pos_data.append({'Lap': int(lap_num), 'Driver': row['Driver'], 
                                    'Position': int(row['Position']), 'Team': row.get('Team', '')})
                
                if pos_data:
                    pos_df = pd.DataFrame(pos_data)
                    total_laps = pos_df['Lap'].max()
                    
                    color_map = {d: TEAM_COLORS.get(pos_df[pos_df['Driver']==d]['Team'].iloc[0], '#888888') 
                                 for d in pos_df['Driver'].unique()}
                    
                    selected_lap = st.slider("Lap", 1, int(total_laps), 1, key="pos_lap")
                    lap_pos = pos_df[pos_df['Lap'] == selected_lap].sort_values('Position')
                    
                    if not lap_pos.empty:
                        fig = go.Figure()
                        for _, row in lap_pos.iterrows():
                            fig.add_trace(go.Bar(x=[100 - (row['Position']-1)*4], y=[row['Driver']], orientation='h',
                                marker_color=color_map.get(row['Driver'], '#888'), text=f"P{int(row['Position'])}",
                                textposition='inside', textfont=dict(color='white', size=12), showlegend=False))
                        
                        fig.update_layout(title=f"Lap {selected_lap}/{int(total_laps)}", xaxis=dict(visible=False, range=[0,110]),
                            yaxis=dict(categoryorder='array', categoryarray=lap_pos.sort_values('Position', ascending=False)['Driver'].tolist()),
                            height=max(400, len(lap_pos)*25), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
                    
                    # Position evolution
                    st.markdown("**Position Evolution**")
                    fig2 = go.Figure()
                    for driver in pos_df['Driver'].unique()[:15]:
                        drv_pos = pos_df[pos_df['Driver'] == driver].sort_values('Lap')
                        fig2.add_trace(go.Scatter(x=drv_pos['Lap'], y=drv_pos['Position'], mode='lines',
                            name=driver, line=dict(color=color_map.get(driver, '#888888'), width=2)))
                    fig2.update_layout(title="Position Changes", xaxis_title="Lap", yaxis_title="Position",
                        yaxis=dict(autorange='reversed', dtick=1), height=500, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 7: BATTLE ANALYSIS
    with analysis_tabs[6]:
        st.subheader("Battle Analysis")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                c1, c2 = st.columns(2)
                with c1:
                    battle_a = st.selectbox("Driver A", drivers_list, key="battle_a")
                with c2:
                    battle_b = st.selectbox("Driver B", [d for d in drivers_list if d != battle_a], key="battle_b")
                
                # Context Map
                with st.expander("Circuit Context", expanded=False):
                    ctx_fig = plot_circuit_context(session)
                    if ctx_fig: show_plotly_chart(ctx_fig, use_container_width=False)

                if st.button("Analyze Battle", type="primary", key="battle_btn"):
                    laps_a = laps[(laps['Driver'] == battle_a) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    laps_b = laps[(laps['Driver'] == battle_b) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    
                    if not laps_a.empty and not laps_b.empty:
                        team_a = laps_a['Team'].iloc[0] if 'Team' in laps_a.columns else ''
                        team_b = laps_b['Team'].iloc[0] if 'Team' in laps_b.columns else ''
                        color_a = TEAM_COLORS.get(team_a, '#FF4444')
                        color_b = TEAM_COLORS.get(team_b, '#44FF44')
                        
                        common_laps = set(laps_a['LapNumber']) & set(laps_b['LapNumber'])
                        faster_a, faster_b = 0, 0
                        
                        for lap in common_laps:
                            t_a = laps_a[laps_a['LapNumber'] == lap]['LapTime'].dt.total_seconds().iloc[0]
                            t_b = laps_b[laps_b['LapNumber'] == lap]['LapTime'].dt.total_seconds().iloc[0]
                            if t_a < t_b: faster_a += 1
                            else: faster_b += 1
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"""<div style="text-align:center;padding:20px;background:linear-gradient(135deg,{color_a}55 0%,{color_a}22 100%);border-radius:10px;border:2px solid {color_a};">
                                <h3 style="color:white;margin:0;">{battle_a}</h3><h1 style="color:{color_a};margin:10px 0;">{faster_a}</h1><p style="color:#888;">Faster Laps</p></div>""", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""<div style="text-align:center;padding:20px;background:#1a1a2e;border-radius:10px;border:2px solid #444;">
                                <h3 style="color:white;margin:0;">VS</h3><h2 style="color:#FFD700;margin:10px 0;">{len(common_laps)}</h2><p style="color:#888;">Laps</p></div>""", unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"""<div style="text-align:center;padding:20px;background:linear-gradient(135deg,{color_b}55 0%,{color_b}22 100%);border-radius:10px;border:2px solid {color_b};">
                                <h3 style="color:white;margin:0;">{battle_b}</h3><h1 style="color:{color_b};margin:10px 0;">{faster_b}</h1><p style="color:#888;">Faster Laps</p></div>""", unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=laps_a['LapNumber'], y=laps_a['LapTime'].dt.total_seconds(), name=battle_a,
                            line=dict(color=color_a, width=2)))
                        fig.add_trace(go.Scatter(x=laps_b['LapNumber'], y=laps_b['LapTime'].dt.total_seconds(), name=battle_b,
                            line=dict(color=color_b, width=2)))
                        fig.update_layout(title=f"{battle_a} vs {battle_b}", xaxis_title="Lap", yaxis_title="Time (s)",
                            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 8: STRATEGY TOOLS
    with analysis_tabs[7]:
        st.subheader("Strategy Analysis")
        try:
            # 1. Tyre Visuals (New)
            st.markdown("### Tyre Stint Analysis")
            s_laps = session.laps
            driver_sel = st.selectbox("Select Driver for Stint View", sorted(s_laps['Driver'].unique()), key="stint_viz_drv")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                # Plot Donut for Current Stint / All Stints ?
                # Let's plot the Start Tyres
                t_fig = plot_tyre_shape(session, driver_sel)
                if t_fig: show_plotly_chart(t_fig, use_container_width=True)
            
            with c2:
                # Existing Pit Detail
                pass 

            pit_stops = get_pit_stops(session)
            if pit_stops is not None and not pit_stops.empty:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Total Stops", len(pit_stops))
                with c2: st.metric("Avg Time", f"{pit_stops['PitTime'].mean():.1f}s")
                with c3: st.metric("Fastest", f"{pit_stops['PitTime'].min():.1f}s")
                with c4: st.metric("Popular Lap", f"L{int(pit_stops['Lap'].mode().iloc[0])}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pit_stops['Lap'], y=pit_stops['PitTime'], mode='markers',
                    marker=dict(size=12, color=pit_stops['PitTime'], colorscale='RdYlGn_r', showscale=True),
                    text=pit_stops['Driver'], hovertemplate="%{text}<br>Lap: %{x}<br>Time: %{y:.1f}s<extra></extra>"))
                fig.update_layout(title="Pit Stop Distribution", xaxis_title="Lap", yaxis_title="Time (s)",
                    height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                show_plotly_chart(fig, use_container_width=True)
                
                pit_laps = pit_stops.groupby('Lap').size().reset_index(name='Stops')
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=pit_laps['Lap'], y=pit_laps['Stops'], marker_color='#E10600'))
                fig2.update_layout(title="Pit Windows", xaxis_title="Lap", yaxis_title="Stops",
                    height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            else:
                st.info("No pit stop data")
        except Exception as e:
            st.error(f"Error: {e}")

        # --- GOD MODE SIMULATION INTEGRATION ---
        st.divider()
        st.markdown("### ⚡ Dynamic Strategy Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            base_lap = st.number_input("Base Lap Time (s)", 80.0, 120.0, 90.0, step=0.1, key="strat_base")
            total_laps_sim = st.number_input("Total Laps", 10, 80, 52, key="strat_tot")
        
        sim = RaceStrategySimulator(base_lap, total_laps_sim)
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1: driver_sim = st.text_input("Driver", "Verstappen", key="strat_drv")
        with sc2: start_tire = st.selectbox("Start Compound", ["SOFT", "MEDIUM", "HARD"], key="strat_tire")
        with sc3: current_lap_sim = st.slider("Current Lap", 0, int(total_laps_sim), 0, key="strat_lap")
        
        if st.button("Run Strategy Simulation", type="primary"):
            strategy = sim.predict_strategy(driver_sim, start_tire, current_lap_sim)
            st.success(f"Recommended: **{strategy['recommended']}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("1-Stop", f"{strategy['1_stop_time']:.1f}s")
            m2.metric("2-Stop", f"{strategy['2_stop_time']:.1f}s")
            m3.metric("Delta", f"{strategy['delta']:.1f}s")
            
            # Catch up Logic
            st.subheader("Catch-Up Prediction")
            cc1, cc2 = st.columns(2)
            with cc1: gap = st.number_input("Gap (s)", 0.0, 60.0, 5.0, key="strat_gap")
            with cc2: 
                chaser_tyre = st.selectbox("Chaser", ["SOFT", "MEDIUM"], key="strat_chaser")
                leader_tyre = st.selectbox("Leader", ["MEDIUM", "HARD"], index=1, key="strat_leader")
            
            laps_catch = sim.catch_up_prediction(gap, chaser_tyre, leader_tyre, total_laps_sim - current_lap_sim)
            if laps_catch != -1:
                st.info(f"🚀 Overtake in **{laps_catch} laps**")
            else:
                st.warning("⚠️ Overtake unlikely")
    
    # TAB 9: DRIVER SCORES
    with analysis_tabs[8]:
        st.subheader("Driver Performance Scores")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                scores = []
                session_best = laps['LapTime'].dt.total_seconds().min()
                field_median = laps['LapTime'].dt.total_seconds().median()
                
                for driver in laps['Driver'].unique():
                    drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())]
                    if len(drv_laps) >= 3:
                        times = drv_laps['LapTime'].dt.total_seconds()
                        pace = max(0, 100 - (times.min() - session_best) * 10)
                        consistency = max(0, 100 - times.std() * 20)
                        race_pace = max(0, 100 - (times.mean() - field_median) * 5)
                        overall = pace * 0.4 + consistency * 0.35 + race_pace * 0.25
                        team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                        scores.append({'Driver': driver, 'Team': team, 'Pace': round(pace,1), 
                            'Consistency': round(consistency,1), 'Race Pace': round(race_pace,1), 
                            'Overall': round(overall,1), 'Best': f"{times.min():.3f}"})
                
                if scores:
                    scores_df = pd.DataFrame(scores).sort_values('Overall', ascending=False)
                    scores_df['Rank'] = range(1, len(scores_df)+1)
                    
                    st.divider()
                    
                    # --- NEW: RADAR CHART ---
                    st.markdown("### 🕸️ Driver Capability Radar")
                    col_radar, col_table = st.columns([1, 1])
                    
                    with col_radar:
                        radar_drivers = st.multiselect("Compare Drivers", scores_df['Driver'].unique(), 
                                                     default=scores_df['Driver'].head(3).tolist() if len(scores_df)>=3 else scores_df['Driver'].tolist())
                        
                        if radar_drivers:
                            fig_radar = go.Figure()
                            for d in radar_drivers:
                                d_row = scores_df[scores_df['Driver'] == d].iloc[0]
                                d_team = d_row['Team']
                                d_color = TEAM_COLORS.get(d_team, '#888')
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[d_row['Pace'], d_row['Consistency'], d_row['Race Pace'], d_row['Overall']],
                                    theta=['Quali Pace', 'Consistency', 'Race Pace', 'Overall Rating'],
                                    fill='toself',
                                    name=d,
                                    line_color=d_color
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#444'),
                                    bgcolor='rgba(0,0,0,0)'
                                ),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                margin=dict(l=40, r=40, t=20, b=20),
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                            )
                            show_plotly_chart(fig_radar, use_container_width=True)
                    
                    with col_table:
                        st.markdown("### Leaderboard")
                        st.dataframe(
                            scores_df[['Rank','Driver','Team','Overall']].style.background_gradient(subset=['Overall'], cmap='plasma'), 
                            use_container_width=True, hide_index=True
                        )
                    
                    # Radar chart
                    fig = go.Figure()
                    for _, row in scores_df.head(5).iterrows():
                        fig.add_trace(go.Scatterpolar(r=[row['Pace'], row['Consistency'], row['Race Pace']],
                            theta=['Pace', 'Consistency', 'Race Pace'], fill='toself', name=row['Driver'],
                            line=dict(color=TEAM_COLORS.get(row['Team'], '#888888'))))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100]), bgcolor='rgba(0,0,0,0)'),
                        height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")


    # TAB 10: TELEMETRY COMPARE
    with analysis_tabs[9]:
        st.subheader("Driver Telemetry Comparison")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    drv1 = st.selectbox("Driver 1", drivers_list, key="tel_d1")
                with col2:
                    drv2 = st.selectbox("Driver 2", [d for d in drivers_list if d != drv1], key="tel_d2")
                with col3:
                    lap_mode = st.radio("Lap Selection", ["Fastest Lap", "Specific Lap"], horizontal=True, key="tel_mode")
                
                selected_lap = None
                if lap_mode == "Specific Lap":
                    max_laps = int(laps['LapNumber'].max())
                    selected_lap = st.slider("Select Lap", 1, max_laps, 1, key="tel_lap_sel")
                
                if st.button("Compare Telemetry", type="primary", key="tel_comp_btn"):
                    with st.spinner("Generating detailed telemetry..."):
                        fig = plot_telemetry_comparison(session, drv1, drv2, selected_lap)
                        if fig:
                            show_plotly_chart(fig, use_container_width=True)
                            
                            # Also show gear shift map
                            st.subheader("Gear Shift Analysis")
                            c1, c2 = st.columns(2)
                            with c1:
                                fig_g1 = plot_gear_shift_trace(session, drv1)
                                if fig_g1: show_plotly_chart(fig_g1, use_container_width=True)
                            with c2:
                                fig_g2 = plot_gear_shift_trace(session, drv2)
                                if fig_g2: show_plotly_chart(fig_g2, use_container_width=True)
                        else:
                            st.error("Could not generate telemetry plot")
            else:
                st.info("No session data")
        except Exception as e:
            st.error(f"Error: {e}")

    # TAB 10: RACE CONTROL
    with analysis_tabs[10]:
        st.subheader("Race Control Events")
        try:
             rc_msgs = get_race_control_messages(session)
             if not rc_msgs.empty:
                 # Color code flags
                 def highlight_flag(val):
                     color = ''
                     val_str = str(val).upper()
                     if 'RED' in val_str: color = 'background-color: #ff4b4b; color: white'
                     elif 'YELLOW' in val_str: color = 'background-color: #fca130; color: black'
                     elif 'GREEN' in val_str: color = 'background-color: #09ab3b; color: white'
                     elif 'BLACK' in val_str: color = 'background-color: black; color: white'
                     elif 'BLUE' in val_str: color = 'background-color: #0068c9; color: white'
                     return color

                 st.dataframe(rc_msgs.style.map(highlight_flag, subset=['Flag']), 
                     use_container_width=True, hide_index=True)
             else:
                 st.info("No race control messages available.")
        except Exception as e:
            st.error(f"Error loading race control: {e}")

    # TAB 11: ENGINEERING
    with analysis_tabs[11]:
        st.subheader("Engineering & Reliability")
        eng_tabs = st.tabs(["Pit Stop Details", "Tyre Degradation"])
        
        with eng_tabs[0]:
            st.markdown("#### Pit Stop Analysis")
            try:
                pit_detailed = get_detailed_pit_analysis(session)
                if not pit_detailed.empty:
                    st.dataframe(pit_detailed, use_container_width=True, hide_index=True)
                    
                    # Pit Duration Distribution
                    fig_pit = px.histogram(pit_detailed, x="Duration", nbins=20, 
                        title="Pit Stop Duration Distribution", color_discrete_sequence=['#00D2BE'])
                    fig_pit.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig_pit, use_container_width=True)
                else:
                    st.info("No pit detail data.")
            except Exception as e:
                st.error(f"Error in pit analysis: {e}")

        with eng_tabs[1]:
             st.markdown("#### Tyre Life Estimation")
             try:
                 laps = session.laps
                 if laps is not None and not laps.empty:
                    drivers = st.multiselect("Select Drivers", sorted(laps['Driver'].unique()), default=[laps['Driver'].iloc[0]], key="eng_tyre_drv")
                    if drivers:
                        fig_deg = go.Figure()
                        for d in drivers:
                            d_laps = laps.pick_driver(d).pick_quicklaps()
                            if not d_laps.empty:
                                fig_deg.add_trace(go.Scatter(x=d_laps['TyreLife'], y=d_laps['LapTime'].dt.total_seconds(),
                                    mode='markers', name=d))
                        fig_deg.update_layout(title="Lap Time vs Tyre Life", xaxis_title="Tyre Life (Laps)", 
                            yaxis_title="Time (s)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig_deg, use_container_width=True)
             except Exception as e:
                 st.error(f"Error in tyre degradation: {e}")


def render_prediction_tab(df, total_points_combined=None):
    """AI Race Predictions tab content."""
    st.header("AI Race Predictor")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.warning("Model not trained. Please train the model first (see documentation).")
        return

    # Get next race details from schedule
    try:
        schedule = fastf1.get_event_schedule(2025)
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        
        now = pd.Timestamp.now(tz='UTC')
        upcoming = schedule[schedule['EventDate'] >= (now - timedelta(days=3))].head(1)
        
        if not upcoming.empty:
            next_event = upcoming.iloc[0]
            race_name = next_event['EventName']
            st.info(f"Predicting for: {race_name}")
        else:
            race_name = "Upcoming Race"
            st.info("Predicting for next available race.")
            
    except Exception:
        race_name = "Next Race"

    # Prediction tabs
    pred_tabs = st.tabs(["Full Grid Prediction", "Model Insights"])
    
    with pred_tabs[0]:
        st.subheader(f"Predicted Results - {race_name}")
        
        if st.button("Generate Predictions", type="primary", key="gen_pred_btn"):
            with st.spinner("Running Random Forest Regressor..."):
                # Prepare input data
                drivers = sorted(df['Driver'].unique().tolist())
                driver_stats = calculate_driver_stats(df)
                
                predictions = []
                
                # Get encoders if available (would need to load them, but assuming prepare_features handles it or we recreate basic encoding for display)
                # Ideally we use the pipeline from model.py. 
                # For now, we'll construct the feature DF and assume encoders are handled or we pass raw if pipeline supports it.
                # Actually, prepare_features(train_mode=False) loads encoders.
                
                # Live Data Injection
                real_grid = get_real_grid_positions(2025, race_name)
                using_live = False
                
                if real_grid:
                    st.success(f"Using Live Qualifying Grid for {race_name}")
                    using_live = True
                else:
                    st.warning("Qualifying data unavailable. Using historical averages (accuracy lower).")

                # Create a dataframe for all drivers for the next race
                pred_rows = []
                for driver in drivers:
                    driver_data = df[df['Driver'] == driver]
                    if driver_data.empty:
                        continue
                        
                    team = driver_data['Team'].iloc[0]
                    
                    # Estimate grid position from season average
                    if driver_stats is not None and driver in driver_stats.index:
                        # avg_grid not in driver_stats by default, let's calculate it on the fly
                        avg_grid = driver_data['Starting Grid'].mean() if 'Starting Grid' in driver_data.columns else 10
                    else:
                        avg_grid = 10
                    
                    # INJECT LIVE GRID
                    if using_live:
                         start_pos = real_grid.get(driver, avg_grid)
                    else:
                         start_pos = round(avg_grid)

                    pred_rows.append({
                        'Driver': driver,
                        'Team': team,
                        'Starting Grid': start_pos,
                        'Track': race_name,
                        'Finished': True # Dummy for feature prep
                    })
                
                pred_df = pd.DataFrame(pred_rows)
                
                try:
                    # Transform features
                    X_pred, _, _ = prepare_features(pred_df, train_mode=False)
                    
                    # Predict
                    y_pred = model.predict(X_pred)
                    
                    pred_df['Predicted_Position'] = y_pred
                    pred_df = pred_df.sort_values('Predicted_Position')
                    pred_df['Rank'] = range(1, len(pred_df) + 1)
                    
                    # Display
                    st.dataframe(pred_df[['Rank', 'Driver', 'Team', 'Starting Grid', 'Predicted_Position']], 
                                 use_container_width=True, hide_index=True)
                    
                    # Winner
                    winner = pred_df.iloc[0]
                    st.success(f"Predicted Winner: {winner['Driver']} ({winner['Team']})")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.caption("Ensure categorical encoders (Driver, Team, Track) are generated via training.")

    with pred_tabs[1]:
        st.subheader("Prediction Factors")
        
        if hasattr(model, 'feature_importances_'):
            # Hardcoded feature names based on features.py
            feature_names = ['Starting Grid', 'Driver', 'Team', 'Track']
            
            if len(model.feature_importances_) == len(feature_names):
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importances, x='Importance', y='Feature', orientation='h',
                             title="Model Feature Importance")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                show_plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Key Factors:**
                * **Starting Grid:** Historical data shows qualifying performance is the strongest predictor.
                * **Driver/Team:** Adjusts for car performance relative to the field.
                * **Track:** Accounts for circuit-specific performance characteristics.
                """)
            else:
                st.info("Feature importance details unavailable.")


def render_telemetry_tab(df):
    """Live Telemetry Monitor tab content."""
    st.header("Live Telemetry Monitor")
    st.markdown("Engineer-style telemetry display with detailed car data")
    
    # Race and driver selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_races = F1_2025_COMPLETED_RACES
        selected_race = st.selectbox("Select Grand Prix", available_races, key="telem_race")
    
    with col2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"], key="telem_session")
    
    with col3:
        drivers = df['Driver'].unique().tolist() if df is not None else []
        selected_driver = st.selectbox("Select Driver", sorted(drivers), key="telem_driver")
    
    if st.button("Load Telemetry", type="primary"):
        st.divider()
        
        with st.spinner("Loading telemetry data..."):
            setup_fastf1_cache()
            session = load_fastf1_session(2025, selected_race, session_type)
        
        if session is None:
            st.error(f"Could not load session: {selected_race}")
            return
        
        # Get driver lap data
        try:
            # Get driver abbreviation
            driver_abbr = None
            for drv in session.drivers:
                drv_info = session.get_driver(drv)
                if selected_driver in str(drv_info.get('FullName', '')):
                    driver_abbr = drv_info.get('Abbreviation')
                    break
            
            if driver_abbr is None:
                # Try to match by last name
                for drv in session.drivers:
                    drv_info = session.get_driver(drv)
                    if selected_driver.split()[-1].upper() in str(drv_info.get('FullName', '')).upper():
                        driver_abbr = drv_info.get('Abbreviation')
                        break
            
            if driver_abbr is None:
                st.error(f"Could not find driver {selected_driver} in session")
                return
            
            # Get laps and car data
            driver_laps = session.laps.pick_driver(driver_abbr)
            
            if driver_laps.empty:
                st.error("No lap data available for this driver")
                return
            
            # Get fastest lap
            fastest_lap = driver_laps.pick_fastest()
            
            if fastest_lap is None or fastest_lap.empty:
                st.warning("No valid laps found, using first lap")
                fastest_lap = driver_laps.iloc[0] if len(driver_laps) > 0 else None
            
            if fastest_lap is None:
                st.error("Could not get lap data")
                return
            
            # Get telemetry
            telemetry = fastest_lap.get_telemetry()
            
            if telemetry is None or telemetry.empty:
                st.error("No telemetry data available")
                return
            
            # Get team color
            team = df[df['Driver'] == selected_driver]['Team'].iloc[0] if selected_driver in df['Driver'].values else "Unknown"
            team_color = TEAM_COLORS.get(team, '#E10600')
            
            # Header with lap info
            lap_time = format_f1_time(fastest_lap.get('LapTime', pd.NaT))
            lap_number = fastest_lap.get('LapNumber', 'N/A')
            compound = fastest_lap.get('Compound', 'N/A')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {team_color}33 0%, #1a1a2e 100%); 
                        padding: 20px; border-radius: 15px; border: 2px solid {team_color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <h4 style="color: #888; margin: 0;">Driver</h4>
                        <h2 style="color: white; margin: 5px 0;">{selected_driver}</h2>
                    </div>
                    <div>
                        <h4 style="color: #888; margin: 0;">Lap Time</h4>
                        <h2 style="color: {team_color}; margin: 5px 0;">{lap_time}</h2>
                    </div>
                    <div>
                        <h4 style="color: #888; margin: 0;">Lap</h4>
                        <h2 style="color: white; margin: 5px 0;">{lap_number}</h2>
                    </div>
                    <div>
                        <h4 style="color: #888; margin: 0;">Compound</h4>
                        <h2 style="color: white; margin: 5px 0;">{compound}</h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge row
            st.subheader("Real-time Gauges")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Get latest values
            latest_speed = telemetry['Speed'].iloc[-1] if 'Speed' in telemetry.columns else 0
            latest_throttle = telemetry['Throttle'].iloc[-1] if 'Throttle' in telemetry.columns else 0
            latest_brake = telemetry['Brake'].iloc[-1] if 'Brake' in telemetry.columns else 0
            latest_rpm = telemetry['RPM'].iloc[-1] if 'RPM' in telemetry.columns else 0
            
            max_speed = telemetry['Speed'].max() if 'Speed' in telemetry.columns else 350
            max_rpm = 15000
            
            with col1:
                fig = create_gauge(latest_speed, max_speed, "SPEED (km/h)", team_color)
                show_plotly_chart(fig, use_container_width=True, key="speed_gauge")
            
            with col2:
                fig = create_gauge(latest_throttle, 100, "THROTTLE (%)", "#00FF00")
                show_plotly_chart(fig, use_container_width=True, key="throttle_gauge")
            
            with col3:
                fig = create_gauge(latest_brake, 100, "BRAKE (%)", "#FF0000")
                show_plotly_chart(fig, use_container_width=True, key="brake_gauge")
            
            with col4:
                fig = create_gauge(latest_rpm, max_rpm, "RPM", "#FFD700")
                show_plotly_chart(fig, use_container_width=True, key="rpm_gauge")
            
            st.divider()
            
            # Speed trace
            st.subheader("Speed Trace")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=telemetry['Distance'] if 'Distance' in telemetry.columns else telemetry.index,
                y=telemetry['Speed'] if 'Speed' in telemetry.columns else [],
                mode='lines',
                name='Speed',
                line=dict(color=team_color, width=2),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(team_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
            ))
            
            fig.update_layout(
                height=300,
                xaxis_title="Distance (m)",
                yaxis_title="Speed (km/h)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            show_plotly_chart(fig, use_container_width=True)
            
            # Throttle/Brake trace
            st.subheader("Throttle & Brake Trace")
            
            fig = go.Figure()
            
            x_axis = telemetry['Distance'] if 'Distance' in telemetry.columns else telemetry.index
            
            if 'Throttle' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=telemetry['Throttle'],
                    mode='lines',
                    name='Throttle',
                    line=dict(color='#00FF00', width=2)
                ))
            
            if 'Brake' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=telemetry['Brake'] * 100 if telemetry['Brake'].max() <= 1 else telemetry['Brake'],
                    mode='lines',
                    name='Brake',
                    line=dict(color='#FF0000', width=2)
                ))
            
            fig.update_layout(
                height=250,
                xaxis_title="Distance (m)",
                yaxis_title="Input (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            show_plotly_chart(fig, use_container_width=True)
            
            # Gear and DRS
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gear Usage")
                if 'nGear' in telemetry.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=telemetry['nGear'],
                        mode='lines',
                        name='Gear',
                        line=dict(color='#FFD700', width=2)
                    ))
                    fig.update_layout(
                        height=200,
                        xaxis_title="Distance (m)",
                        yaxis_title="Gear",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    show_plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("DRS Status")
                if 'DRS' in telemetry.columns:
                    fig = go.Figure()
                    drs_values = telemetry['DRS'].apply(lambda x: 1 if x > 0 else 0)
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=drs_values,
                        mode='lines',
                        name='DRS',
                        line=dict(color='#00BFFF', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0,191,255,0.3)'
                    ))
                    fig.update_layout(
                        height=200,
                        xaxis_title="Distance (m)",
                        yaxis_title="DRS Active",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    show_plotly_chart(fig, use_container_width=True)
            
            # Track Map with Speed
            st.subheader("Track Map - Speed Visualization")
            
            if 'X' in telemetry.columns and 'Y' in telemetry.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=telemetry['X'],
                    y=telemetry['Y'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=telemetry['Speed'] if 'Speed' in telemetry.columns else 'white',
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='Speed (km/h)')
                    ),
                    hovertemplate='Speed: %{marker.color:.0f} km/h<extra></extra>'
                ))
                
                fig.update_layout(
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, scaleanchor='x'),
                    showlegend=False
                )
                show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Track position data not available for this lap")
            
        except Exception as e:
            st.error(f"Error loading telemetry: {e}")
            logger.error(f"Telemetry error: {e}", exc_info=True)


def render_live_timing_tab():
    """Live Timing Session tab content."""
    st.header("Live Timing Session") # Removed emoji
    
    # Current time (UTC)
    now = pd.Timestamp.now(tz='UTC')
    st.markdown(f"**Current System Time (UTC):** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get schedule for 2025
    try:
        schedule = fastf1.get_event_schedule(2025)
        
        # Ensure schedule dates are timezone-aware UTC for comparison
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')

        # Find next or current event (filter for events happening today or in future)
        # We look for events where the last session is in the future or today
        upcoming = schedule[schedule['EventDate'] >= (now - timedelta(days=3))].head(1)
        
        if not upcoming.empty:
            next_event = upcoming.iloc[0]
            st.subheader(f"Target Event: {next_event['EventName']}")
            st.write(f"Location: {next_event['Location']}")
            
            # Display sessions
            sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
            
            schedule_data = []
            active_session_name = None
            
            for s in sessions:
                s_date_col = f'{s}Date'
                s_name_col = s
                
                if s_date_col in next_event and s_name_col in next_event:
                    s_date = next_event[s_date_col]
                    s_name = next_event[s_name_col]
                    
                    if pd.notna(s_date):
                        # Ensure s_date is timezone aware for comparison
                        if isinstance(s_date, pd.Timestamp) and s_date.tz is None:
                            s_date = s_date.tz_localize('UTC')

                        status = "Upcoming"
                        # Check if currently active (assuming 2h duration)
                        if isinstance(s_date, pd.Timestamp):
                            s_end = s_date + timedelta(hours=2)
                            if s_date <= now <= s_end:
                                status = "LIVE" # Removed emoji
                            elif now > s_end:
                                status = "Completed"
                        
                        schedule_data.append({
                            'Session': s_name, 
                            'Time': s_date.strftime('%Y-%m-%d %H:%M') if pd.notna(s_date) else 'TBD',
                            'Status': status
                        })
            
            st.table(pd.DataFrame(schedule_data))
            
            # Race Week Notification - Removed emojis
            event_date = next_event['EventDate']
            if isinstance(event_date, pd.Timestamp):
                if event_date.tz is None:
                    event_date = event_date.tz_localize('UTC')
                
                days_diff = (event_date - now).days
                if -1 <= days_diff <= 7:
                    st.markdown(f"""
                    <div style="background-color: #E10600; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; overflow: hidden; white-space: nowrap;">
                        <div style="display: inline-block; animation: marquee 15s linear infinite; font-weight: bold; font-size: 1.2em;">
                            IT'S RACE WEEK! {next_event['EventName'].upper()} IS COMING UP! GET READY FOR THE ACTION!
                        </div>
                    </div>
                    <style>
                    @keyframes marquee {{
                        0% {{ transform: translateX(100%); }}
                        100% {{ transform: translateX(-100%); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)
            
            # Connection controls
            st.divider()
            st.markdown("### Live Data Connection")
            
            col1, col2 = st.columns(2)
            with col1:
                default_idx = 0
                if active_session_name:
                    try:
                        default_idx = [s['Session'] for s in schedule_data].index(active_session_name)
                    except:
                        pass
                session_sel = st.selectbox("Select Session to Monitor", [s['Session'] for s in schedule_data], index=default_idx)
            with col2:
                auto_refresh = st.checkbox("Auto-refresh (Simulated)", value=False)
                if auto_refresh:
                    st.caption("Page will refresh interaction to update data")
                else:
                    if st.button("Refresh Now"): # Removed emoji
                        pass  # Just triggers rerun
            
            # Automatic connection if session is selected
            if session_sel:
                with st.spinner(f"Connecting to {next_event['EventName']} - {session_sel}..."):
                    try:
                        # Attempt to load
                        setup_fastf1_cache()
                        session = fastf1.get_session(2025, next_event['EventName'], session_sel)
                        
                        # For live sessions, we might get partial data
                        # We suppress errors for live data fetch
                        try:
                            session.load(telemetry=False, weather=True, messages=True)
                        except Exception as e:
                            st.warning(f"Full load failed (expected if live): {e}")
                            # Try minimal load
                            pass
                        
                        st.success(f"Connected to {session.name}")
                        
                        # Dashboard
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Session Status", "Connected")
                        with c2:
                            if hasattr(session, 'weather_data') and session.weather_data is not None and not session.weather_data.empty:
                                temp = session.weather_data['TrackTemp'].iloc[-1]
                                st.metric("Track Temp", f"{temp:.1f}°C")
                            else:
                                st.metric("Track Temp", "N/A")
                        with c3:
                            if hasattr(session, 'drivers'):
                                st.metric("Drivers", len(session.drivers))
                            else:
                                st.metric("Drivers", "0")
                        with c4:
                            st.metric("Last Update", now.strftime('%H:%M:%S'))
                            
                        st.divider()
                        
                        # Leaderboard
                        st.subheader("Live Leaderboard")
                        if hasattr(session, 'results') and session.results is not None and not session.results.empty:
                            cols = ['Position', 'Abbreviation', 'Time', 'Status', 'Points']
                            valid_cols = [c for c in cols if c in session.results.columns]
                            st.dataframe(session.results[valid_cols], hide_index=True, use_container_width=True)
                        else:
                            st.info("Waiting for timing data... (Results not yet available)")
                        
                        # Latest Messages
                        st.subheader("Race Control Messages")
                        messages = get_track_status(session)
                        if not messages.empty:
                            st.dataframe(messages.tail(10).sort_values('Time', ascending=False), hide_index=True, use_container_width=True)
                        else:
                            st.info("No messages received.")
                            
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
                        st.info("Note: FastF1 requires the session to be started to fetch data.")
        else:
            st.info("No upcoming events found for 2025 (or schedule API unavailable).")
            
    except Exception as e:
        st.error(f"Error loading schedule: {e}")


def render_qualifying_tab():
    """Qualifying Analysis tab content."""
    st.header("Qualifying Deep Dive")
    st.markdown("Analyze qualifying performance, lap evolution, and sector dominance.")
    
    col1, col2 = st.columns(2)
    with col1:
        q_race = st.selectbox("Select Race for Qualifying Analysis", F1_2025_COMPLETED_RACES, key="q_race")
    
    if st.button("Analyze Qualifying", type="primary"):
        with st.spinner(f"Loading {q_race} Qualifying Data..."):
            setup_fastf1_cache()
            session = load_fastf1_session(2025, q_race, "Qualifying")
            
            if session:
                st.divider()
                
                # 1. Evolution Plot
                st.subheader("Lap Time Evolution")
                fig_evo = plot_qualifying_evolution(session)
                if fig_evo: show_plotly_chart(fig_evo, use_container_width=True)
                else: st.info("Evolution data unavailable")
                
                col1, col2 = st.columns(2)
                
                # 2. Gap to Pole
                with col1:
                    st.subheader("Gap to Pole")
                    fig_gap = plot_qualifying_gap(session)
                    if fig_gap: show_plotly_chart(fig_gap, use_container_width=True)
                    
                # 3. Sector Dominance
                with col2:
                    st.subheader("Sector Dominance")
                    fig_sec = plot_sector_dominance(session)
                    if fig_sec: show_plotly_chart(fig_sec, use_container_width=True)
            else:
                st.error("Could not load session")


def render_teammate_battle_tab(df):
    """Teammate Battle tab content."""
    st.header("Teammate Head-to-Head Wars")
    st.markdown("Comparing performance between teammates across the season.")
    
    if df is None or df.empty:
        st.error("No season data available")
        return
        
    comparison_df = calculate_teammate_comparison(df)
    
    if comparison_df.empty:
        st.warning("Not enough data to generate comparisons.")
        return
        
    # Team Selector
    teams = sorted(comparison_df['Team'].unique())
    selected_team = st.selectbox("Select Team to Compare", teams, key="tm_comp_team")
    
    if selected_team:
        team_data_row = comparison_df[comparison_df['Team'] == selected_team].iloc[0] # Use a different var name
        
        d1 = team_data_row['Driver 1']
        d2 = team_data_row['Driver 2']
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"<h2 style='text-align: center; color: {TEAM_COLORS.get(selected_team, 'white')}'>{d1}</h2>", unsafe_allow_html=True)
            prof1 = DRIVER_PROFILES.get(d1, {})
            if prof1.get('image_url'):
                st.image(prof1['image_url'], use_container_width=True)
                
        with col2:
            st.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
            
            comp_metrics = [
                ("Races Together", team_data_row['Races Together'], team_data_row['Races Together']),
                ("Race Head-to-Head", team_data_row['D1 Race Wins'], team_data_row['D2 Race Wins']),
                ("Quali Head-to-Head", team_data_row['Quali H2H'].split(' - ')[0], team_data_row['Quali H2H'].split(' - ')[1]),
                ("Total Points", int(team_data_row['Pts 1']), int(team_data_row['Pts 2']))
            ]
            
            for label, v1, v2 in comp_metrics:
                c_a, c_b, c_c = st.columns([1, 2, 1])
                with c_a: st.markdown(f"<h3 style='text-align: right;'>{v1}</h3>", unsafe_allow_html=True)
                with c_b: st.markdown(f"<p style='text-align: center; color: #888;'>{label}</p>", unsafe_allow_html=True)
                with c_c: st.markdown(f"<h3 style='text-align: left;'>{v2}</h3>", unsafe_allow_html=True)
                st.divider()

        with col3:
            st.markdown(f"<h2 style='text-align: center; color: {TEAM_COLORS.get(selected_team, 'white')}'>{d2}</h2>", unsafe_allow_html=True)
            prof2 = DRIVER_PROFILES.get(d2, {})
            if prof2.get('image_url'):
                st.image(prof2['image_url'], use_container_width=True)
                
        with st.expander("View All Teammate Comparisons"):
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_official_plots_tab():
    """Official F1 Style Plots tab content."""
    st.header("Official F1 Style Plots")
    st.markdown("High-quality static visualizations using FastF1's matplotlib integration.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_race = st.selectbox("Select Race", F1_2025_COMPLETED_RACES, key="plot_race")
    with col2:
        plot_session = st.selectbox("Session Type", ["Race", "Qualifying"], key="plot_session")
    with col3:
        st.markdown("Driver specific plots:")
        plot_driver = st.text_input("Driver Code (e.g., VER, HAM)", value="VER", key="plot_driver")
        
    if st.button("Generate Plots", type="primary"):
        with st.spinner("Generating plots..."):
            setup_fastf1_cache()
            session = load_fastf1_session(2025, plot_race, plot_session)
            if session:
                st.subheader("Circuit with Corners")
                fig1 = plot_circuit_with_corners(session)
                if fig1: st.pyplot(fig1)
                
                st.subheader("Team Pace Comparison")
                fig2 = plot_team_pace_comparison(session)
                if fig2: st.pyplot(fig2)
                
                st.subheader("Tyre Strategy Summary")
                fig3 = plot_tyre_strategy_summary(session)
                if fig3: st.pyplot(fig3)
                
                st.subheader(f"Speed Visualization ({plot_driver})")
                fig4 = plot_speed_on_track(session, plot_driver)
                if fig4: st.pyplot(fig4)
                
                st.subheader(f"Gear Shift Visualization ({plot_driver})")
                fig5 = plot_gear_shift_on_track(session, plot_driver)
                if fig5: st.pyplot(fig5)
            else:
                st.error("Could not load session.")


def main():
    """Main application entry point."""
    render_header()
    
    # Load data
    df = load_race_data()
    
    # Calculate total points properly (race + sprint)
    total_points_combined = {}
    total_laps_all = 0
    total_all_points = 0
    if df is not None:
        data_dir = Path(__file__).parent.parent / 'data'
        try:
            race_df = pd.read_csv(data_dir / 'Formula1_2025Season_RaceResults.csv')
            sprint_df = pd.read_csv(data_dir / 'Formula1_2025Season_SprintResults.csv')
            
            race_points = race_df.groupby('Driver')['Points'].sum()
            sprint_points = sprint_df.groupby('Driver')['Points'].sum()
            total_points_combined = race_points.add(sprint_points, fill_value=0).to_dict()
            total_laps_all = race_df['Laps'].sum()
            total_all_points = sum(total_points_combined.values())
        except:
            total_points_combined = df.groupby('Driver')['Points'].sum().to_dict()
            total_laps_all = df['Laps'].sum() if 'Laps' in df.columns else 0
            total_all_points = sum(total_points_combined.values())
    
    # Sidebar
    with st.sidebar:
        # Removed emoji from image source
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/512px-F1.svg.png", width=80) 
        st.title("F1 2025")
        st.markdown("---")
        
        if df is not None:
            st.markdown(f"**Races:** {len(F1_2025_COMPLETED_RACES)}/24")
            st.markdown(f"**Total Points:** {int(total_all_points):,}")
        
        st.markdown("---")
        st.caption("Created by **Maxvy**")

    # Main Navigation (Grouped Tabs) - Removed emojis
    main_tabs = st.tabs([
        "Home",
        "Season Stats", 
        "Race Center",
        "Analysis",
        "Live"
    ])
    
    # 1. Home
    with main_tabs[0]:
        render_home_tab()
    
    # 2. Season Stats
    with main_tabs[1]:
        st.subheader("Season Statistics")
        season_subtabs = st.tabs(["Overview", "Drivers", "Constructors"])
        
        with season_subtabs[0]:
            render_overview_tab(df, total_points_combined)
        with season_subtabs[1]:
            render_drivers_tab(df, total_points_combined)
        with season_subtabs[2]:
            render_teams_tab(df)
            
    # 3. Race Center
    with main_tabs[2]:
        st.subheader("Race Weekend Center")
        race_subtabs = st.tabs([
            "Weekend Details", 
            "Race Analysis", 
            "Qualifying", 
            "Telemetry",
            "Race Replay",
            "Official Plots",
            "Export Data"
        ])
        
        with race_subtabs[0]:
            render_race_detail_tab(df)
        with race_subtabs[1]:
            render_race_analysis_tab(df)
        with race_subtabs[2]:
            render_qualifying_tab()
        with race_subtabs[3]:
            render_telemetry_tab(df)
        with race_subtabs[4]:
            render_race_replay_tab()
        with race_subtabs[5]:
            render_official_plots_tab()
        with race_subtabs[6]:
            # Data Export Logic Inline
            st.subheader("Data Export")
            st.markdown("Export session data to CSV for external analysis.")
            c1, c2 = st.columns(2)
            with c1:
                ex_race = st.selectbox("Select Race", F1_2025_COMPLETED_RACES, key="export_race")
            with c2:
                ex_session = st.selectbox("Session Type", ["Race", "Qualifying", "Sprint"], key="export_session")
                
            if st.button("Prepare Export", type="primary"):
                with st.spinner("Processing data..."):
                    setup_fastf1_cache()
                    session = load_fastf1_session(2025, ex_race, ex_session)
                    if session:
                        exports = export_session_to_csv(session)
                        cols = st.columns(3)
                        if 'laps' in exports:
                            cols[0].download_button("Download Laps", exports['laps'], f"{ex_race}_laps.csv", "text/csv")
                        if 'results' in exports:
                            cols[1].download_button("Download Results", exports['results'], f"{ex_race}_results.csv", "text/csv")
                        if 'weather' in exports:
                            cols[2].download_button("Download Weather", exports['weather'], f"{ex_race}_weather.csv", "text/csv")
                        st.success("Data ready for download!")
                    else:
                        st.error("Could not load session.")

    # 4. Analysis
    with main_tabs[3]:
        st.subheader("Advanced Analysis")
        # removed God Mode separate tab link, moved content to Strategy Tools
        analysis_subtabs = st.tabs(["Teammate Battle", "Race Predictions"])
        
        with analysis_subtabs[0]:
            render_teammate_battle_tab(df)
        with analysis_subtabs[1]:
            render_prediction_tab(df, total_points_combined)
            
    # 5. Live
    with main_tabs[4]:
        render_live_timing_tab()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Critical Error: {e}")
        st.code(traceback.format_exc())
