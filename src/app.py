"""
F1 2025 Season Dashboard - Comprehensive Analytics
Professional dashboard with driver profiles, race details, telemetry, and predictions.
2025 Season Only.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from datetime import datetime
import logging
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from config import TEAM_COLORS, DRIVER_PROFILES, F1_2025_COMPLETED_RACES, F1_2025_CALENDAR
from loader import load_data as load_csv_data, clean_data, load_combined_data
from analysis import calculate_driver_stats, calculate_team_stats, calculate_combined_constructor_standings
from config import DATA_FILES
from fastf1_extended import (
    get_session_info, get_weather_summary, get_tyre_stints,
    get_pit_stops, get_sector_times, get_speed_data,
    get_track_status, get_car_data, get_position_data,
    get_position_changes, get_flag_events,
    get_race_results, get_session_schedule, get_gaps_to_leader
)

# Page config
st.set_page_config(
    page_title="F1 2025 Analytics Dashboard",
    page_icon="https://cdn-icons-png.flaticon.com/512/2418/2418779.png",
    layout="wide",
    initial_sidebar_state="expanded"
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


@st.cache_data(ttl=3600)
def load_fastf1_session(year: int, race: str, session_type: str):
    """Load FastF1 session with caching."""
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load()
        return session
    except Exception as e:
        logger.error(f"Error loading FastF1 session: {e}")
        return None


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


def render_header():
    """Render dashboard header."""
    st.markdown('<h1 class="main-header">F1 2025 Season Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Analytics | Race Data | Telemetry | Predictions</p>', unsafe_allow_html=True)


def render_overview_tab(df, total_points_combined=None):
    """Tab 1: Season Overview."""
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
                margin=dict(l=10, r=80, t=10, b=40)
            )
            st.plotly_chart(fig, width='stretch')
    
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
                margin=dict(l=10, r=80, t=10, b=40)
            )
            st.plotly_chart(fig, width='stretch')
    
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
        st.plotly_chart(fig, width='stretch')


def render_drivers_tab(df, total_points_combined=None):
    """Tab 2: Driver Profiles with Photos and Career Stats."""
    st.header("Driver Profiles")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Get drivers from 2025 data (exclude Jack Doohan who is no longer racing)
    drivers_2025 = [d for d in df['Driver'].unique().tolist() if d != 'Jack Doohan']
    
    # Driver selector with session state to preserve selection
    selected_driver = st.selectbox(
        "Select Driver", 
        sorted(drivers_2025),
        key="driver_profile_selector"
    )
    
    if selected_driver:
        # Get driver profile from config
        profile = DRIVER_PROFILES.get(selected_driver, {})
        driver_df = df[df['Driver'] == selected_driver]
        team = driver_df['Team'].iloc[0] if not driver_df.empty else "Unknown"
        team_color = TEAM_COLORS.get(team, '#666666')
        
        st.divider()
        
        # Profile section
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            # Driver photo - no border, clean look
            if profile.get('image_url'):
                st.markdown(f"""
                <div style="text-align: center;">
                    <img src="{profile['image_url']}" 
                         style="width: 200px; height: 200px; border-radius: 15px; 
                                object-fit: cover; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <h2 style="color: white; margin-top: 15px;">{selected_driver}</h2>
                    <p style="color: {team_color}; font-size: 1.2rem; font-weight: bold;">{team}</p>
                    <p style="color: #888; font-size: 2rem; font-weight: bold;">#{profile.get('number', '')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 200px; height: 200px; border-radius: 15px; 
                                background: linear-gradient(135deg, {team_color}55 0%, #1a1a2e 100%);
                                display: flex; align-items: center; justify-content: center;
                                margin: 0 auto; font-size: 4rem; color: white;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                        {selected_driver[0]}
                    </div>
                    <h2 style="color: white; margin-top: 15px;">{selected_driver}</h2>
                    <p style="color: {team_color}; font-size: 1.2rem; font-weight: bold;">{team}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Biography")
            
            # Driver info - use correct field names from config
            number = profile.get('number', 'N/A')
            country = profile.get('country', 'Unknown')
            debut = profile.get('debut_year', 'Unknown')
            dob = profile.get('date_of_birth', 'Unknown')
            pob = profile.get('place_of_birth', 'Unknown')
            bio = profile.get('bio', '')
            
            st.markdown(f"""
            | Info | Value |
            |------|-------|
            | **Number** | {number} |
            | **Nationality** | {country} |
            | **Date of Birth** | {dob} |
            | **Place of Birth** | {pob} |
            | **F1 Debut** | {debut} |
            | **Current Team** | {team} |
            | **Seasons in F1** | {2025 - debut if isinstance(debut, int) else 'N/A'} |
            """)
            
            if bio:
                st.markdown(f"**Bio:** {bio}")
        
        with col3:
            st.subheader("Career Statistics (All Time)")
            
            # Career metrics - use correct field names from config
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Races", profile.get('career_points', 0) // 10 if profile.get('career_points') else 0)  # Estimate
                st.metric("Poles", profile.get('career_poles', 0))
            with c2:
                st.metric("Wins", profile.get('career_wins', 0))
                st.metric("Fastest Laps", profile.get('career_fastest_laps', 0))
            with c3:
                st.metric("Podiums", profile.get('career_podiums', 0))
                st.metric("Championships", profile.get('championships', 0))
        
        st.divider()
        
        # 2025 Season Stats
        st.subheader("2025 Season Performance")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Season metrics - use combined points if available
            race_points = driver_df['Points'].sum()
            # Get total combined points (race + sprint)
            if total_points_combined and selected_driver in total_points_combined:
                total_points = total_points_combined[selected_driver]
            else:
                total_points = race_points
            
            races = len(driver_df)
            avg_position = driver_df['Position'].mean()
            wins_2025 = len(driver_df[driver_df['Position'] == 1])
            podiums_2025 = len(driver_df[driver_df['Position'] <= 3])
            best_finish = driver_df['Position'].min()
            avg_pts_per_race = total_points / races if races > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Races", races)
                st.metric("Wins", wins_2025)
                st.metric("Podiums", podiums_2025)
            with c2:
                st.metric("Best", f"P{int(best_finish)}")
                st.metric("Avg Pts/Race", f"{avg_pts_per_race:.1f}")
                st.metric("Total Points", int(total_points))
        
        with col2:
            # Position chart for 2025
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=driver_df['Track'],
                y=driver_df['Position'],
                mode='lines+markers',
                line=dict(color=team_color, width=3),
                marker=dict(size=12, color=team_color),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(team_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
            ))
            
            fig.update_layout(
                title="2025 Race Positions",
                yaxis=dict(autorange='reversed', title='Position'),
                xaxis=dict(title='Race', tickangle=45),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, width='stretch')
        
        # Points per race
        st.subheader("Points Scored per Race")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=driver_df['Track'],
            y=driver_df['Points'],
            marker_color=team_color,
            text=driver_df['Points'],
            textposition='outside'
        ))
        fig.update_layout(
            height=300,
            xaxis=dict(tickangle=45),
            yaxis_title="Points",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, width='stretch')
        
        # All 2025 results table
        st.subheader("2025 Race Results")
        display_cols = ['Track', 'Position', 'Points', 'Laps'] if 'Laps' in driver_df.columns else ['Track', 'Position', 'Points']
        st.dataframe(
            driver_df[display_cols].reset_index(drop=True),
            width='stretch',
            hide_index=True
        )


def render_teams_tab(df):
    """Tab 3: Team Analysis with car photos and specs."""
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
                st.image(team_info['car_image'], width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')
        
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
        st.plotly_chart(fig, width='stretch')


def render_race_detail_tab(df):
    """Tab 4: Race Details with FastF1 Data."""
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
                rain_status = "🌧️ Yes" if weather.get('rainfall', False) else "☀️ No"
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
                    display_results.columns = ['Pos', 'Code', 'Driver', 'Team', 'Time', 'Status']
                    st.dataframe(display_results, width='stretch', hide_index=True)
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
                st.plotly_chart(fig, width='stretch')
                
                # Detailed pit stop table
                st.markdown("**Detailed Pit Stop Data**")
                display_pit = pit_stops.copy()
                display_pit['PitTime'] = display_pit['PitTime'].apply(lambda x: f"{x:.1f}s")
                display_pit = display_pit.rename(columns={'Stop': 'Stop #', 'PitTime': 'Duration'})
                st.dataframe(display_pit, width='stretch', hide_index=True)
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
                # Tyre compound colors and emojis
                compound_colors = {
                    'SOFT': '#FF3333',
                    'MEDIUM': '#FFD700', 
                    'HARD': '#FFFFFF',
                    'INTERMEDIATE': '#43B02A',
                    'WET': '#0067AD',
                    'UNKNOWN': '#888888'
                }
                
                # Sort drivers by finishing position if available
                try:
                    results = session.results
                    if results is not None and not results.empty:
                        driver_order = results.sort_values('Position')['Abbreviation'].tolist()
                    else:
                        driver_order = tyre_data['Driver'].unique().tolist()
                except:
                    driver_order = tyre_data['Driver'].unique().tolist()
                
                # Filter to top 20 drivers
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
                        
                        fig.add_trace(go.Bar(
                            x=[laps],
                            y=[driver],
                            orientation='h',
                            base=start - 1,
                            marker_color=color,
                            marker_line_color='#333333',
                            marker_line_width=1,
                            name=compound,
                            showlegend=False,
                            text=f"{compound[0]}" if compound != 'UNKNOWN' else "?",
                            textposition='inside',
                            textfont=dict(color='black' if compound in ['MEDIUM', 'HARD'] else 'white', size=10),
                            hovertemplate=f"<b>{driver}</b><br>{compound}<br>Laps {start}-{end} ({laps} laps)<extra></extra>"
                        ))
                
                total_laps = tyre_data['EndLap'].max() if 'EndLap' in tyre_data.columns else 50
                
                fig.update_layout(
                    title="Tyre Strategy - Race Overview",
                    xaxis_title="Lap",
                    xaxis=dict(range=[0, total_laps + 2]),
                    height=max(450, len(drivers) * 25),
                    barmode='overlay',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(categoryorder='array', categoryarray=drivers[::-1])
                )
                st.plotly_chart(fig, width='stretch')
                
                # Compound Legend with visual
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown("🔴 **SOFT**")
                with col2:
                    st.markdown("🟡 **MEDIUM**")
                with col3:
                    st.markdown("⚪ **HARD**")
                with col4:
                    st.markdown("🟢 **INTERMEDIATE**")
                with col5:
                    st.markdown("🔵 **WET**")
                
                # Stint details table
                with st.expander("View Detailed Stint Data"):
                    display_tyres = tyre_data[['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'Laps']].copy()
                    display_tyres['Stint'] = display_tyres['Stint'] + 1
                    display_tyres = display_tyres.rename(columns={'Stint': 'Stint #', 'StartLap': 'Start', 'EndLap': 'End'})
                    st.dataframe(display_tyres, width='stretch', hide_index=True)
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
                st.plotly_chart(fig, width='stretch')
            
            if flag_events is not None and len(flag_events) > 0:
                st.subheader("Flag Events")
                flag_df = pd.DataFrame(flag_events)
                st.dataframe(flag_df, width='stretch', hide_index=True)
            else:
                st.info("Flag event data not available")
        
        with race_tabs[4]:
            # Sector Times
            st.subheader("Sector Times Analysis")
            
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
                                        name=f'S{i}',
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
                        st.plotly_chart(fig, width='stretch')
                        
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
                        margin=dict(l=10, r=80, t=40, b=40),
                        xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1)
                    )
                    st.plotly_chart(fig, width='stretch')
                
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
                    st.plotly_chart(fig2, width='stretch')
                
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
                
                st.dataframe(display_df, width='stretch', hide_index=True)
            else:
                st.info("Position change data not available")
        
        with race_tabs[6]:
            # Lap Times Analysis
            st.subheader("🏁 Lap Time Analysis")
            
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
                        st.plotly_chart(fig, width='stretch')
                        
                        # Fastest laps table
                        st.markdown("**Fastest Laps:**")
                        fastest_laps = laps.groupby('Driver').apply(
                            lambda x: x.nsmallest(1, 'LapTime')
                        ).reset_index(drop=True)
                        
                        if not fastest_laps.empty:
                            display_cols = ['Driver', 'LapNumber', 'LapTime', 'Compound']
                            available_cols = [c for c in display_cols if c in fastest_laps.columns]
                            fastest_display = fastest_laps[available_cols].sort_values('LapTime').head(10)
                            st.dataframe(fastest_display, width='stretch', hide_index=True)
                else:
                    st.info("Lap time data not available")
            except Exception as e:
                st.error(f"Error loading lap times: {e}")


def render_race_analysis_tab(df):
    """Tab 5: Comprehensive Race Analysis with advanced features."""
    st.header("Race Analysis Center")
    st.markdown("*Lap times, Pace comparison, Strategy analysis, 2D Track Animation*")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Race selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_race = st.selectbox("Select Grand Prix", F1_2025_COMPLETED_RACES, key="analysis_race")
    with col2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"], key="analysis_session")
    
    # Analysis sub-tabs
    analysis_tabs = st.tabs([
        "Lap Analysis", "Pace Comparison", "Stint Analysis", 
        "Gap Chart", "2D Track Animation", "Position Chart", "Battle Analysis",
        "Strategy Tools", "Driver Scores"
    ])
    
    # Load session
    with st.spinner("Loading session data..."):
        setup_fastf1_cache()
        session = load_fastf1_session(2025, selected_race, session_type)
    
    if session is None:
        st.error(f"Could not load session: {selected_race}")
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
                    st.plotly_chart(fig, width='stretch')
                    
                    # Box plot
                    fig2 = go.Figure()
                    for driver in selected_drivers:
                        drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())]
                        if not drv_laps.empty:
                            times = drv_laps['LapTime'].dt.total_seconds()
                            team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                            fig2.add_trace(go.Box(y=times, name=driver, marker_color=TEAM_COLORS.get(team, '#888888'), boxmean=True))
                    fig2.update_layout(title="Lap Time Distribution", yaxis_title="Time (s)", height=350,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig2, width='stretch')
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
                        st.plotly_chart(fig, width='stretch')
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
                        laps_count = end - start + 1
                        color = compound_colors.get(compound, '#888888')
                        
                        fig.add_trace(go.Bar(x=[laps_count], y=[driver], orientation='h', base=start-1,
                            marker_color=color, marker_line_color='#333', marker_line_width=1,
                            showlegend=False, text=compound[0] if compound != 'UNKNOWN' else '?',
                            textposition='inside', textfont=dict(color='black' if compound in ['MEDIUM','HARD'] else 'white', size=10)))
                
                total_laps = tyre_data['EndLap'].max() if 'EndLap' in tyre_data.columns else 50
                fig.update_layout(title="Tyre Strategy", xaxis_title="Lap", xaxis=dict(range=[0, total_laps+2]),
                    height=max(450, len(drivers)*25), barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    yaxis=dict(categoryorder='array', categoryarray=drivers[::-1]))
                st.plotly_chart(fig, width='stretch')
                
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.markdown("🔴 **SOFT**")
                with c2: st.markdown("🟡 **MEDIUM**")
                with c3: st.markdown("⚪ **HARD**")
                with c4: st.markdown("🟢 **INTER**")
                with c5: st.markdown("🔵 **WET**")
            else:
                st.info("No tyre data")
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
                        st.plotly_chart(fig, width='stretch')
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 5: 2D TRACK ANIMATION - Real Car Movement
    with analysis_tabs[4]:
        st.subheader("2D Track Animation - Live Race Replay")
        
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
                                
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(f"Animation: {len(interpolated_data)} drivers, {num_points} frames. Click Play to start.")
            
            # Show static track preview
            st.markdown("---")
            st.markdown("**Track Layout**")
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
                                st.plotly_chart(fig_preview, use_container_width=True)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 6: POSITION CHART
    with analysis_tabs[5]:
        st.subheader("Position Chart")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                pos_data = []
                for lap_num in sorted(laps['LapNumber'].unique()):
                    lap = laps[laps['LapNumber'] == lap_num]
                    for _, row in lap.iterrows():
                        if 'Position' in row and pd.notna(row['Position']):
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
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                    st.plotly_chart(fig2, use_container_width=True)
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
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 8: STRATEGY TOOLS
    with analysis_tabs[7]:
        st.subheader("Strategy Analysis")
        try:
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
                st.plotly_chart(fig, width='stretch')
                
                pit_laps = pit_stops.groupby('Lap').size().reset_index(name='Stops')
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=pit_laps['Lap'], y=pit_laps['Stops'], marker_color='#E10600'))
                fig2.update_layout(title="Pit Windows", xaxis_title="Lap", yaxis_title="Stops",
                    height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("No pit stop data")
        except Exception as e:
            st.error(f"Error: {e}")
    
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
                    
                    # Podium
                    podium = st.columns(3)
                    medals = ['🥇', '🥈', '🥉']
                    for i, (_, row) in enumerate(scores_df.head(3).iterrows()):
                        tc = TEAM_COLORS.get(row['Team'], '#666')
                        with podium[i]:
                            st.markdown(f"""<div style="text-align:center;padding:15px;background:linear-gradient(135deg,{tc}55 0%,{tc}22 100%);border-radius:10px;border:2px solid {tc};">
                                <h2 style="margin:0;">{medals[i]}</h2><h4 style="color:white;margin:5px 0;">{row['Driver']}</h4>
                                <p style="color:{tc};margin:0;font-weight:bold;">{row['Overall']:.1f}</p></div>""", unsafe_allow_html=True)
                    
                    st.dataframe(scores_df[['Rank','Driver','Team','Pace','Consistency','Race Pace','Overall','Best']], 
                        width='stretch', hide_index=True)
                    
                    # Radar chart
                    fig = go.Figure()
                    for _, row in scores_df.head(5).iterrows():
                        fig.add_trace(go.Scatterpolar(r=[row['Pace'], row['Consistency'], row['Race Pace']],
                            theta=['Pace', 'Consistency', 'Race Pace'], fill='toself', name=row['Driver'],
                            line=dict(color=TEAM_COLORS.get(row['Team'], '#888888'))))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100]), bgcolor='rgba(0,0,0,0)'),
                        height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")


def render_prediction_tab(df, total_points_combined=None):
    """Tab 6: AI Race Predictions - Only for Abu Dhabi Grand Prix (final race)."""
    st.header("AI Race Predictor - Abu Dhabi Grand Prix")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    st.info("**Season Finale:** Abu Dhabi Grand Prix is the only remaining race of 2025!")
    
    # Get drivers (exclude Jack Doohan who is no longer racing)
    drivers = sorted([d for d in df['Driver'].unique().tolist() if d != 'Jack Doohan'])
    
    # Prediction tabs
    pred_tabs = st.tabs(["Race Prediction", "Full Grid Prediction", "Championship Outlook"])
    
    with pred_tabs[0]:
        st.subheader("Individual Driver Prediction - Abu Dhabi")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_driver = st.selectbox("Select Driver", drivers, key="pred_driver_abu")
            predict_btn = st.button("Generate Prediction", type="primary", key="pred_btn_abu")
        
        with col2:
            if predict_btn and selected_driver:
                driver_df = df[df['Driver'] == selected_driver]
                
                if len(driver_df) < 2:
                    st.warning("Not enough data for prediction")
                else:
                    # Calculate stats
                    avg_pos = driver_df['Position'].mean()
                    std_pos = driver_df['Position'].std() if len(driver_df) > 1 else 3
                    best_pos = driver_df['Position'].min()
                    wins = len(driver_df[driver_df['Position'] == 1])
                    podiums = len(driver_df[driver_df['Position'] <= 3])
                    
                    # Get total points from combined
                    if total_points_combined and selected_driver in total_points_combined:
                        total_pts = total_points_combined[selected_driver]
                    else:
                        total_pts = driver_df['Points'].sum()
                    
                    avg_grid = driver_df['Starting Grid'].mean() if 'Starting Grid' in driver_df.columns else avg_pos
                    
                    # Predictions
                    pred_quali = max(1, int(round(avg_grid * 0.95)))
                    pred_race = max(1, int(round(avg_pos)))
                    confidence = max(50, min(95, 100 - std_pos * 8))
                    podium_chance = max(5, min(95, (4 - avg_pos) * 25)) if avg_pos <= 6 else max(5, 30 - avg_pos * 2)
                    
                    team = driver_df['Team'].iloc[0]
                    team_color = TEAM_COLORS.get(team, '#666666')
                    
                    st.markdown(f"### {selected_driver} - {team}")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Quali Prediction", f"P{pred_quali}")
                    with col_b:
                        st.metric("Race Prediction", f"P{pred_race}")
                    with col_c:
                        st.metric("Podium Chance", f"{podium_chance:.0f}%")
                    with col_d:
                        st.metric("Confidence", f"{confidence:.0f}%")
                    
                    st.divider()
                    
                    # Season stats
                    st.markdown("**2025 Season Stats**")
                    stat_cols = st.columns(6)
                    races = len(driver_df)
                    avg_pts = total_pts / races if races > 0 else 0
                    
                    with stat_cols[0]:
                        st.metric("Races", races)
                    with stat_cols[1]:
                        st.metric("Wins", wins)
                    with stat_cols[2]:
                        st.metric("Podiums", podiums)
                    with stat_cols[3]:
                        st.metric("Best", f"P{int(best_pos)}")
                    with stat_cols[4]:
                        st.metric("Avg Pts/Race", f"{avg_pts:.1f}")
                    with stat_cols[5]:
                        st.metric("Total Points", int(total_pts))
    
    with pred_tabs[1]:
        st.subheader("Full Race Prediction - Abu Dhabi")
        
        if st.button("Predict Full Grid", type="primary", key="pred_btn_full"):
            # Calculate predictions for all drivers
            predictions = []
            
            for driver in drivers:
                driver_df = df[df['Driver'] == driver]
                if len(driver_df) >= 1:
                    avg_pos = driver_df['Position'].mean()
                    std_pos = driver_df['Position'].std() if len(driver_df) > 1 else 3
                    team = driver_df['Team'].iloc[0]
                    
                    if total_points_combined and driver in total_points_combined:
                        pts = total_points_combined[driver]
                    else:
                        pts = driver_df['Points'].sum()
                    
                    predictions.append({
                        'Driver': driver,
                        'Team': team,
                        'Predicted': avg_pos,
                        'Total_Points': pts,
                        'Confidence': max(50, min(95, 100 - std_pos * 8))
                    })
            
            pred_df = pd.DataFrame(predictions).sort_values('Predicted')
            pred_df['Position'] = range(1, len(pred_df) + 1)
            
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            pred_df['Race_Points'] = pred_df['Position'].map(points_map).fillna(0)
            
            # Podium
            st.markdown("**Predicted Podium**")
            podium_cols = st.columns(3)
            medals = ['🥇', '🥈', '🥉']
            
            for i, (_, row) in enumerate(pred_df.head(3).iterrows()):
                team_color = TEAM_COLORS.get(row['Team'], '#666666')
                with podium_cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, {team_color}55 0%, {team_color}22 100%); border-radius: 10px; border: 2px solid {team_color};">
                        <h2 style="margin: 0;">{medals[i]}</h2>
                        <h4 style="color: white; margin: 5px 0;">{row['Driver']}</h4>
                        <p style="color: #888; margin: 0; font-size: 0.9em;">{row['Team']}</p>
                        <p style="color: #FFD700; margin: 5px 0;">+{int(row['Race_Points'])} pts</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # Full results
            st.markdown("**Full Predicted Results**")
            display_df = pred_df[['Position', 'Driver', 'Team', 'Race_Points', 'Confidence']].copy()
            display_df.columns = ['Pos', 'Driver', 'Team', 'Pts', 'Conf %']
            st.dataframe(display_df, width='stretch', hide_index=True)
    
    with pred_tabs[2]:
        st.subheader("Championship Outlook After Abu Dhabi")
        
        # Get current standings with combined points
        standings = []
        for driver in drivers:
            driver_df = df[df['Driver'] == driver]
            if len(driver_df) >= 1:
                if total_points_combined and driver in total_points_combined:
                    pts = total_points_combined[driver]
                else:
                    pts = driver_df['Points'].sum()
                
                team = driver_df['Team'].iloc[0]
                wins = len(driver_df[driver_df['Position'] == 1])
                
                standings.append({
                    'Driver': driver,
                    'Team': team,
                    'Points': pts,
                    'Wins': wins
                })
        
        standings_df = pd.DataFrame(standings).sort_values('Points', ascending=False)
        standings_df['Position'] = range(1, len(standings_df) + 1)
        
        st.markdown("**Current Championship Standings (23/24 Races)**")
        
        # Top 5 visualization
        top_5 = standings_df.head(5)
        
        fig = go.Figure()
        for _, row in top_5.iterrows():
            team_color = TEAM_COLORS.get(row['Team'], '#666666')
            fig.add_trace(go.Bar(
                x=[row['Driver']],
                y=[row['Points']],
                marker_color=team_color,
                text=f"{int(row['Points'])} pts",
                textposition='outside',
                name=row['Driver']
            ))
        
        fig.update_layout(
            title="Top 5 Drivers - Current Points",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
        
        # Full standings table
        display_standings = standings_df[['Position', 'Driver', 'Team', 'Points', 'Wins']].head(20)
        display_standings.columns = ['Pos', 'Driver', 'Team', 'Points', 'Wins']
        st.dataframe(display_standings, width='stretch', hide_index=True)
        
        # Points gap analysis
        st.markdown("**Points Gap to Leader**")
        leader_pts = standings_df.iloc[0]['Points']
        gaps = []
        for _, row in standings_df.head(10).iterrows():
            gaps.append({
                'Driver': row['Driver'],
                'Gap': leader_pts - row['Points']
            })
        
        gap_df = pd.DataFrame(gaps)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=gap_df['Driver'],
            y=gap_df['Gap'],
            marker_color=['#FFD700'] + ['#888888'] * (len(gap_df) - 1),
            text=[f"-{int(g)}" if g > 0 else "Leader" for g in gap_df['Gap']],
            textposition='outside'
        ))
        fig2.update_layout(
            title="Points Behind Leader",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig2, width='stretch')


def render_telemetry_tab(df):
    """Tab 6: Engineer-Style Telemetry Display."""
    st.header("Live Telemetry Monitor")
    st.markdown("*Engineer-style telemetry display with detailed car data*")
    
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
            lap_time = fastest_lap.get('LapTime', 'N/A')
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
                st.plotly_chart(fig, width='stretch', key="speed_gauge")
            
            with col2:
                fig = create_gauge(latest_throttle, 100, "THROTTLE (%)", "#00FF00")
                st.plotly_chart(fig, width='stretch', key="throttle_gauge")
            
            with col3:
                fig = create_gauge(latest_brake, 100, "BRAKE (%)", "#FF0000")
                st.plotly_chart(fig, width='stretch', key="brake_gauge")
            
            with col4:
                fig = create_gauge(latest_rpm, max_rpm, "RPM", "#FFD700")
                st.plotly_chart(fig, width='stretch', key="rpm_gauge")
            
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
            st.plotly_chart(fig, width='stretch')
            
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
            st.plotly_chart(fig, width='stretch')
            
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
                    st.plotly_chart(fig, width='stretch')
            
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
                    st.plotly_chart(fig, width='stretch')
            
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
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Track position data not available for this lap")
            
        except Exception as e:
            st.error(f"Error loading telemetry: {e}")
            logger.error(f"Telemetry error: {e}", exc_info=True)


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
        st.image("https://cdn-icons-png.flaticon.com/512/2418/2418779.png", width=80)
        st.title("F1 2025")
        st.markdown("---")
        
        if df is not None:
            st.markdown(f"**Races:** {len(F1_2025_COMPLETED_RACES)}/24")
            st.markdown(f"**Circuits Raced:** {df['Track'].nunique()}")
            st.markdown(f"**Total Race Laps:** {total_laps_all:,}")
            st.markdown(f"**Total Points:** {int(total_all_points):,}")
            st.markdown(f"**Active Drivers:** {df['Driver'].nunique()}")
            st.markdown(f"**Teams:** {df['Team'].nunique()}")
        
        st.markdown("---")
        st.markdown("**Race Replay**")
        st.markdown("Go to **Race Analysis** tab")
        st.markdown("Select 2D Track Animation")
        st.markdown("for live position replay")
        
        st.markdown("---")
        st.caption("Created by **Maxvy**")
    
    # Main tabs - expanded with new analysis features
    tabs = st.tabs([
        "Overview",
        "Drivers", 
        "Teams",
        "Race Details",
        "Race Analysis",
        "AI Predictor",
        "Telemetry"
    ])
    
    with tabs[0]:
        render_overview_tab(df, total_points_combined)
    
    with tabs[1]:
        render_drivers_tab(df, total_points_combined)
    
    with tabs[2]:
        render_teams_tab(df)
    
    with tabs[3]:
        render_race_detail_tab(df)
    
    with tabs[4]:
        render_race_analysis_tab(df)
    
    with tabs[5]:
        render_prediction_tab(df, total_points_combined)
    
    with tabs[6]:
        render_telemetry_tab(df)


if __name__ == "__main__":
    main()

