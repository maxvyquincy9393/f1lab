"""
F1 2025 Season Dashboard - Dash Version
Professional dashboard with driver profiles, race details, telemetry, and predictions.
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import local modules
from config import TEAM_COLORS, DRIVER_PROFILES, F1_2025_COMPLETED_RACES, F1_2025_CALENDAR
from loader import load_data as load_csv_data, clean_data, load_combined_data
from analysis import calculate_driver_stats, calculate_team_stats, calculate_combined_constructor_standings
from config import DATA_FILES

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="F1 2025 Analytics Dashboard"
)

server = app.server  # For deployment

# Custom CSS
custom_css = """
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #E10600;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #888;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #E10600;
}
.metric-label {
    font-size: 0.9rem;
    color: #888;
}
.driver-card {
    background: #15151e;
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #333;
}
.nav-tabs .nav-link {
    color: #888 !important;
    border: none;
    padding: 15px 25px;
}
.nav-tabs .nav-link.active {
    color: #E10600 !important;
    background: transparent;
    border-bottom: 3px solid #E10600;
}
"""

# =====================
# DATA LOADING
# =====================

def load_race_data():
    """Load race data."""
    try:
        data_dir = str(Path(DATA_FILES.race_results).parent)
        df = load_combined_data(data_dir)
        if df is not None:
            df = clean_data(df)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_sprint_data():
    """Load sprint data."""
    try:
        sprint_file = Path(DATA_FILES.race_results).parent / 'Formula1_2025Season_SprintResults.csv'
        if sprint_file.exists():
            df = load_csv_data(str(sprint_file))
            if df is not None:
                df = clean_data(df)
            return df
        return None
    except Exception as e:
        print(f"Error loading sprint data: {e}")
        return None

# Load data on startup
df_race = load_race_data()
df_sprint = load_sprint_data()

# =====================
# COMPONENTS
# =====================

def create_metric_card(value, label, icon=None):
    """Create a metric card component."""
    return dbc.Card(
        dbc.CardBody([
            html.I(className=f"fas {icon} fa-2x mb-2", style={"color": "#E10600"}) if icon else None,
            html.H2(value, className="metric-value"),
            html.P(label, className="metric-label")
        ]),
        className="metric-card h-100"
    )

def create_header():
    """Create dashboard header."""
    return html.Div([
        html.H1("🏎️ F1 2025 Season Dashboard", className="main-header"),
        html.P("Comprehensive Analytics | Race Data | Telemetry | Predictions", className="sub-header")
    ], className="mb-4")

def create_navbar():
    """Create navigation tabs."""
    return dbc.Tabs(
        id="main-tabs",
        active_tab="overview",
        children=[
            dbc.Tab(label="📊 Overview", tab_id="overview"),
            dbc.Tab(label="👤 Drivers", tab_id="drivers"),
            dbc.Tab(label="🏢 Teams", tab_id="teams"),
            dbc.Tab(label="🏁 Race Details", tab_id="race-details"),
            dbc.Tab(label="📈 Predictions", tab_id="predictions"),
        ],
        className="mb-4"
    )

# =====================
# TAB CONTENTS
# =====================

def render_overview_tab():
    """Render Overview tab content."""
    if df_race is None or df_race.empty:
        return html.Div("No data available", className="text-danger")
    
    total_races = df_race['Track'].nunique()
    total_drivers = df_race['Driver'].nunique()
    total_teams = df_race['Team'].nunique()
    total_laps = df_race['Laps'].sum() if 'Laps' in df_race.columns else 0
    total_points = df_race['Points'].sum()
    
    # Driver standings
    driver_stats = calculate_driver_stats(df_race)
    driver_stats_display = driver_stats.reset_index() if driver_stats is not None else pd.DataFrame()
    top_drivers = driver_stats_display.nlargest(10, 'Total_Points') if not driver_stats_display.empty else pd.DataFrame()
    
    # Constructor standings
    if df_sprint is not None:
        constructor_standings = calculate_combined_constructor_standings(df_race, df_sprint)
    else:
        constructor_standings = calculate_team_stats(df_race)
    team_stats_display = constructor_standings.reset_index() if constructor_standings is not None else pd.DataFrame()
    team_stats_display = team_stats_display.sort_values('Total_Points', ascending=False).head(10) if not team_stats_display.empty else pd.DataFrame()
    
    # Driver teams mapping
    driver_teams = df_race.groupby('Driver')['Team'].first().to_dict()
    
    # Drivers chart
    if not top_drivers.empty:
        colors_drivers = [TEAM_COLORS.get(driver_teams.get(drv, ''), '#666666') for drv in top_drivers['Driver']]
        fig_drivers = go.Figure(go.Bar(
            x=top_drivers['Total_Points'],
            y=top_drivers['Driver'],
            orientation='h',
            marker_color=colors_drivers,
            text=top_drivers['Total_Points'].astype(int),
            textposition='outside'
        ))
        fig_drivers.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Points",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=10, r=80, t=10, b=40)
        )
    else:
        fig_drivers = go.Figure()
    
    # Teams chart
    if not team_stats_display.empty:
        colors_teams = [TEAM_COLORS.get(team, '#666666') for team in team_stats_display['Team']]
        fig_teams = go.Figure(go.Bar(
            x=team_stats_display['Total_Points'],
            y=team_stats_display['Team'],
            orientation='h',
            marker_color=colors_teams,
            text=team_stats_display['Total_Points'].astype(int),
            textposition='outside'
        ))
        fig_teams.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Points",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=10, r=80, t=10, b=40)
        )
    else:
        fig_teams = go.Figure()
    
    # Points progression chart
    top_5_drivers = driver_stats_display.nlargest(5, 'Total_Points')['Driver'].tolist() if not driver_stats_display.empty else []
    fig_progression = go.Figure()
    
    for driver in top_5_drivers:
        driver_data = df_race[df_race['Driver'] == driver].copy()
        driver_data = driver_data.sort_values('Track')
        cumsum_points = driver_data['Points'].cumsum()
        team = driver_data['Team'].iloc[0] if not driver_data.empty else ''
        color = TEAM_COLORS.get(team, '#666666')
        
        fig_progression.add_trace(go.Scatter(
            x=list(range(1, len(cumsum_points) + 1)),
            y=cumsum_points,
            name=driver,
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))
    
    fig_progression.update_layout(
        height=400,
        xaxis_title="Race Number",
        yaxis_title="Cumulative Points",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return html.Div([
        html.H3("2025 Season Overview", className="mb-4"),
        
        # Metrics row
        dbc.Row([
            dbc.Col(create_metric_card(total_races, "Races Completed", "fa-flag-checkered"), md=2),
            dbc.Col(create_metric_card(total_drivers, "Drivers", "fa-user"), md=2),
            dbc.Col(create_metric_card(total_teams, "Teams", "fa-users"), md=2),
            dbc.Col(create_metric_card(f"{total_laps:,}", "Total Laps", "fa-sync"), md=3),
            dbc.Col(create_metric_card(f"{int(total_points):,}", "Total Points", "fa-trophy"), md=3),
        ], className="mb-4"),
        
        html.Hr(),
        
        html.H4("Championship Standings", className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.H5("Drivers Championship", className="text-center mb-3"),
                dcc.Graph(figure=fig_drivers, config={'displayModeBar': False})
            ], md=6),
            dbc.Col([
                html.H5("Constructors Championship", className="text-center mb-3"),
                dcc.Graph(figure=fig_teams, config={'displayModeBar': False})
            ], md=6),
        ]),
        
        html.Hr(),
        
        html.H4("Championship Points Progression", className="mb-3"),
        dcc.Graph(figure=fig_progression, config={'displayModeBar': False})
    ])

def render_drivers_tab():
    """Render Drivers tab content."""
    if df_race is None or df_race.empty:
        return html.Div("No data available", className="text-danger")
    
    drivers_2025 = sorted([d for d in df_race['Driver'].unique().tolist() if d != 'Jack Doohan'])
    
    return html.Div([
        html.H3("Driver Profiles", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='driver-selector',
                    options=[{'label': d, 'value': d} for d in drivers_2025],
                    value=drivers_2025[0] if drivers_2025 else None,
                    placeholder="Select a driver...",
                    className="mb-4"
                )
            ], md=4)
        ]),
        
        html.Div(id='driver-profile-content')
    ])

def render_teams_tab():
    """Render Teams tab content."""
    if df_race is None or df_race.empty:
        return html.Div("No data available", className="text-danger")
    
    teams = sorted(df_race['Team'].unique().tolist())
    
    return html.Div([
        html.H3("Team Analysis", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='team-selector',
                    options=[{'label': t, 'value': t} for t in teams],
                    value=teams[0] if teams else None,
                    placeholder="Select a team...",
                    className="mb-4"
                )
            ], md=4)
        ]),
        
        html.Div(id='team-profile-content')
    ])

def render_race_details_tab():
    """Render Race Details tab content."""
    if df_race is None or df_race.empty:
        return html.Div("No data available", className="text-danger")
    
    races = df_race['Track'].unique().tolist()
    
    return html.Div([
        html.H3("Race Details & Analysis", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='race-selector',
                    options=[{'label': r, 'value': r} for r in races],
                    value=races[-1] if races else None,
                    placeholder="Select a race...",
                    className="mb-4"
                )
            ], md=4)
        ]),
        
        html.Div(id='race-details-content')
    ])

def render_predictions_tab():
    """Render Predictions tab content."""
    return html.Div([
        html.H3("Race Predictions & ML Analysis", className="mb-4"),
        
        dbc.Card([
            dbc.CardBody([
                html.H5("🔮 Coming Soon", className="text-center"),
                html.P("Machine learning predictions for race outcomes, championship projections, and more.", 
                       className="text-center text-muted")
            ])
        ], className="driver-card")
    ])

# =====================
# LAYOUT
# =====================

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + custom_css + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dbc.Container([
        create_header(),
        create_navbar(),
        html.Div(id="tab-content", className="mt-4")
    ], fluid=True, className="py-4")
])

# =====================
# CALLBACKS
# =====================

@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    if active_tab == "overview":
        return render_overview_tab()
    elif active_tab == "drivers":
        return render_drivers_tab()
    elif active_tab == "teams":
        return render_teams_tab()
    elif active_tab == "race-details":
        return render_race_details_tab()
    elif active_tab == "predictions":
        return render_predictions_tab()
    return html.Div("Select a tab")

@callback(
    Output("driver-profile-content", "children"),
    Input("driver-selector", "value")
)
def update_driver_profile(selected_driver):
    """Update driver profile based on selection."""
    if not selected_driver or df_race is None:
        return html.Div()
    
    profile = DRIVER_PROFILES.get(selected_driver, {})
    driver_df = df_race[df_race['Driver'] == selected_driver]
    team = driver_df['Team'].iloc[0] if not driver_df.empty else "Unknown"
    team_color = TEAM_COLORS.get(team, '#666666')
    
    # 2025 stats
    total_points_2025 = driver_df['Points'].sum()
    races_2025 = len(driver_df)
    wins_2025 = len(driver_df[driver_df['Position'] == 1])
    podiums_2025 = len(driver_df[driver_df['Position'] <= 3])
    avg_pos = driver_df['Position'].mean()
    best_pos = driver_df['Position'].min()
    
    # Results chart
    driver_results = driver_df.copy()
    fig_results = go.Figure()
    fig_results.add_trace(go.Bar(
        x=driver_results['Track'],
        y=driver_results['Points'],
        marker_color=team_color,
        text=driver_results['Points'].astype(int),
        textposition='outside'
    ))
    fig_results.update_layout(
        height=300,
        xaxis_title="Race",
        yaxis_title="Points",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=20, b=100)
    )
    
    # Position chart
    fig_positions = go.Figure()
    fig_positions.add_trace(go.Scatter(
        x=driver_results['Track'],
        y=driver_results['Position'],
        mode='lines+markers',
        line=dict(color=team_color, width=2),
        marker=dict(size=10, color=team_color)
    ))
    fig_positions.update_layout(
        height=300,
        xaxis_title="Race",
        yaxis_title="Finish Position",
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=20, b=100)
    )
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Photo & Basic Info
                dbc.Col([
                    html.Div([
                        html.Img(
                            src=profile.get('image_url', ''),
                            style={
                                "width": "200px", "height": "200px", 
                                "borderRadius": "15px", "objectFit": "cover",
                                "boxShadow": "0 4px 15px rgba(0,0,0,0.3)"
                            }
                        ) if profile.get('image_url') else html.Div(
                            selected_driver[0],
                            style={
                                "width": "200px", "height": "200px",
                                "borderRadius": "15px",
                                "background": f"linear-gradient(135deg, {team_color}55 0%, #1a1a2e 100%)",
                                "display": "flex", "alignItems": "center", "justifyContent": "center",
                                "fontSize": "4rem", "color": "white"
                            }
                        ),
                        html.H3(selected_driver, className="mt-3 text-white"),
                        html.P(team, style={"color": team_color, "fontSize": "1.2rem", "fontWeight": "bold"}),
                        html.P(f"#{profile.get('number', '')}", className="text-muted", 
                               style={"fontSize": "2rem", "fontWeight": "bold"})
                    ], className="text-center")
                ], md=3),
                
                # Biography
                dbc.Col([
                    html.H4("Biography", className="mb-3"),
                    html.Table([
                        html.Tr([html.Td("Number", style={"color": "#888"}), html.Td(profile.get('number', 'N/A'))]),
                        html.Tr([html.Td("Nationality", style={"color": "#888"}), html.Td(profile.get('country', 'Unknown'))]),
                        html.Tr([html.Td("Date of Birth", style={"color": "#888"}), html.Td(profile.get('date_of_birth', 'Unknown'))]),
                        html.Tr([html.Td("Place of Birth", style={"color": "#888"}), html.Td(profile.get('place_of_birth', 'Unknown'))]),
                        html.Tr([html.Td("F1 Debut", style={"color": "#888"}), html.Td(profile.get('debut_year', 'Unknown'))]),
                        html.Tr([html.Td("Current Team", style={"color": "#888"}), html.Td(team)]),
                    ], className="table table-sm text-white"),
                    html.P(profile.get('bio', ''), className="text-muted mt-3") if profile.get('bio') else None
                ], md=4),
                
                # Career Stats
                dbc.Col([
                    html.H4("Career Statistics", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H3(profile.get('career_wins', 0), className="text-danger"),
                            html.P("Wins", className="text-muted")
                        ], className="text-center"),
                        dbc.Col([
                            html.H3(profile.get('career_podiums', 0), className="text-warning"),
                            html.P("Podiums", className="text-muted")
                        ], className="text-center"),
                        dbc.Col([
                            html.H3(profile.get('career_poles', 0), className="text-info"),
                            html.P("Poles", className="text-muted")
                        ], className="text-center"),
                        dbc.Col([
                            html.H3(profile.get('championships', 0), className="text-success"),
                            html.P("Championships", className="text-muted")
                        ], className="text-center"),
                    ])
                ], md=5),
            ]),
            
            html.Hr(),
            
            html.H4("2025 Season Performance", className="mb-3"),
            
            dbc.Row([
                dbc.Col(create_metric_card(int(total_points_2025), "Points"), md=2),
                dbc.Col(create_metric_card(races_2025, "Races"), md=2),
                dbc.Col(create_metric_card(wins_2025, "Wins"), md=2),
                dbc.Col(create_metric_card(podiums_2025, "Podiums"), md=2),
                dbc.Col(create_metric_card(f"{avg_pos:.1f}", "Avg Position"), md=2),
                dbc.Col(create_metric_card(int(best_pos), "Best Finish"), md=2),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Points per Race", className="text-center"),
                    dcc.Graph(figure=fig_results, config={'displayModeBar': False})
                ], md=6),
                dbc.Col([
                    html.H5("Finish Positions", className="text-center"),
                    dcc.Graph(figure=fig_positions, config={'displayModeBar': False})
                ], md=6),
            ])
        ])
    ], className="driver-card")

@callback(
    Output("team-profile-content", "children"),
    Input("team-selector", "value")
)
def update_team_profile(selected_team):
    """Update team profile based on selection."""
    if not selected_team or df_race is None:
        return html.Div()
    
    team_df = df_race[df_race['Team'] == selected_team]
    team_color = TEAM_COLORS.get(selected_team, '#666666')
    
    # Team stats
    total_points = team_df['Points'].sum()
    if df_sprint is not None:
        sprint_team_df = df_sprint[df_sprint['Team'] == selected_team]
        total_points += sprint_team_df['Points'].sum()
    
    drivers = team_df['Driver'].unique().tolist()
    total_races = team_df['Track'].nunique()
    wins = len(team_df[team_df['Position'] == 1])
    podiums = len(team_df[team_df['Position'] <= 3])
    
    # Driver comparison
    driver_points = team_df.groupby('Driver')['Points'].sum().sort_values(ascending=False)
    
    fig_drivers = go.Figure()
    fig_drivers.add_trace(go.Pie(
        labels=driver_points.index.tolist(),
        values=driver_points.values,
        marker=dict(colors=[team_color, f"{team_color}88"]),
        hole=0.4
    ))
    fig_drivers.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1)
    )
    
    # Points by race
    race_points = team_df.groupby('Track')['Points'].sum().reset_index()
    
    fig_race_points = go.Figure()
    fig_race_points.add_trace(go.Bar(
        x=race_points['Track'],
        y=race_points['Points'],
        marker_color=team_color
    ))
    fig_race_points.update_layout(
        height=300,
        xaxis_title="Race",
        yaxis_title="Points",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=20, b=100)
    )
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(
                            style={
                                "width": "150px", "height": "150px",
                                "borderRadius": "15px",
                                "background": team_color,
                                "margin": "0 auto"
                            }
                        ),
                        html.H3(selected_team, className="mt-3 text-white text-center"),
                        html.P(f"Drivers: {', '.join(drivers)}", className="text-muted text-center")
                    ])
                ], md=3),
                
                dbc.Col([
                    dbc.Row([
                        dbc.Col(create_metric_card(int(total_points), "Total Points"), md=3),
                        dbc.Col(create_metric_card(total_races, "Races"), md=3),
                        dbc.Col(create_metric_card(wins, "Wins"), md=3),
                        dbc.Col(create_metric_card(podiums, "Podiums"), md=3),
                    ])
                ], md=9)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Driver Points Distribution", className="text-center"),
                    dcc.Graph(figure=fig_drivers, config={'displayModeBar': False})
                ], md=5),
                dbc.Col([
                    html.H5("Points per Race", className="text-center"),
                    dcc.Graph(figure=fig_race_points, config={'displayModeBar': False})
                ], md=7),
            ])
        ])
    ], className="driver-card")

@callback(
    Output("race-details-content", "children"),
    Input("race-selector", "value")
)
def update_race_details(selected_race):
    """Update race details based on selection."""
    if not selected_race or df_race is None:
        return html.Div()
    
    race_df = df_race[df_race['Track'] == selected_race].copy()
    race_df = race_df.sort_values('Position')
    
    # Results table
    results_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Pos"),
                html.Th("Driver"),
                html.Th("Team"),
                html.Th("Points"),
                html.Th("Laps"),
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(row['Position']),
                html.Td(row['Driver']),
                html.Td(row['Team'], style={"color": TEAM_COLORS.get(row['Team'], '#888')}),
                html.Td(row['Points']),
                html.Td(row.get('Laps', '-')),
            ]) for _, row in race_df.iterrows()
        ])
    ], bordered=True, dark=True, striped=True, className="mb-4")
    
    # Points distribution chart
    fig_points = go.Figure()
    colors = [TEAM_COLORS.get(team, '#666666') for team in race_df['Team']]
    fig_points.add_trace(go.Bar(
        x=race_df['Driver'],
        y=race_df['Points'],
        marker_color=colors,
        text=race_df['Points'].astype(int),
        textposition='outside'
    ))
    fig_points.update_layout(
        height=400,
        xaxis_title="Driver",
        yaxis_title="Points",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-45
    )
    
    return dbc.Card([
        dbc.CardBody([
            html.H4(f"🏁 {selected_race} Grand Prix", className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Race Results"),
                    results_table
                ], md=6),
                dbc.Col([
                    html.H5("Points Distribution"),
                    dcc.Graph(figure=fig_points, config={'displayModeBar': False})
                ], md=6),
            ])
        ])
    ], className="driver-card")

# =====================
# RUN
# =====================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
