"""
Configuration settings for F1 Visualization project.

This module provides centralized configuration for paths, constants,
and settings used throughout the application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
CACHE_DIR = PROJECT_ROOT / 'cache'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


# ============================================================
# DATA FILE PATHS
# ============================================================

@dataclass
class DataFiles:
    """Data file path configuration."""
    race_results: Path = DATA_DIR / 'Formula1_2025Season_RaceResults.csv'
    qualifying_results: Path = DATA_DIR / 'Formula1_2025Season_QualifyingResults.csv'
    sprint_results: Path = DATA_DIR / 'Formula1_2025Season_SprintResults.csv'
    sprint_qualifying: Path = DATA_DIR / 'Formula1_2025Season_SprintQualifyingResults.csv'


DATA_FILES = DataFiles()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    model_path: Path = MODELS_DIR / 'f1_model.pkl'
    driver_encoder_path: Path = MODELS_DIR / 'Driver_encoder.pkl'
    team_encoder_path: Path = MODELS_DIR / 'Team_encoder.pkl'
    track_encoder_path: Path = MODELS_DIR / 'Track_encoder.pkl'
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100


MODEL_CONFIG = ModelConfig()


# ============================================================
# FASTF1 CONFIGURATION
# ============================================================

@dataclass
class FastF1Config:
    """FastF1 API configuration."""
    cache_dir: Path = CACHE_DIR
    default_year: int = 2025
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: int = 30


FASTF1_CONFIG = FastF1Config()


# ============================================================
# F1 2025 SEASON DATA
# ============================================================

# Official 2025 F1 Calendar
F1_2025_CALENDAR: List[Dict] = [
    {'round': 1, 'name': 'Australian Grand Prix', 'location': 'Melbourne', 'country': 'Australia'},
    {'round': 2, 'name': 'Chinese Grand Prix', 'location': 'Shanghai', 'country': 'China'},
    {'round': 3, 'name': 'Japanese Grand Prix', 'location': 'Suzuka', 'country': 'Japan'},
    {'round': 4, 'name': 'Bahrain Grand Prix', 'location': 'Sakhir', 'country': 'Bahrain'},
    {'round': 5, 'name': 'Saudi Arabian Grand Prix', 'location': 'Jeddah', 'country': 'Saudi Arabia'},
    {'round': 6, 'name': 'Miami Grand Prix', 'location': 'Miami', 'country': 'USA'},
    {'round': 7, 'name': 'Emilia Romagna Grand Prix', 'location': 'Imola', 'country': 'Italy'},
    {'round': 8, 'name': 'Monaco Grand Prix', 'location': 'Monte Carlo', 'country': 'Monaco'},
    {'round': 9, 'name': 'Spanish Grand Prix', 'location': 'Barcelona', 'country': 'Spain'},
    {'round': 10, 'name': 'Canadian Grand Prix', 'location': 'Montreal', 'country': 'Canada'},
    {'round': 11, 'name': 'Austrian Grand Prix', 'location': 'Spielberg', 'country': 'Austria'},
    {'round': 12, 'name': 'British Grand Prix', 'location': 'Silverstone', 'country': 'Great Britain'},
    {'round': 13, 'name': 'Belgian Grand Prix', 'location': 'Spa', 'country': 'Belgium'},
    {'round': 14, 'name': 'Hungarian Grand Prix', 'location': 'Budapest', 'country': 'Hungary'},
    {'round': 15, 'name': 'Dutch Grand Prix', 'location': 'Zandvoort', 'country': 'Netherlands'},
    {'round': 16, 'name': 'Italian Grand Prix', 'location': 'Monza', 'country': 'Italy'},
    {'round': 17, 'name': 'Azerbaijan Grand Prix', 'location': 'Baku', 'country': 'Azerbaijan'},
    {'round': 18, 'name': 'Singapore Grand Prix', 'location': 'Marina Bay', 'country': 'Singapore'},
    {'round': 19, 'name': 'United States Grand Prix', 'location': 'Austin', 'country': 'USA'},
    {'round': 20, 'name': 'Mexico City Grand Prix', 'location': 'Mexico City', 'country': 'Mexico'},
    {'round': 21, 'name': 'São Paulo Grand Prix', 'location': 'Interlagos', 'country': 'Brazil'},
    {'round': 22, 'name': 'Las Vegas Grand Prix', 'location': 'Las Vegas', 'country': 'USA'},
    {'round': 23, 'name': 'Qatar Grand Prix', 'location': 'Lusail', 'country': 'Qatar'},
    {'round': 24, 'name': 'Abu Dhabi Grand Prix', 'location': 'Yas Marina', 'country': 'UAE'},
]

# F1 Points System
POINTS_RACE: Dict[int, int] = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

POINTS_SPRINT: Dict[int, int] = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
    6: 3, 7: 2, 8: 1
}

POINTS_FASTEST_LAP: int = 1

# 2025 F1 Teams with Official Colors (Updated to 2025 Specifications)
F1_2025_TEAMS: List[Dict] = [
    {'name': 'McLaren Mercedes', 'short': 'McLaren', 'color': '#FF8700'},
    {'name': 'Red Bull Racing Honda RBPT', 'short': 'Red Bull', 'color': '#1E41FF'},
    {'name': 'Ferrari', 'short': 'Ferrari', 'color': '#DC0000'},
    {'name': 'Mercedes', 'short': 'Mercedes', 'color': '#00D2BE'},
    {'name': 'Aston Martin Aramco Mercedes', 'short': 'Aston Martin', 'color': '#006F62'},
    {'name': 'Alpine Renault', 'short': 'Alpine', 'color': '#0090FF'},
    {'name': 'Williams Mercedes', 'short': 'Williams', 'color': '#005AFF'},
    {'name': 'Racing Bulls Honda RBPT', 'short': 'Racing Bulls', 'color': '#F6E500'},  # Yellow 2025
    {'name': 'Kick Sauber Ferrari', 'short': 'Kick Sauber', 'color': '#00E701'},  # Neon Green Stake 2025
    {'name': 'Haas Ferrari', 'short': 'Haas', 'color': '#E6002B'},
]

# Team Colors Dictionary - All possible team name variations (2025 Official)
TEAM_COLORS: Dict[str, str] = {
    # Main team names - 2025 Official Colors
    'Mercedes': '#00D2BE',
    'Red Bull': '#1E41FF',
    'Ferrari': '#DC0000',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'Racing Bulls': '#F6E500',  # Yellow 2025 (formerly VCARB)
    'Kick Sauber': '#00E701',  # Neon Green Stake 2025
    'Haas': '#E6002B',
    
    # Alternative/Full team names
    'Mercedes-AMG': '#00D2BE',
    'Mercedes-AMG Petronas F1 Team': '#00D2BE',
    'Red Bull Racing': '#1E41FF',
    'Red Bull Racing Honda RBPT': '#1E41FF',
    'Scuderia Ferrari': '#DC0000',
    'Scuderia Ferrari HP': '#DC0000',
    'McLaren Formula 1 Team': '#FF8700',
    'McLaren Mercedes': '#FF8700',
    'Aston Martin Aramco': '#006F62',
    'Aston Martin Aramco Mercedes': '#006F62',
    'BWT Alpine F1 Team': '#0090FF',
    'Alpine Renault': '#0090FF',
    'Williams Racing': '#005AFF',
    'Williams Mercedes': '#005AFF',
    'Visa Cash App RB': '#F6E500',  # 2025 Yellow
    'VCARB': '#F6E500',  # Old name, 2025 Yellow
    'RB': '#F6E500',  # 2025 Yellow
    'Racing Bulls Honda RBPT': '#F6E500',  # 2025 Yellow
    'Stake F1 Team Kick Sauber': '#00E701',  # 2025 Neon Green
    'Sauber': '#00E701',  # 2025 Neon Green
    'Kick Sauber Ferrari': '#00E701',  # 2025 Neon Green
    'MoneyGram Haas F1 Team': '#E6002B',
    'Haas Ferrari': '#E6002B',
}


def get_team_color(team_name: str) -> str:
    """
    Get the official color for a team with error handling.
    
    Args:
        team_name: Team name (any variation)
        
    Returns:
        Hex color code string (2025 official color or default gray)
        
    Raises:
        None - Always returns a valid color
    """
    try:
        # Validate input
        if not team_name or not isinstance(team_name, str):
            logger.warning(f"Invalid team name provided: {team_name}")
            return '#808080'
        
        # Direct match (case-insensitive)
        team_name_stripped = team_name.strip()
        for key, color in TEAM_COLORS.items():
            if key.lower() == team_name_stripped.lower():
                return color
        
        # Partial match
        for key, color in TEAM_COLORS.items():
            if key.lower() in team_name_stripped.lower() or team_name_stripped.lower() in key.lower():
                logger.debug(f"Partial match: '{team_name}' -> '{key}' ({color})")
                return color
        
        # No match found
        logger.warning(f"No color found for team: '{team_name}'. Using default gray.")
        return '#808080'
        
    except Exception as e:
        logger.error(f"Error getting team color for '{team_name}': {e}")
        return '#808080'


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('f1_visualization')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


# Default logger
logger = setup_logging()


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

@dataclass
class StreamlitConfig:
    """Streamlit dashboard configuration."""
    page_title: str = "F1 2025 Season Dashboard"
    page_icon: str = "🏎️"
    layout: str = "wide"
    theme_primary_color: str = "#FF1E00"
    theme_background_color: str = "#0E1117"
    cache_ttl: int = 3600  # 1 hour


STREAMLIT_CONFIG = StreamlitConfig()


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    plotly_template: str = "plotly_dark"
    matplotlib_style: str = "seaborn-v0_8-darkgrid"
    default_colorscale: str = "RdYlGn_r"
    chart_height: int = 500
    chart_width: int = 800


VIZ_CONFIG = VisualizationConfig()


# ============================================================
# EXPORT CONFIGURATION
# ============================================================

def get_config() -> Dict:
    """
    Get all configuration as a dictionary.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        'project_root': str(PROJECT_ROOT),
        'data_dir': str(DATA_DIR),
        'models_dir': str(MODELS_DIR),
        'cache_dir': str(CACHE_DIR),
        'data_files': {
            'race': str(DATA_FILES.race_results),
            'qualifying': str(DATA_FILES.qualifying_results),
            'sprint': str(DATA_FILES.sprint_results),
            'sprint_qualifying': str(DATA_FILES.sprint_qualifying),
        },
        'model_config': {
            'test_size': MODEL_CONFIG.test_size,
            'random_state': MODEL_CONFIG.random_state,
            'n_estimators': MODEL_CONFIG.n_estimators,
        },
        'fastf1_config': {
            'default_year': FASTF1_CONFIG.default_year,
            'retry_attempts': FASTF1_CONFIG.retry_attempts,
        },
        'points_race': POINTS_RACE,
        'points_sprint': POINTS_SPRINT,
    }


# ============================================================
# DRIVER PROFILES - 2025 Season
# ============================================================

DRIVER_PROFILES: Dict[str, Dict] = {
    # McLaren
    "Lando Norris": {
        "number": 4,
        "abbreviation": "NOR",
        "team": "McLaren",
        "country": "United Kingdom",
        "country_flag": "gb",
        "date_of_birth": "1999-11-13",
        "place_of_birth": "Bristol, England",
        "debut_year": 2019,
        "debut_race": "2019 Australian GP",
        "championships": 0,
        "career_wins": 4,
        "career_podiums": 28,
        "career_poles": 4,
        "career_fastest_laps": 8,
        "career_points": 872,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/norris.png",
        "bio": "British racing driver who made his F1 debut in 2019 with McLaren. Known for his raw speed and consistency."
    },
    "Oscar Piastri": {
        "number": 81,
        "abbreviation": "PIA",
        "team": "McLaren",
        "country": "Australia",
        "country_flag": "au",
        "date_of_birth": "2001-04-06",
        "place_of_birth": "Melbourne, Australia",
        "debut_year": 2023,
        "debut_race": "2023 Bahrain GP",
        "championships": 0,
        "career_wins": 3,
        "career_podiums": 12,
        "career_poles": 2,
        "career_fastest_laps": 3,
        "career_points": 350,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/piastri.png",
        "bio": "Australian driver and 2021 F2 Champion. Joined McLaren in 2023 and quickly established himself as a front-runner."
    },
    # Red Bull
    "Max Verstappen": {
        "number": 1,
        "abbreviation": "VER",
        "team": "Red Bull",
        "country": "Netherlands",
        "country_flag": "nl",
        "date_of_birth": "1997-09-30",
        "place_of_birth": "Hasselt, Belgium",
        "debut_year": 2015,
        "debut_race": "2015 Australian GP",
        "championships": 4,
        "career_wins": 63,
        "career_podiums": 112,
        "career_poles": 40,
        "career_fastest_laps": 33,
        "career_points": 2998,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/verstappen.png",
        "bio": "Four-time World Champion (2021-2024). Youngest F1 race winner at 18. Dominant force in modern F1."
    },
    "Liam Lawson": {
        "number": 30,
        "abbreviation": "LAW",
        "team": "Red Bull",
        "country": "New Zealand",
        "country_flag": "nz",
        "date_of_birth": "2002-02-11",
        "place_of_birth": "Hastings, New Zealand",
        "debut_year": 2023,
        "debut_race": "2023 Dutch GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 1,
        "career_poles": 0,
        "career_fastest_laps": 1,
        "career_points": 46,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/lawson.png",
        "bio": "New Zealand driver promoted to Red Bull for 2025 after impressive substitute appearances in 2023-2024."
    },
    # Ferrari
    "Charles Leclerc": {
        "number": 16,
        "abbreviation": "LEC",
        "team": "Ferrari",
        "country": "Monaco",
        "country_flag": "mc",
        "date_of_birth": "1997-10-16",
        "place_of_birth": "Monte Carlo, Monaco",
        "debut_year": 2018,
        "debut_race": "2018 Australian GP",
        "championships": 0,
        "career_wins": 8,
        "career_podiums": 40,
        "career_poles": 26,
        "career_fastest_laps": 10,
        "career_points": 1250,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/leclerc.png",
        "bio": "Monegasque driver and 2017 F2 Champion. Ferrari's lead driver known for his qualifying speed."
    },
    "Lewis Hamilton": {
        "number": 44,
        "abbreviation": "HAM",
        "team": "Ferrari",
        "country": "United Kingdom",
        "country_flag": "gb",
        "date_of_birth": "1985-01-07",
        "place_of_birth": "Stevenage, England",
        "debut_year": 2007,
        "debut_race": "2007 Australian GP",
        "championships": 7,
        "career_wins": 105,
        "career_podiums": 202,
        "career_poles": 104,
        "career_fastest_laps": 67,
        "career_points": 4800,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/hamilton.png",
        "bio": "Seven-time World Champion, tied with Michael Schumacher for most titles. Joined Ferrari for 2025 after 12 years at Mercedes."
    },
    # Mercedes
    "George Russell": {
        "number": 63,
        "abbreviation": "RUS",
        "team": "Mercedes",
        "country": "United Kingdom",
        "country_flag": "gb",
        "date_of_birth": "1998-02-15",
        "place_of_birth": "King's Lynn, England",
        "debut_year": 2019,
        "debut_race": "2019 Australian GP",
        "championships": 0,
        "career_wins": 3,
        "career_podiums": 16,
        "career_poles": 4,
        "career_fastest_laps": 7,
        "career_points": 545,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/russell.png",
        "bio": "British driver and 2018 F2 Champion. Mercedes team leader for 2025 season."
    },
    "Andrea Kimi Antonelli": {
        "number": 12,
        "abbreviation": "ANT",
        "team": "Mercedes",
        "country": "Italy",
        "country_flag": "it",
        "date_of_birth": "2006-08-25",
        "place_of_birth": "Bologna, Italy",
        "debut_year": 2025,
        "debut_race": "2025 Australian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 0,
        "career_poles": 0,
        "career_fastest_laps": 0,
        "career_points": 0,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/A/ANDANT01_Andrea_Kimi_Antonelli/andant01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/antonelli.png",
        "bio": "Italian prodigy and 2024 F2 Champion. Youngest driver on the 2025 grid at 18 years old."
    },
    # Aston Martin
    "Fernando Alonso": {
        "number": 14,
        "abbreviation": "ALO",
        "team": "Aston Martin",
        "country": "Spain",
        "country_flag": "es",
        "date_of_birth": "1981-07-29",
        "place_of_birth": "Oviedo, Spain",
        "debut_year": 2001,
        "debut_race": "2001 Australian GP",
        "championships": 2,
        "career_wins": 32,
        "career_podiums": 106,
        "career_poles": 22,
        "career_fastest_laps": 24,
        "career_points": 2298,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/alonso.png",
        "bio": "Two-time World Champion (2005-2006). Most experienced driver on the grid with 400+ race starts."
    },
    "Lance Stroll": {
        "number": 18,
        "abbreviation": "STR",
        "team": "Aston Martin",
        "country": "Canada",
        "country_flag": "ca",
        "date_of_birth": "1998-10-29",
        "place_of_birth": "Montreal, Canada",
        "debut_year": 2017,
        "debut_race": "2017 Australian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 3,
        "career_poles": 1,
        "career_fastest_laps": 0,
        "career_points": 292,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/stroll.png",
        "bio": "Canadian driver who scored a podium in his rookie season. Son of Aston Martin team owner Lawrence Stroll."
    },
    # Alpine
    "Pierre Gasly": {
        "number": 10,
        "abbreviation": "GAS",
        "team": "Alpine",
        "country": "France",
        "country_flag": "fr",
        "date_of_birth": "1996-02-07",
        "place_of_birth": "Rouen, France",
        "debut_year": 2017,
        "debut_race": "2017 Malaysian GP",
        "championships": 0,
        "career_wins": 1,
        "career_podiums": 4,
        "career_poles": 0,
        "career_fastest_laps": 3,
        "career_points": 394,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/gasly.png",
        "bio": "French driver with a memorable victory at Monza 2020. Alpine's experienced hand."
    },
    "Jack Doohan": {
        "number": 7,
        "abbreviation": "DOO",
        "team": "Alpine",
        "country": "Australia",
        "country_flag": "au",
        "date_of_birth": "2003-01-20",
        "place_of_birth": "Gold Coast, Australia",
        "debut_year": 2025,
        "debut_race": "2025 Australian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 0,
        "career_poles": 0,
        "career_fastest_laps": 0,
        "career_points": 0,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/J/JACDOO01_Jack_Doohan/jacdoo01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/doohan.png",
        "bio": "Son of motorcycle legend Mick Doohan. Alpine junior promoted for 2025 after serving as reserve driver."
    },
    # Williams
    "Carlos Sainz": {
        "number": 55,
        "abbreviation": "SAI",
        "team": "Williams",
        "country": "Spain",
        "country_flag": "es",
        "date_of_birth": "1994-09-01",
        "place_of_birth": "Madrid, Spain",
        "debut_year": 2015,
        "debut_race": "2015 Australian GP",
        "championships": 0,
        "career_wins": 4,
        "career_podiums": 25,
        "career_poles": 6,
        "career_fastest_laps": 5,
        "career_points": 1108,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/sainz.png",
        "bio": "Spanish driver who moved to Williams for 2025 after successful stint at Ferrari."
    },
    "Alexander Albon": {
        "number": 23,
        "abbreviation": "ALB",
        "team": "Williams",
        "country": "Thailand",
        "country_flag": "th",
        "date_of_birth": "1996-03-23",
        "place_of_birth": "London, England",
        "debut_year": 2019,
        "debut_race": "2019 Australian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 2,
        "career_poles": 0,
        "career_fastest_laps": 0,
        "career_points": 232,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/albon.png",
        "bio": "Thai-British driver who rebuilt his career at Williams after being dropped by Red Bull."
    },
    # Racing Bulls
    "Yuki Tsunoda": {
        "number": 22,
        "abbreviation": "TSU",
        "team": "Racing Bulls",
        "country": "Japan",
        "country_flag": "jp",
        "date_of_birth": "2000-05-11",
        "place_of_birth": "Sagamihara, Japan",
        "debut_year": 2021,
        "debut_race": "2021 Bahrain GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 1,
        "career_poles": 0,
        "career_fastest_laps": 1,
        "career_points": 91,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/tsunoda.png",
        "bio": "Japanese driver in his fifth F1 season. Known for his aggressive driving style and radio outbursts."
    },
    "Isack Hadjar": {
        "number": 6,
        "abbreviation": "HAD",
        "team": "Racing Bulls",
        "country": "France",
        "country_flag": "fr",
        "date_of_birth": "2004-09-28",
        "place_of_birth": "Paris, France",
        "debut_year": 2025,
        "debut_race": "2025 Australian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 0,
        "career_poles": 0,
        "career_fastest_laps": 0,
        "career_points": 0,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/I/ISAHAD01_Isack_Hadjar/isahad01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/hadjar.png",
        "bio": "French-Algerian driver and 2024 F2 runner-up. Red Bull junior promoted for 2025."
    },
    # Kick Sauber
    "Nico Hulkenberg": {
        "number": 27,
        "abbreviation": "HUL",
        "team": "Kick Sauber",
        "country": "Germany",
        "country_flag": "de",
        "date_of_birth": "1987-08-19",
        "place_of_birth": "Emmerich am Rhein, Germany",
        "debut_year": 2010,
        "debut_race": "2010 Bahrain GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 0,
        "career_poles": 1,
        "career_fastest_laps": 2,
        "career_points": 546,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/hulkenberg.png",
        "bio": "German veteran and 2009 GP2 Champion. Holds record for most races without a podium."
    },
    "Gabriel Bortoleto": {
        "number": 5,
        "abbreviation": "BOR",
        "team": "Kick Sauber",
        "country": "Brazil",
        "country_flag": "br",
        "date_of_birth": "2004-10-14",
        "place_of_birth": "Sao Paulo, Brazil",
        "debut_year": 2025,
        "debut_race": "2025 Australian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 0,
        "career_poles": 0,
        "career_fastest_laps": 0,
        "career_points": 0,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/G/GABBOR01_Gabriel_Bortoleto/gabbor01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/bortoleto.png",
        "bio": "Brazilian driver and 2024 F2 Champion. First Brazilian F1 driver since Felipe Massa."
    },
    # Haas
    "Esteban Ocon": {
        "number": 31,
        "abbreviation": "OCO",
        "team": "Haas",
        "country": "France",
        "country_flag": "fr",
        "date_of_birth": "1996-09-17",
        "place_of_birth": "Evreux, France",
        "debut_year": 2016,
        "debut_race": "2016 Belgian GP",
        "championships": 0,
        "career_wins": 1,
        "career_podiums": 4,
        "career_poles": 0,
        "career_fastest_laps": 1,
        "career_points": 427,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/ocon.png",
        "bio": "French driver who won the 2021 Hungarian GP. Joined Haas for 2025 after leaving Alpine."
    },
    "Oliver Bearman": {
        "number": 87,
        "abbreviation": "BEA",
        "team": "Haas",
        "country": "United Kingdom",
        "country_flag": "gb",
        "date_of_birth": "2005-05-08",
        "place_of_birth": "Chelmsford, England",
        "debut_year": 2024,
        "debut_race": "2024 Saudi Arabian GP",
        "championships": 0,
        "career_wins": 0,
        "career_podiums": 0,
        "career_poles": 0,
        "career_fastest_laps": 0,
        "career_points": 7,
        "image_url": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png",
        "helmet_url": "https://media.formula1.com/d_default_fallback_image.png/content/dam/fom-website/manual/Helmets2024/bearman.png",
        "bio": "British teenager who impressed on debut at Saudi Arabia 2024, substituting for Sainz at Ferrari."
    },
}


# Races completed in 2025 season (up to December 2, 2025)
F1_2025_COMPLETED_RACES: List[str] = [
    'Australia', 'China', 'Japan', 'Bahrain', 'Saudi Arabia', 'Miami',
    'Emilia Romagna', 'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain',
    'Belgium', 'Hungary', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore',
    'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar'
]

# Race name mapping for FastF1
F1_2025_RACE_NAMES: Dict[str, str] = {
    'Australia': 'Australian Grand Prix',
    'China': 'Chinese Grand Prix',
    'Japan': 'Japanese Grand Prix',
    'Bahrain': 'Bahrain Grand Prix',
    'Saudi Arabia': 'Saudi Arabian Grand Prix',
    'Miami': 'Miami Grand Prix',
    'Emilia Romagna': 'Emilia Romagna Grand Prix',
    'Monaco': 'Monaco Grand Prix',
    'Spain': 'Spanish Grand Prix',
    'Canada': 'Canadian Grand Prix',
    'Austria': 'Austrian Grand Prix',
    'Great Britain': 'British Grand Prix',
    'Belgium': 'Belgian Grand Prix',
    'Hungary': 'Hungarian Grand Prix',
    'Netherlands': 'Dutch Grand Prix',
    'Italy': 'Italian Grand Prix',
    'Azerbaijan': 'Azerbaijan Grand Prix',
    'Singapore': 'Singapore Grand Prix',
    'United States': 'United States Grand Prix',
    'Mexico': 'Mexico City Grand Prix',
    'Brazil': 'São Paulo Grand Prix',
    'Las Vegas': 'Las Vegas Grand Prix',
    'Qatar': 'Qatar Grand Prix',
    'Abu Dhabi': 'Abu Dhabi Grand Prix',
}


if __name__ == '__main__':
    # Print configuration summary
    print("F1 Visualization Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"2025 Calendar: {len(F1_2025_CALENDAR)} races")
    print(f"Teams: {len(F1_2025_TEAMS)}")
    print(f"Driver Profiles: {len(DRIVER_PROFILES)} drivers")
