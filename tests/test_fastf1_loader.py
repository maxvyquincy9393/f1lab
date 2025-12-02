"""
Tests for the fastf1_loader module.

These tests use mocking to avoid actual API calls.
Tests are skipped if fastf1 is not installed.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if fastf1 is not installed
fastf1_available = pytest.importorskip("fastf1", reason="fastf1 not installed")

from src.fastf1_loader import (
    get_session, get_race_results, get_lap_data,
    get_telemetry, get_weather_data, get_schedule,
    compare_drivers_lap_times, get_track_layout,
    get_driver_telemetry_data, generate_animation_frames,
    get_delta_time_data, get_battle_animation_data
)


class TestGetSchedule:
    """Tests for get_schedule function."""
    
    @patch('src.fastf1_loader.fastf1.get_event_schedule')
    def test_returns_dataframe(self, mock_schedule):
        """Test that schedule returns a DataFrame."""
        mock_schedule.return_value = pd.DataFrame({
            'RoundNumber': [1, 2],
            'EventName': ['Australian GP', 'Bahrain GP']
        })
        
        result = get_schedule(2025)
        
        assert isinstance(result, pd.DataFrame)
    
    @patch('src.fastf1_loader.fastf1.get_event_schedule')
    def test_handles_error(self, mock_schedule):
        """Test that errors return empty DataFrame."""
        mock_schedule.side_effect = Exception('API Error')
        
        result = get_schedule(2025)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestGetSession:
    """Tests for get_session function."""
    
    @patch('src.fastf1_loader.fastf1.get_session')
    def test_loads_session(self, mock_get_session):
        """Test that session is loaded correctly."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        result = get_session(2025, 'Australia', 'R')
        
        mock_get_session.assert_called_once_with(2025, 'Australia', 'R')
        mock_session.load.assert_called_once()
    
    @patch('src.fastf1_loader.fastf1.get_session')
    def test_handles_error(self, mock_get_session):
        """Test that errors return None."""
        mock_get_session.side_effect = Exception('Session not found')
        
        result = get_session(2025, 'NonExistent', 'R')
        
        assert result is None


class TestGetRaceResults:
    """Tests for get_race_results function."""
    
    def test_returns_empty_df_for_none_session(self):
        """Test that None session returns empty DataFrame."""
        result = get_race_results(None)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_extracts_correct_columns(self):
        """Test that correct columns are extracted."""
        mock_session = MagicMock()
        mock_session.results = pd.DataFrame({
            'DriverNumber': [1, 2],
            'Abbreviation': ['NOR', 'VER'],
            'FullName': ['Lando Norris', 'Max Verstappen'],
            'TeamName': ['McLaren', 'Red Bull'],
            'Position': [1, 2],
            'Points': [25, 18],
            'Status': ['Finished', 'Finished'],
            'Time': ['1:30:00', '+5.0s'],
            'ExtraCol': ['x', 'y']  # Should be filtered out
        })
        
        result = get_race_results(mock_session)
        
        assert 'Abbreviation' in result.columns
        assert 'ExtraCol' not in result.columns


class TestGetLapData:
    """Tests for get_lap_data function."""
    
    def test_returns_empty_df_for_none_session(self):
        """Test that None session returns empty DataFrame."""
        result = get_lap_data(None)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_returns_all_laps(self):
        """Test that all laps are returned when no driver specified."""
        mock_session = MagicMock()
        mock_session.laps = pd.DataFrame({
            'Driver': ['NOR', 'NOR', 'VER', 'VER'],
            'LapNumber': [1, 2, 1, 2],
            'LapTime': [90, 91, 90, 92]
        })
        mock_session.laps.copy = MagicMock(return_value=mock_session.laps)
        
        result = get_lap_data(mock_session)
        
        assert len(result) == 4
    
    def test_filters_by_driver(self):
        """Test that laps are filtered by driver."""
        mock_laps = pd.DataFrame({
            'Driver': ['NOR', 'NOR', 'VER', 'VER'],
            'LapNumber': [1, 2, 1, 2]
        })
        mock_session = MagicMock()
        mock_session.laps = mock_laps
        mock_session.laps.copy = MagicMock(return_value=mock_laps.copy())
        
        result = get_lap_data(mock_session, driver='NOR')
        
        assert all(result['Driver'] == 'NOR')


class TestGetTelemetry:
    """Tests for get_telemetry function."""
    
    def test_returns_empty_df_for_none(self):
        """Test that None returns empty DataFrame."""
        result = get_telemetry(None)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_extracts_telemetry_from_lap_object(self):
        """Test telemetry extraction from lap object."""
        mock_lap = MagicMock()
        mock_telemetry = pd.DataFrame({
            'Speed': [300, 310, 320],
            'Throttle': [100, 100, 80],
            'Brake': [0, 0, 50]
        })
        mock_lap.get_telemetry.return_value = mock_telemetry
        
        result = get_telemetry(mock_lap)
        
        assert len(result) == 3
        assert 'Speed' in result.columns


class TestGetWeatherData:
    """Tests for get_weather_data function."""
    
    def test_returns_empty_df_for_none_session(self):
        """Test that None session returns empty DataFrame."""
        result = get_weather_data(None)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_returns_weather_data(self):
        """Test that weather data is returned."""
        mock_session = MagicMock()
        mock_weather = pd.DataFrame({
            'AirTemp': [25, 26, 27],
            'TrackTemp': [40, 42, 44]
        })
        mock_session.weather_data = mock_weather
        mock_session.weather_data.copy = MagicMock(return_value=mock_weather.copy())
        
        result = get_weather_data(mock_session)
        
        assert 'AirTemp' in result.columns
        assert 'TrackTemp' in result.columns


class TestCompareDriversLapTimes:
    """Tests for compare_drivers_lap_times function."""
    
    def test_returns_empty_df_for_none_session(self):
        """Test that None session returns empty DataFrame."""
        result = compare_drivers_lap_times(None, ['NOR', 'VER'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestAnimationFunctions:
    """Tests for animation-related functions."""
    
    def test_get_track_layout_returns_dict(self):
        """Test that get_track_layout returns correct structure."""
        result = get_track_layout(None)
        
        assert isinstance(result, dict)
        assert 'x' in result
        assert 'y' in result
        assert 'speed' in result
    
    def test_get_driver_telemetry_data_returns_dict(self):
        """Test that get_driver_telemetry_data returns dict."""
        result = get_driver_telemetry_data(None, 'NOR')
        
        assert isinstance(result, dict)
    
    def test_generate_animation_frames_returns_list(self):
        """Test that generate_animation_frames returns list."""
        result = generate_animation_frames(None, 'NOR')
        
        assert isinstance(result, list)
    
    def test_get_delta_time_data_returns_dict(self):
        """Test that get_delta_time_data returns dict."""
        result = get_delta_time_data(None, 'NOR', 'VER')
        
        assert isinstance(result, dict)
    
    def test_get_battle_animation_data_returns_dict(self):
        """Test that get_battle_animation_data returns dict."""
        result = get_battle_animation_data(None, 'NOR', 'VER')
        
        assert isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
