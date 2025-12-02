"""
Tests for the analysis module.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import calculate_driver_stats, calculate_team_stats, calculate_race_stats


@pytest.fixture
def sample_race_data():
    """Create sample race data for testing."""
    return pd.DataFrame({
        'Driver': ['Driver A', 'Driver A', 'Driver B', 'Driver B'],
        'Team': ['Team X', 'Team X', 'Team Y', 'Team Y'],
        'Track': ['Race 1', 'Race 2', 'Race 1', 'Race 2'],
        'Position': [1, 2, 2, 1],
        'Points': [25, 18, 18, 25],
        'Starting Grid': [1, 3, 2, 1],
        'Finished': [True, True, True, True],
        'Set Fastest Lap': ['Yes', 'No', 'No', 'Yes']
    })


class TestDriverStats:
    """Tests for calculate_driver_stats function."""
    
    def test_calculates_total_points(self, sample_race_data):
        """Test total points calculation."""
        stats = calculate_driver_stats(sample_race_data)
        
        assert 'Total_Points' in stats.columns
        assert stats.loc['Driver A', 'Total_Points'] == 43
        assert stats.loc['Driver B', 'Total_Points'] == 43
    
    def test_calculates_wins(self, sample_race_data):
        """Test wins calculation."""
        stats = calculate_driver_stats(sample_race_data)
        
        assert 'Wins' in stats.columns
        assert stats.loc['Driver A', 'Wins'] == 1
        assert stats.loc['Driver B', 'Wins'] == 1
    
    def test_calculates_podiums(self, sample_race_data):
        """Test podium count calculation."""
        stats = calculate_driver_stats(sample_race_data)
        
        assert 'Podium' in stats.columns
        # Both positions are podiums (1 and 2)
        assert stats.loc['Driver A', 'Podium'] == 2
        assert stats.loc['Driver B', 'Podium'] == 2


class TestTeamStats:
    """Tests for calculate_team_stats function."""
    
    def test_calculates_team_points(self, sample_race_data):
        """Test team total points calculation."""
        stats = calculate_team_stats(sample_race_data)
        
        assert 'Total_Points' in stats.columns
        assert stats.loc['Team X', 'Total_Points'] == 43
        assert stats.loc['Team Y', 'Total_Points'] == 43


class TestRaceStats:
    """Tests for calculate_race_stats function."""
    
    def test_calculates_entries_per_race(self, sample_race_data):
        """Test entries count per race."""
        stats = calculate_race_stats(sample_race_data)
        
        assert 'Entries' in stats.columns
        assert stats.loc['Race 1', 'Entries'] == 2
        assert stats.loc['Race 2', 'Entries'] == 2
    
    def test_calculates_finish_rate(self, sample_race_data):
        """Test finish rate calculation."""
        stats = calculate_race_stats(sample_race_data)
        
        assert 'Finish_Rate' in stats.columns
        # All finished, so 100%
        assert stats.loc['Race 1', 'Finish_Rate'] == 100.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
