# F1 2025 Season Visualization - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-02

### Added
- Initial release of F1 2025 Season Visualization project
- Streamlit dashboard with 5 tabs (Season Overview, Drivers, Teams, Races, AI Predictor)
- FastF1 integration for official telemetry data
- Machine Learning model (Random Forest) for position prediction
- 6 Jupyter notebooks for data analysis:
  - `01_data_exploration.ipynb` - Basic EDA
  - `02_race_analysis.ipynb` - Race results visualization
  - `03_qualifying_analysis.ipynb` - Qualifying performance
  - `04_sprint_analysis.ipynb` - Sprint race analysis
  - `05_fastf1_telemetry.ipynb` - Advanced telemetry
  - `f1_experiments.ipynb` - Experimental analysis

### Data Sources
- Local CSV files for 2025 season data
- FastF1 API for historical telemetry

### Features
- Driver championship standings and progression
- Constructor performance analysis
- Head-to-head driver comparisons
- Qualifying vs Race position analysis
- Sprint race performance metrics
- DNF and reliability statistics
- Interactive Plotly visualizations

## [Unreleased]

### Planned
- Real-time race data integration
- More advanced ML models
- Team strategy analysis
- Weather impact analysis
