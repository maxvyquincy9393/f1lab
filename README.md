# F1 2025 Season Visualization

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FastF1](https://img.shields.io/badge/FastF1-3.7+-red.svg)

This project transforms raw F1 race results into an interactive, professional dashboard using **Streamlit** and **Plotly**. It also includes advanced telemetry analysis using the **FastF1** library.

## Features
- **Season Overview**: Instant view of Championship leaders and top performers.
- **Driver Analysis**: Deep dive into individual driver stats (Wins, Podiums, Consistency).
- **Team Performance**: Analyze constructor efficiency and reliability.
- **Race Statistics**: Track-by-track breakdown of finish rates and points.
- **Qualifying Analysis**: Q1/Q2/Q3 performance breakdown.
- **Sprint Race Analysis**: Sprint weekend performance metrics.
- **AI Race Predictor**: Machine learning model to predict race positions.
- **FastF1 Telemetry**: Advanced telemetry data (speed, throttle, brake, sector times).
- **Telemetry Animation**: Animated racing lines, driver battles, speed traces.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run src/app.py
   ```

3. **Run Jupyter Notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

4. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

## Project Structure
```
f1-visualization/
├── data/
│   ├── Formula1_2025Season_RaceResults.csv
│   ├── Formula1_2025Season_QualifyingResults.csv
│   ├── Formula1_2025Season_SprintResults.csv
│   └── Formula1_2025Season_SprintQualifyingResults.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── loader.py               # Load raw CSV data
│   ├── fastf1_loader.py        # FastF1 API wrapper functions
│   ├── analysis.py             # Statistics calculations
│   ├── features.py             # Feature engineering for ML
│   ├── model.py                # ML model training (Random Forest)
│   ├── evaluate.py             # Evaluation metrics and plots
│   └── app.py                  # Streamlit Dashboard
│
├── notebooks/
│   ├── data_exploration.ipynb     # Basic EDA and data loading
│   ├── race_analysis.ipynb        # Race results visualization
│   ├── qualifying_analysis.ipynb  # Qualifying performance
│   ├── sprint_analysis.ipynb      # Sprint race analysis
│   ├── fastf1_telemetry.ipynb     # FastF1 advanced telemetry
│   ├── fastf1_animation.ipynb     # Telemetry animations
│   └── prediction.ipynb           # Race Position Prediction (ML)
│
├── models/
│   ├── f1_model.pkl            # Trained Random Forest model
│   ├── Driver_encoder.pkl      # Label encoder for drivers
│   ├── Team_encoder.pkl        # Label encoder for teams
│   └── Track_encoder.pkl       # Label encoder for tracks
│
├── tests/
│   ├── test_loader.py          # Tests for data loading
│   ├── test_analysis.py        # Tests for analysis functions
│   ├── test_features.py        # Tests for feature engineering
│   └── test_model.py           # Tests for ML model
│
├── cache/                      # FastF1 cache directory
├── README.md
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Data Sources

### Local CSV Data
- `Formula1_2025Season_RaceResults.csv` - Race positions, points, status
- `Formula1_2025Season_QualifyingResults.csv` - Q1, Q2, Q3 times
- `Formula1_2025Season_SprintResults.csv` - Sprint race results
- `Formula1_2025Season_SprintQualifyingResults.csv` - Sprint shootout times

### FastF1 API
The project uses FastF1 library for accessing official F1 telemetry data:
- Lap times and sector times
- Telemetry (speed, throttle, brake, gear, DRS)
- Weather data (air temp, track temp, rainfall)
- Tyre compound information
- Car position coordinates (X, Y)

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `data_exploration.ipynb` | Load all datasets, check missing values, basic statistics |
| `race_analysis.ipynb` | Championship standings, podiums, DNF analysis, heatmaps |
| `qualifying_analysis.ipynb` | Pole positions, Q3 rate, gap to pole, consistency |
| `sprint_analysis.ipynb` | Sprint winners, points contribution, grid vs finish |
| `fastf1_telemetry.ipynb` | Telemetry traces, driver comparison, tyre strategy |
| `fastf1_animation.ipynb` | Racing line animation, battle visualization, speed traces |
| `prediction.ipynb` | Race position prediction using Machine Learning |

## Usage Examples

### Load Data with FastF1
```python
from src.fastf1_loader import get_session, get_lap_data, get_telemetry

# Load 2025 Australian GP Race Session
session = get_session(2025, 'Australia', 'R')

# Get lap data
laps = get_lap_data(session)

# Get telemetry for fastest lap
fastest = laps.pick_fastest()
telemetry = get_telemetry(fastest)
```

### Run ML Prediction
```python
from src.model import load_trained_model
import pandas as pd

# Load model
model = load_trained_model()

# Prepare input features
input_data = pd.DataFrame({
    'Driver_encoded': [0],
    'Team_encoded': [1],
    'Track_encoded': [2],
    'Starting Grid': [3]
})

# Predict
prediction = model.predict(input_data)
print(f"Predicted Position: {prediction[0]:.1f}")
```

### Calculate Championship Standings
```python
from src.loader import load_data
from src.analysis import calculate_combined_standings

# Load race + sprint data
race_df = load_data('data/Formula1_2025Season_RaceResults.csv')
sprint_df = load_data('data/Formula1_2025Season_SprintResults.csv')

# Get combined standings
standings = calculate_combined_standings(race_df, sprint_df)
print(standings.head(10))
```

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.2.0
streamlit>=1.22.0
fastf1>=3.0.0
pytest>=7.0.0
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Author

**maxvy** - [GitHub](https://github.com/maxvyquincy9393)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*F1 data is provided by FastF1 library which sources data from official F1 timing.*
*This project is for educational purposes only.*
