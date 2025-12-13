# F1 Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastF1](https://img.shields.io/badge/Data-FastF1-red)](https://github.com/theOehrly/Fast-F1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Real-time Formula 1 analytics dashboard with telemetry visualization, race strategy analysis, and machine learning predictions for the 2025 season.

## Features

- **Season Overview** — Driver and constructor standings with trend analysis
- **Race Analysis** — Telemetry comparison, pit strategy, and position tracking
- **Qualifying Insights** — Sector dominance and lap time evolution
- **ML Predictions** — Race outcome forecasting using gradient boosting
- **Live Timing** — Real-time session monitoring (when available)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Runtime | Python 3.10+ |
| Frontend | Streamlit, Plotly Dash |
| Data | Pandas, NumPy, FastF1 |
| Visualization | Plotly, Matplotlib |
| ML | Scikit-learn (Gradient Boosting) |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/f1-analytics.git
cd f1-analytics
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run
streamlit run src/app.py
```

### Docker

```bash
docker build -t f1-analytics .
docker run -p 8501:8501 f1-analytics
```

## Project Structure

```
├── src/                   # Application source code
│   ├── app.py             # Streamlit entry point
│   ├── dash_app.py        # Dash alternative frontend
│   ├── analysis.py        # Statistical computations
│   ├── config.py          # Configuration constants
│   ├── fastf1_*.py        # Telemetry data modules
│   └── *_viz.py           # Visualization components
├── data/                  # Season datasets (CSV)
├── models/                # Trained model artifacts
├── notebooks/             # Analysis notebooks
└── tests/                 # Test suite
```

## Development

```bash
# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/
isort src/ tests/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

F1, Formula 1, and related marks are trademarks of Formula One Licensing B.V.
