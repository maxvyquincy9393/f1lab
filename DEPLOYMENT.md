# Deployment Guide

## Streamlit Cloud

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and set entry point to `src/app.py`
4. Deploy

Configuration in `.streamlit/config.toml` handles theming and UI settings.

## Docker

```bash
docker build -t f1-analytics .
docker run -p 8501:8501 f1-analytics
```

Access at `http://localhost:8501`

## Vercel (Dash)

The `api/index.py` module exposes the Dash app for Vercel serverless deployment.

```bash
vercel deploy
```

## Local Development

```bash
.venv\Scripts\activate  # Windows
streamlit run src/app.py
```

## Notes

- FastF1 cache is generated on first load (may take a few minutes)
- Data files in `data/` are loaded automatically
- See `.streamlit/config.toml` for theme configuration
