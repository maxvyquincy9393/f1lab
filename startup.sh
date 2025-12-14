#!/bin/bash
# Production server startup
gunicorn --bind 0.0.0.0:8000 src.dash_app:server
