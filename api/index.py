# -*- coding: utf-8 -*-
"""Vercel serverless entry point."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dash_app import app

server = app.server
