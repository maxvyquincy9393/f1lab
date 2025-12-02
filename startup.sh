#!/bin/bash
# Azure startup script
pip install -r requirements.txt
streamlit run src/app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
