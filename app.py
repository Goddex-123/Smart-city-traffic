"""
Smart City Traffic Congestion Prediction & Optimization System
Root app.py for Streamlit Cloud deployment
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
from src.visualization.dashboard import *
