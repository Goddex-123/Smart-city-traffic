"""Smart City Traffic Congestion Prediction & Optimization System"""

__version__ = "1.0.0"
__author__ = "Smart City Traffic Team"

from src.data import TrafficDataGenerator, TrafficDataProcessor
from src.models import TimeSeriesForecaster, CongestionClassifier, ModelEvaluator
from src.optimization import SignalOptimizer, TrafficSimulator
from src.utils import get_config, get_logger

__all__ = [
    'TrafficDataGenerator',
    'TrafficDataProcessor',
    'TimeSeriesForecaster',
    'CongestionClassifier',
    'ModelEvaluator',
    'SignalOptimizer',
    'TrafficSimulator',
    'get_config',
    'get_logger'
]
