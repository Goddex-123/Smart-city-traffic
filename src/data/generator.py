"""
Traffic Data Generator for Smart City Traffic System.
Generates synthetic traffic data with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class TrafficDataGenerator:
    """Generate synthetic traffic data with realistic patterns."""
    
    def __init__(self):
        """Initialize the data generator."""
        self.config = get_config()
        self.params = self.config.get('data_generation', {})
        
        # Road network parameters
        self.num_segments = self.params.get('num_road_segments', 120)
        self.num_intersections = self.params.get('num_intersections', 30)
        
        # Time parameters
        self.start_date = datetime.strptime(self.params.get('start_date'), '%Y-%m-%d')
        self.end_date = datetime.strptime(self.params.get('end_date'), '%Y-%m-%d')
        self.time_resolution = self.params.get('time_resolution', 15)  # minutes
        
        # Traffic patterns
        self.morning_rush = (self.params.get('morning_rush_start', 7), 
                            self.params.get('morning_rush_end', 9))
        self.evening_rush = (self.params.get('evening_rush_start', 17), 
                            self.params.get('evening_rush_end', 19))
        
        # Random events
        self.accident_prob = self.params.get('accident_probability', 0.001)
        self.weather_impact_prob = self.params.get('weather_impact_probability', 0.15)
        
        # Speed limits
        self.speed_limits = {
            'highway': self.params.get('highway_speed_limit', 100),
            'arterial': self.params.get('arterial_speed_limit', 60),
            'local': self.params.get('local_speed_limit', 40)
        }
        
        logger.info(f"Traffic Data Generator initialized")
        logger.info(f"Generating data from {self.start_date} to {self.end_date}")
        logger.info(f"Road segments: {self.num_segments}, Intersections: {self.num_intersections}")
    
    def generate_road_network(self) -> pd.DataFrame:
        """
        Generate road network with segments and their properties.
        
        Returns:
            DataFrame with road segment information
        """
        logger.info("Generating road network...")
        
        # Road types distribution
        road_types = np.random.choice(
            ['highway', 'arterial', 'local'],
            size=self.num_segments,
            p=[0.2, 0.4, 0.4]  # 20% highway, 40% arterial, 40% local
        )
        
        # Create road network
        roads = []
        for i in range(self.num_segments):
            road_type = road_types[i]
            
            road = {
                'segment_id': f'SEG_{i:03d}',
                'road_type': road_type,
                'speed_limit': self.speed_limits[road_type],
                'length_km': np.random.uniform(0.5, 3.0),  # 0.5 to 3 km
                'num_lanes': np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]),
                'latitude': 40.7128 + np.random.uniform(-0.1, 0.1),  # NYC area
                'longitude': -74.0060 + np.random.uniform(-0.1, 0.1),
                'has_traffic_signal': np.random.random() < 0.3,  # 30% have signals
            }
            roads.append(road)
        
        road_network = pd.DataFrame(roads)
        logger.success(f"Generated {len(road_network)} road segments")
        
        return road_network
    
    def generate_traffic_data(self, road_network: pd.DataFrame) -> pd.DataFrame:
        """
        Generate traffic data for all segments over time.
        
        Args:
            road_network: DataFrame with road segment information
            
        Returns:
            DataFrame with traffic measurements
        """
        logger.info("Generating traffic data...")
        
        # Create time index
        time_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f'{self.time_resolution}min'
        )
        
        traffic_data = []
        total_records = len(road_network) * len(time_index)
        
        logger.info(f"Generating {total_records:,} traffic records...")
        
        for _, road in road_network.iterrows():
            segment_id = road['segment_id']
            speed_limit = road['speed_limit']
            road_type = road['road_type']
            
            # Generate time series for this segment
            for timestamp in time_index:
                # Base traffic pattern
                speed, volume = self._calculate_traffic_pattern(
                    timestamp, speed_limit, road_type
                )
                
                # Add random variations
                speed *= np.random.uniform(0.9, 1.1)
                volume *= np.random.uniform(0.8, 1.2)
                
                # Weather impact (random bad weather days)
                if np.random.random() < self.weather_impact_prob:
                    speed *= 0.7
                    volume *= 1.3
                
                # Accident events (rare)
                has_accident = np.random.random() < self.accident_prob
                if has_accident:
                    speed *= 0.3  # Severe slowdown
                    volume *= 1.5  # Congestion
                
                # Calculate derived metrics
                density = volume / speed if speed > 0 else 0
                travel_time = (road['length_km'] / speed) * 60 if speed > 0 else 999  # minutes
                
                record = {
                    'timestamp': timestamp,
                    'segment_id': segment_id,
                    'speed_kmh': max(5, min(speed, speed_limit)),  # Clamp to reasonable range
                    'volume_vehicles': int(max(0, volume)),
                    'density_vehicles_per_km': density,
                    'travel_time_min': travel_time,
                    'has_accident': has_accident,
                    'day_of_week': timestamp.dayofweek,
                    'hour': timestamp.hour,
                    'is_weekend': timestamp.dayofweek >= 5,
                    'is_rush_hour': self._is_rush_hour(timestamp.hour)
                }
                
                traffic_data.append(record)
        
        df = pd.DataFrame(traffic_data)
        logger.success(f"Generated {len(df):,} traffic records")
        
        return df
    
    def _calculate_traffic_pattern(self, timestamp: datetime, 
                                   speed_limit: float, 
                                   road_type: str) -> Tuple[float, float]:
        """
        Calculate speed and volume based on time patterns.
        
        Args:
            timestamp: Current timestamp
            speed_limit: Speed limit for the road
            road_type: Type of road (highway, arterial, local)
            
        Returns:
            Tuple of (speed_kmh, volume_vehicles_per_hour)
        """
        hour = timestamp.hour
        is_weekend = timestamp.dayofweek >= 5
        
        # Base volume (vehicles per hour per lane)
        if road_type == 'highway':
            base_volume = 1000
        elif road_type == 'arterial':
            base_volume = 600
        else:  # local
            base_volume = 300
        
        # Time-based adjustment
        if is_weekend:
            # Lower traffic on weekends
            volume_multiplier = 0.6
            speed_multiplier = 1.0
        else:
            # Rush hour patterns
            if self.morning_rush[0] <= hour < self.morning_rush[1]:
                # Morning rush
                volume_multiplier = 1.8
                speed_multiplier = 0.6
            elif self.evening_rush[0] <= hour < self.evening_rush[1]:
                # Evening rush (heavier)
                volume_multiplier = 2.0
                speed_multiplier = 0.5
            elif 10 <= hour < 15:
                # Midday moderate
                volume_multiplier = 1.2
                speed_multiplier = 0.85
            elif 22 <= hour or hour < 6:
                # Night - low traffic
                volume_multiplier = 0.3
                speed_multiplier = 1.0
            else:
                # Normal
                volume_multiplier = 1.0
                speed_multiplier = 0.9
        
        volume = base_volume * volume_multiplier
        speed = speed_limit * speed_multiplier
        
        return speed, volume
    
    def _is_rush_hour(self, hour: int) -> bool:
        """Check if given hour is rush hour."""
        return (self.morning_rush[0] <= hour < self.morning_rush[1] or
                self.evening_rush[0] <= hour < self.evening_rush[1])
    
    def generate_traffic_signals(self, road_network: pd.DataFrame) -> pd.DataFrame:
        """
        Generate traffic signal data for intersections.
        
        Args:
            road_network: DataFrame with road segment information
            
        Returns:
            DataFrame with traffic signal information
        """
        logger.info("Generating traffic signal data...")
        
        # Select segments with traffic signals
        segments_with_signals = road_network[road_network['has_traffic_signal']].copy()
        
        signals = []
        for i, (_, road) in enumerate(segments_with_signals.iterrows()):
            if i >= self.num_intersections:
                break
            
            signal = {
                'signal_id': f'SIG_{i:03d}',
                'segment_id': road['segment_id'],
                'latitude': road['latitude'],
                'longitude': road['longitude'],
                'cycle_length': 120,  # seconds
                'green_time_ns': 50,  # North-South
                'green_time_ew': 50,  # East-West
                'yellow_time': 3,
                'all_red_time': 2,
            }
            signals.append(signal)
        
        signals_df = pd.DataFrame(signals)
        logger.success(f"Generated {len(signals_df)} traffic signals")
        
        return signals_df
    
    def save_data(self, road_network: pd.DataFrame, 
                  traffic_data: pd.DataFrame,
                  traffic_signals: pd.DataFrame):
        """
        Save generated data to files.
        
        Args:
            road_network: Road network DataFrame
            traffic_data: Traffic measurements DataFrame
            traffic_signals: Traffic signals DataFrame
        """
        logger.info("Saving generated data...")
        
        data_path = self.config.get_path('data_raw')
        
        # Save as CSV
        road_network.to_csv(data_path / 'road_network.csv', index=False)
        traffic_data.to_csv(data_path / 'traffic_data.csv', index=False)
        traffic_signals.to_csv(data_path / 'traffic_signals.csv', index=False)
        
        # Save as pickle for faster loading
        road_network.to_pickle(data_path / 'road_network.pkl')
        traffic_data.to_pickle(data_path / 'traffic_data.pkl')
        traffic_signals.to_pickle(data_path / 'traffic_signals.pkl')
        
        logger.success(f"Data saved to {data_path}")
        logger.info(f"  - Road network: {len(road_network)} segments")
        logger.info(f"  - Traffic data: {len(traffic_data):,} records")
        logger.info(f"  - Traffic signals: {len(traffic_signals)} signals")
        
        # Print data size
        csv_size = sum((data_path / f).stat().st_size 
                      for f in ['road_network.csv', 'traffic_data.csv', 'traffic_signals.csv'])
        logger.info(f"  - Total CSV size: {csv_size / 1024 / 1024:.2f} MB")
    
    def generate_all(self):
        """Generate all data components."""
        logger.info("=" * 60)
        logger.info("Starting Traffic Data Generation")
        logger.info("=" * 60)
        
        # Generate components
        road_network = self.generate_road_network()
        traffic_signals = self.generate_traffic_signals(road_network)
        traffic_data = self.generate_traffic_data(road_network)
        
        # Save data
        self.save_data(road_network, traffic_data, traffic_signals)
        
        logger.success("=" * 60)
        logger.success("Traffic Data Generation Complete!")
        logger.success("=" * 60)
        
        return road_network, traffic_data, traffic_signals


def main():
    """Main function to generate traffic data."""
    generator = TrafficDataGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
