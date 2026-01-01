"""
Traffic Data Processor for Smart City Traffic System.
Handles data preprocessing, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class TrafficDataProcessor:
    """Process and engineer features from traffic data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.config = get_config()
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        self.test_size = self.preprocessing_config.get('test_size', 0.2)
        self.val_size = self.preprocessing_config.get('validation_size', 0.1)
        self.random_state = self.preprocessing_config.get('random_state', 42)
        
        # Feature engineering parameters
        self.lag_features = self.preprocessing_config.get('lag_features', [1, 2, 3, 6, 12, 24])
        self.rolling_windows = self.preprocessing_config.get('rolling_window_sizes', [4, 8, 12])
        
        # Congestion thresholds
        self.thresholds = {
            'free_flow': self.preprocessing_config.get('free_flow_threshold', 0.8),
            'moderate': self.preprocessing_config.get('moderate_threshold', 0.6),
            'heavy': self.preprocessing_config.get('heavy_threshold', 0.4)
        }
        
        self.scalers = {}
        self.encoders = {}
        
        logger.info("Traffic Data Processor initialized")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load generated traffic data.
        
        Returns:
            Tuple of (road_network, traffic_data, traffic_signals)
        """
        logger.info("Loading traffic data...")
        
        data_path = self.config.get_path('data_raw')
        
        # Load from pickle for faster loading
        road_network = pd.read_pickle(data_path / 'road_network.pkl')
        traffic_data = pd.read_pickle(data_path / 'traffic_data.pkl')
        traffic_signals = pd.read_pickle(data_path / 'traffic_signals.pkl')
        
        logger.success(f"Loaded {len(traffic_data):,} traffic records")
        
        return road_network, traffic_data, traffic_signals
    
    def add_congestion_labels(self, df: pd.DataFrame, 
                             road_network: pd.DataFrame) -> pd.DataFrame:
        """
        Add congestion level labels based on speed relative to speed limit.
        
        Args:
            df: Traffic data DataFrame
            road_network: Road network DataFrame with speed limits
            
        Returns:
            DataFrame with congestion labels added
        """
        logger.info("Adding congestion labels...")
        
        # Merge with road network to get speed limits
        df = df.merge(
            road_network[['segment_id', 'speed_limit', 'road_type']], 
            on='segment_id', 
            how='left'
        )
        
        # Calculate speed ratio (actual speed / speed limit)
        df['speed_ratio'] = df['speed_kmh'] / df['speed_limit']
        
        # Create congestion labels
        conditions = [
            df['speed_ratio'] >= self.thresholds['free_flow'],
            df['speed_ratio'] >= self.thresholds['moderate'],
            df['speed_ratio'] >= self.thresholds['heavy'],
        ]
        
        labels = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
        df['congestion_level'] = np.select(conditions, labels[:-1], default=labels[-1])
        
        # Encode labels
        df['congestion_code'] = df['congestion_level'].map({
            'Free Flow': 0,
            'Moderate': 1,
            'Heavy': 2,
            'Severe': 3
        })
        
        logger.success("Congestion labels added")
        logger.info(f"Congestion distribution:\n{df['congestion_level'].value_counts()}")
        
        return df
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal features from timestamp.
        
        Args:
            df: Traffic data DataFrame
            
        Returns:
            DataFrame with temporal features added
        """
        logger.info("Engineering temporal features...")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding for hour and day of week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Boolean features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['is_rush_hour'].astype(int)
        
        logger.success("Temporal features engineered")
        
        return df
    
    def engineer_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time series modeling.
        
        Args:
            df: Traffic data DataFrame
            
        Returns:
            DataFrame with lag features added
        """
        logger.info("Engineering lag features...")
        
        df = df.sort_values(['segment_id', 'timestamp'])
        
        # Create lag features for key metrics
        for lag in self.lag_features:
            df[f'speed_lag_{lag}'] = df.groupby('segment_id')['speed_kmh'].shift(lag)
            df[f'volume_lag_{lag}'] = df.groupby('segment_id')['volume_vehicles'].shift(lag)
            df[f'density_lag_{lag}'] = df.groupby('segment_id')['density_vehicles_per_km'].shift(lag)
        
        logger.success(f"Created lag features for lags: {self.lag_features}")
        
        return df
    
    def engineer_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Traffic data DataFrame
            
        Returns:
            DataFrame with rolling features added
        """
        logger.info("Engineering rolling features...")
        
        df = df.sort_values(['segment_id', 'timestamp'])
        
        for window in self.rolling_windows:
            # Rolling mean
            df[f'speed_rolling_mean_{window}'] = (
                df.groupby('segment_id')['speed_kmh']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling std
            df[f'speed_rolling_std_{window}'] = (
                df.groupby('segment_id')['speed_kmh']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling max
            df[f'volume_rolling_max_{window}'] = (
                df.groupby('segment_id')['volume_vehicles']
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )
        
        logger.success(f"Created rolling features for windows: {self.rolling_windows}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        missing_before = df.isnull().sum().sum()
        
        # Forward fill for lag features (reasonable for time series)
        lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        df[lag_cols] = df.groupby('segment_id')[lag_cols].fillna(method='ffill')
        
        # Fill remaining with 0 or median
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
        
        missing_after = df.isnull().sum().sum()
        
        logger.success(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return df
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               task: str = 'classification') -> Tuple:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Processed DataFrame
            task: 'classification' or 'regression'
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info(f"Preparing features for {task}...")
        
        # Define feature columns (exclude identifiers and targets)
        exclude_cols = [
            'timestamp', 'segment_id', 'speed_kmh', 'congestion_level', 
            'congestion_code', 'speed_limit', 'speed_ratio', 'travel_time_min',
            'has_accident'  # Don't use accident as feature (it's not predictable)
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Prepare target
        if task == 'classification':
            y = df['congestion_code']
        else:  # regression
            y = df['speed_kmh']
        
        logger.success(f"Prepared {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:10]}... (showing first 10)")
        
        return X, y, feature_cols
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   temporal: bool = True) -> Dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            temporal: If True, use temporal split (recommended for time series)
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info(f"Splitting data (temporal={temporal})...")
        
        if temporal:
            # Temporal split - last 20% for test, 10% before that for validation
            n = len(X)
            test_idx = int(n * (1 - self.test_size))
            val_idx = int(n * (1 - self.test_size - self.val_size))
            
            X_train = X.iloc[:val_idx]
            y_train = y.iloc[:val_idx]
            
            X_val = X.iloc[val_idx:test_idx]
            y_val = y.iloc[val_idx:test_idx]
            
            X_test = X.iloc[test_idx:]
            y_test = y.iloc[test_idx:]
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
            )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        logger.success(f"Data split complete:")
        logger.info(f"  Train: {len(X_train):,} samples")
        logger.info(f"  Val:   {len(X_val):,} samples")
        logger.info(f"  Test:  {len(X_test):,} samples")
        
        return splits
    
    def scale_features(self, splits: Dict) -> Dict:
        """
        Scale features using StandardScaler.
        
        Args:
            splits: Dictionary with train/val/test splits
            
        Returns:
            Dictionary with scaled splits
        """
        logger.info("Scaling features...")
        
        scaler = StandardScaler()
        
        # Fit on training data only
        X_train_scaled = scaler.fit_transform(splits['X_train'])
        X_val_scaled = scaler.transform(splits['X_val'])
        X_test_scaled = scaler.transform(splits['X_test'])
        
        # Convert back to DataFrames
        scaled_splits = {
            'X_train': pd.DataFrame(X_train_scaled, columns=splits['X_train'].columns),
            'X_val': pd.DataFrame(X_val_scaled, columns=splits['X_val'].columns),
            'X_test': pd.DataFrame(X_test_scaled, columns=splits['X_test'].columns),
            'y_train': splits['y_train'],
            'y_val': splits['y_val'],
            'y_test': splits['y_test']
        }
        
        # Save scaler
        self.scalers['standard_scaler'] = scaler
        
        logger.success("Features scaled")
        
        return scaled_splits
    
    def save_processed_data(self, df: pd.DataFrame, splits: Dict, 
                           feature_names: list, filename: str = 'processed_traffic_data'):
        """
        Save processed data and splits.
        
        Args:
            df: Processed DataFrame
            splits: Dictionary with train/val/test splits
            feature_names: List of feature column names
            filename: Base filename for saving
        """
        logger.info("Saving processed data...")
        
        processed_path = self.config.get_path('data_processed')
        
        # Save full processed dataframe
        df.to_pickle(processed_path / f'{filename}.pkl')
        df.to_csv(processed_path / f'{filename}.csv', index=False)
        
        # Save splits
        with open(processed_path / 'data_splits.pkl', 'wb') as f:
            pickle.dump(splits, f)
        
        # Save feature names
        with open(processed_path / 'feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        
        # Save scalers
        with open(processed_path / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        logger.success(f"Processed data saved to {processed_path}")
        
    def process_all(self, task: str = 'classification'):
        """
        Run complete preprocessing pipeline.
        
        Args:
            task: 'classification' or 'regression'
        """
        logger.info("=" * 60)
        logger.info("Starting Data Processing Pipeline")
        logger.info("=" * 60)
        
        # Load data
        road_network, traffic_data, traffic_signals = self.load_data()
        
        # Add labels
        traffic_data = self.add_congestion_labels(traffic_data, road_network)
        
        # Feature engineering
        traffic_data = self.engineer_temporal_features(traffic_data)
        traffic_data = self.engineer_lag_features(traffic_data)
        traffic_data = self.engineer_rolling_features(traffic_data)
        
        # Handle missing values
        traffic_data = self.handle_missing_values(traffic_data)
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features_target(traffic_data, task)
        
        # Split data
        splits = self.split_data(X, y, temporal=True)
        
        # Scale features
        scaled_splits = self.scale_features(splits)
        
        # Save processed data
        self.save_processed_data(traffic_data, scaled_splits, feature_names)
        
        logger.success("=" * 60)
        logger.success("Data Processing Complete!")
        logger.success("=" * 60)
        
        return traffic_data, scaled_splits, feature_names


def main():
    """Main function to process traffic data."""
    processor = TrafficDataProcessor()
    processor.process_all(task='classification')


if __name__ == "__main__":
    main()
