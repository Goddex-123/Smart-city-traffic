"""
Time Series Forecasting Models for Smart City Traffic System.
Implements ARIMA, Prophet, and LSTM for traffic prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class TimeSeriesForecaster:
    """Train and evaluate time series forecasting models."""
    
    def __init__(self):
        """Initialize the forecaster."""
        self.config = get_config()
        self.model_config = self.config.get('models', {})
        
        self.models = {}
        self.results = {}
        
        logger.info("Time Series Forecaster initialized")
    
    def load_traffic_data(self, segment_id: str = None) -> pd.DataFrame:
        """
        Load traffic data for time series forecasting.
        
        Args:
            segment_id: Specific segment to load (if None, uses first segment)
            
        Returns:
            Time series DataFrame for the segment
        """
        logger.info("Loading traffic data for forecasting...")
        
        processed_path = self.config.get_path('data_processed')
        df = pd.read_pickle(processed_path / 'processed_traffic_data.pkl')
        
        # Select a specific segment for demonstration
        if segment_id is None:
            segment_id = df['segment_id'].iloc[0]
        
        # Filter for one segment
        segment_data = df[df['segment_id'] == segment_id].copy()
        segment_data = segment_data.sort_values('timestamp')
        segment_data = segment_data.set_index('timestamp')
        
        logger.success(f"Loaded {len(segment_data)} records for segment {segment_id}")
        
        return segment_data
    
    def prepare_lstm_data(self, data: pd.DataFrame, 
                         lookback: int = 24, 
                         horizon: int = 4) -> Tuple:
        """
        Prepare data for LSTM model.
        
        Args:
            data: Time series data
            lookback: Number of past timesteps to use
            horizon: Number of future timesteps to predict
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info(f"Preparing LSTM data (lookback={lookback}, horizon={horizon})...")
        
        # Use speed as target
        values = data['speed_kmh'].values
        
        X, y = [], []
        for i in range(lookback, len(values) - horizon):
            X.append(values[i-lookback:i])
            y.append(values[i:i+horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        logger.success(f"LSTM data prepared: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def train_arima(self, data: pd.DataFrame) -> Dict:
        """
        Train ARIMA model for traffic forecasting.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with model and predictions
        """
        logger.info("Training ARIMA model...")
        
        # Use speed as target
        series = data['speed_kmh']
        
        # Train/test split
        split_idx = int(0.8 * len(series))
        train, test = series[:split_idx], series[split_idx:]
        
        # Fit ARIMA model (order determined by auto-correlation)
        try:
            model = ARIMA(train, order=(5, 1, 2))  # (p, d, q)
            fitted_model = model.fit()
            
            # Forecast
            forecast_steps = len(test)
            forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Calculate metrics
            mse = mean_squared_error(test, forecast)
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test - forecast) / test)) * 100
            
            logger.success(f"ARIMA trained - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            
            results = {
                'model': fitted_model,
                'forecast': forecast,
                'actual': test,
                'metrics': {'rmse': rmse, 'mae': mae, 'mape': mape}
            }
            
            self.models['arima'] = fitted_model
            self.results['arima'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return None
    
    def train_prophet(self, data: pd.DataFrame) -> Dict:
        """
        Train Prophet model for traffic forecasting.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with model and predictions
        """
        logger.info("Training Prophet model...")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_data = data.reset_index()[['timestamp', 'speed_kmh']]
        prophet_data.columns = ['ds', 'y']
        
        # Train/test split
        split_idx = int(0.8 * len(prophet_data))
        train = prophet_data[:split_idx]
        test = prophet_data[split_idx:]
        
        # Configure Prophet
        prophet_config = self.model_config.get('prophet', {})
        model = Prophet(
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_config.get('seasonality_prior_scale', 10),
            seasonality_mode=prophet_config.get('seasonality_mode', 'multiplicative'),
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        
        # Train
        model.fit(train)
        
        # Forecast
        future = model.make_future_dataframe(periods=len(test), freq='15min')
        forecast = model.predict(future)
        
        # Get predictions for test period
        test_forecast = forecast.iloc[-len(test):]
        
        # Calculate metrics
        mse = mean_squared_error(test['y'], test_forecast['yhat'])
        mae = mean_absolute_error(test['y'], test_forecast['yhat'])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test['y'] - test_forecast['yhat']) / test['y'])) * 100
        
        logger.success(f"Prophet trained - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        
        results = {
            'model': model,
            'forecast': test_forecast,
            'actual': test,
            'metrics': {'rmse': rmse, 'mae': mae, 'mape': mape}
        }
        
        self.models['prophet'] = model
        self.results['prophet'] = results
        
        return results
    
    def train_lstm(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train LSTM model for traffic forecasting.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with model and predictions
        """
        logger.info("Training LSTM model...")
        
        lstm_config = self.model_config.get('lstm', {})
        
        # Build LSTM model
        model = models.Sequential([
            layers.LSTM(lstm_config.get('units', 128), 
                       return_sequences=True, 
                       input_shape=(X_train.shape[1], 1)),
            layers.Dropout(lstm_config.get('dropout', 0.2)),
            
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(lstm_config.get('dropout', 0.2)),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(y_train.shape[1])  # Output horizon
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lstm_config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=lstm_config.get('epochs', 50),
            batch_size=lstm_config.get('batch_size', 64),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Predict
        predictions = model.predict(X_test, verbose=0)
        
        # Calculate metrics (average across horizon)
        mse = mean_squared_error(y_test.flatten(), predictions.flatten())
        mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        logger.success(f"LSTM trained - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        
        results = {
            'model': model,
            'predictions': predictions,
            'actual': y_test,
            'history': history.history,
            'metrics': {'rmse': rmse, 'mae': mae, 'mape': mape}
        }
        
        self.models['lstm'] = model
        self.results['lstm'] = results
        
        return results
    
    def save_models(self):
        """Save trained forecasting models."""
        logger.info("Saving forecasting models...")
        
        models_path = self.config.get_path('models')
        
        # Save LSTM model
        if 'lstm' in self.models:
            self.models['lstm'].save(models_path / 'lstm_forecasting_model.h5')
        
        # Save Prophet model
        if 'prophet' in self.models:
            with open(models_path / 'prophet_model.pkl', 'wb') as f:
                pickle.dump(self.models['prophet'], f)
        
        # Save results
        with open(models_path / 'forecasting_results.pkl', 'wb') as f:
            # Remove Prophet model from results before saving (not picklable)
            results_to_save = {k: v for k, v in self.results.items() if k != 'prophet'}
            pickle.dump(results_to_save, f)
        
        logger.success(f"Forecasting models saved to {models_path}")
    
    def train_all(self, segment_id: str = None):
        """
        Train all forecasting models.
        
        Args:
            segment_id: Segment ID to train on (if None, uses first segment)
        """
        logger.info("=" * 60)
        logger.info("Starting Forecasting Model Training")
        logger.info("=" * 60)
        
        # Load data
        data = self.load_traffic_data(segment_id)
        
        # Train ARIMA
        self.train_arima(data)
        
        # Train Prophet
        self.train_prophet(data)
        
        # Train LSTM
        X_train, y_train, X_test, y_test = self.prepare_lstm_data(data)
        self.train_lstm(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_models()
        
        logger.success("=" * 60)
        logger.success("Forecasting Model Training Complete!")
        logger.success("=" * 60)
        
        # Print comparison
        logger.info("\nModel Comparison:")
        for name, results in self.results.items():
            metrics = results['metrics']
            logger.info(f"{name.upper():10s} - RMSE: {metrics['rmse']:6.2f} | "
                       f"MAE: {metrics['mae']:6.2f} | MAPE: {metrics['mape']:6.2f}%")
        
        return self.models, self.results


def main():
    """Main function to train forecasting models."""
    forecaster = TimeSeriesForecaster()
    forecaster.train_all()


if __name__ == "__main__":
    main()
