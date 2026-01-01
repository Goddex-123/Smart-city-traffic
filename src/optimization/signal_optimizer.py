"""
Traffic Signal Optimization for Smart City Traffic System.
Optimizes signal timings to minimize congestion and wait times.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
from scipy.optimize import minimize, differential_evolution

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class SignalOptimizer:
    """Optimize traffic signal timings."""
    
    def __init__(self):
        """Initialize the signal optimizer."""
        self.config = get_config()
        self.opt_config = self.config.get('optimization', {})
        
        self.min_green = self.opt_config.get('min_green_time', 20)
        self.max_green = self.opt_config.get('max_green_time', 120)
        self.yellow_time = self.opt_config.get('yellow_time', 3)
        self.all_red_time = self.opt_config.get('all_red_time', 2)
        
        # Objective weights
        self.weight_wait = self.opt_config.get('weight_wait_time', 0.4)
        self.weight_throughput = self.opt_config.get('weight_throughput', 0.4)
        self.weight_emissions = self.opt_config.get('weight_emissions', 0.2)
        
        logger.info("Signal Optimizer initialized")
        logger.info(f"Green time range: {self.min_green}-{self.max_green} seconds")
    
    def load_traffic_signals(self) -> pd.DataFrame:
        """
        Load traffic signal data.
        
        Returns:
            DataFrame with traffic signal information
        """
        logger.info("Loading traffic signals...")
        
        data_path = self.config.get_path('data_raw')
        signals = pd.read_pickle(data_path / 'traffic_signals.pkl')
        
        logger.success(f"Loaded {len(signals)} traffic signals")
        
        return signals
    
    def load_traffic_data(self) -> pd.DataFrame:
        """
        Load traffic data for optimization.
        
        Returns:
            Traffic data DataFrame
        """
        logger.info("Loading traffic data...")
        
        processed_path = self.config.get_path('data_processed')
        traffic_data = pd.read_pickle(processed_path / 'processed_traffic_data.pkl')
        
        logger.success(f"Loaded {len(traffic_data):,} traffic records")
        
        return traffic_data
    
    def calculate_intersection_metrics(self, traffic_data: pd.DataFrame, 
                                       signal: pd.Series) -> Dict:
        """
        Calculate current metrics for an intersection.
        
        Args:
            traffic_data: Traffic data DataFrame
            signal: Signal information Series
            
        Returns:
            Dictionary with current metrics
        """
        # Filter data for this segment
        segment_data = traffic_data[traffic_data['segment_id'] == signal['segment_id']]
        
        if len(segment_data) == 0:
            return None
        
        # Calculate average metrics during peak hours
        peak_data = segment_data[segment_data['is_rush_hour'] == 1]
        
        if len(peak_data) == 0:
            peak_data = segment_data
        
        metrics = {
            'avg_volume': peak_data['volume_vehicles'].mean(),
            'avg_speed': peak_data['speed_kmh'].mean(),
            'avg_density': peak_data['density_vehicles_per_km'].mean(),
            'congestion_hours': (peak_data['congestion_code'] >= 2).sum() / len(peak_data),
            'current_cycle': signal['cycle_length'],
            'current_green_ns': signal['green_time_ns'],
            'current_green_ew': signal['green_time_ew']
        }
        
        return metrics
    
    def objective_function(self, green_times: np.ndarray, 
                          volume_ns: float, volume_ew: float) -> float:
        """
        Objective function to minimize.
        Combines wait time, throughput, and emissions.
        
        Args:
            green_times: Array of [green_ns, green_ew] in seconds
            volume_ns: North-South traffic volume (vehicles/hour)
            volume_ew: East-West traffic volume (vehicles/hour)
            
        Returns:
            Objective value (lower is better)
        """
        green_ns, green_ew = green_times
        
        # Calculate cycle length
        cycle_length = green_ns + green_ew + 2 * self.yellow_time + 2 * self.all_red_time
        
        # Average wait time (Webster's formula approximation)
        wait_ns = (cycle_length - green_ns) / 2
        wait_ew = (cycle_length - green_ew) / 2
        avg_wait = (wait_ns * volume_ns + wait_ew * volume_ew) / (volume_ns + volume_ew + 1e-6)
        
        # Throughput (vehicles per cycle)
        # Assuming saturation flow rate of 0.5 vehicles/second/lane
        throughput_ns = (green_ns * 0.5)
        throughput_ew = (green_ew * 0.5)
        total_throughput = throughput_ns + throughput_ew
        
        # For throughput, we want to maximize, so we minimize negative
        throughput_score = -total_throughput
        
        # Emissions (proportional to idle time)
        # More waiting = more emissions
        emissions_score = avg_wait * (volume_ns + volume_ew) / 100
        
        # Combined objective
        objective = (
            self.weight_wait * avg_wait +
            self.weight_throughput * throughput_score +
            self.weight_emissions * emissions_score
        )
        
        return objective
    
    def optimize_signal(self, volume_ns: float, volume_ew: float,
                       method: str = 'differential_evolution') -> Dict:
        """
        Optimize signal timings for an intersection.
        
        Args:
            volume_ns: North-South traffic volume (vehicles/hour)
            volume_ew: East-West traffic volume (vehicles/hour)
            method: Optimization method
            
        Returns:
            Dictionary with optimized timings
        """
        # Bounds for green times
        bounds = [(self.min_green, self.max_green), (self.min_green, self.max_green)]
        
        # Initial guess (proportional to volume)
        total_volume = volume_ns + volume_ew + 1e-6
        init_green_ns = max(self.min_green, min(self.max_green, 
                                                int(volume_ns / total_volume * 100)))
        init_green_ew = max(self.min_green, min(self.max_green, 
                                                int(volume_ew / total_volume * 100)))
        
        if method == 'differential_evolution':
            # Global optimization
            result = differential_evolution(
                lambda x: self.objective_function(x, volume_ns, volume_ew),
                bounds=bounds,
                maxiter=100,
                seed=42
            )
        else:
            # Local optimization
            result = minimize(
                lambda x: self.objective_function(x, volume_ns, volume_ew),
                x0=[init_green_ns, init_green_ew],
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        green_ns_opt, green_ew_opt = result.x
        
        optimized = {
            'green_time_ns': int(green_ns_opt),
            'green_time_ew': int(green_ew_opt),
            'cycle_length': int(green_ns_opt + green_ew_opt + 
                              2 * self.yellow_time + 2 * self.all_red_time),
            'objective_value': result.fun
        }
        
        return optimized
    
    def optimize_all_signals(self, signals: pd.DataFrame, 
                            traffic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize all traffic signals.
        
        Args:
            signals: Traffic signals DataFrame
            traffic_data: Traffic data DataFrame
            
        Returns:
            DataFrame with optimized signal timings
        """
        logger.info(f"Optimizing {len(signals)} traffic signals...")
        
        optimized_signals = []
        
        for idx, signal in signals.iterrows():
            # Get metrics for this intersection
            metrics = self.calculate_intersection_metrics(traffic_data, signal)
            
            if metrics is None:
                logger.warning(f"No data for signal {signal['signal_id']}, skipping")
                continue
            
            # Optimize
            volume_ns = metrics['avg_volume']  # Simplified: assume symmetric
            volume_ew = metrics['avg_volume'] * 0.8  # Assume EW has 80% of NS traffic
            
            optimized = self.optimize_signal(volume_ns, volume_ew)
            
            # Combine with original signal data
            opt_signal = signal.to_dict()
            opt_signal['green_time_ns_original'] = opt_signal['green_time_ns']
            opt_signal['green_time_ew_original'] = opt_signal['green_time_ew']
            opt_signal['cycle_length_original'] = opt_signal['cycle_length']
            
            opt_signal['green_time_ns'] = optimized['green_time_ns']
            opt_signal['green_time_ew'] = optimized['green_time_ew']
            opt_signal['cycle_length'] = optimized['cycle_length']
            opt_signal['objective_value'] = optimized['objective_value']
            opt_signal['traffic_volume_ns'] = volume_ns
            opt_signal['traffic_volume_ew'] = volume_ew
            
            optimized_signals.append(opt_signal)
            
            logger.debug(f"Signal {signal['signal_id']}: "
                        f"Green NS {opt_signal['green_time_ns_original']}→{opt_signal['green_time_ns']}, "
                        f"Green EW {opt_signal['green_time_ew_original']}→{opt_signal['green_time_ew']}")
        
        opt_df = pd.DataFrame(optimized_signals)
        
        logger.success(f"Optimized {len(opt_df)} traffic signals")
        
        # Calculate improvement statistics
        avg_change_ns = opt_df['green_time_ns'].mean() - opt_df['green_time_ns_original'].mean()
        avg_change_ew = opt_df['green_time_ew'].mean() - opt_df['green_time_ew_original'].mean()
        
        logger.info(f"Average changes:")
        logger.info(f"  NS Green Time: {avg_change_ns:+.1f} seconds")
        logger.info(f"  EW Green Time: {avg_change_ew:+.1f} seconds")
        
        return opt_df
    
    def save_optimized_signals(self, optimized_signals: pd.DataFrame):
        """
        Save optimized signal timings.
        
        Args:
            optimized_signals: DataFrame with optimized signals
        """
        logger.info("Saving optimized signals...")
        
        processed_path = self.config.get_path('data_processed')
        
        optimized_signals.to_csv(processed_path / 'optimized_signals.csv', index=False)
        optimized_signals.to_pickle(processed_path / 'optimized_signals.pkl')
        
        logger.success(f"Optimized signals saved to {processed_path}")
    
    def run_optimization(self):
        """Run complete optimization pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Traffic Signal Optimization")
        logger.info("=" * 60)
        
        # Load data
        signals = self.load_traffic_signals()
        traffic_data = self.load_traffic_data()
        
        # Optimize
        optimized_signals = self.optimize_all_signals(signals, traffic_data)
        
        # Save
        self.save_optimized_signals(optimized_signals)
        
        logger.success("=" * 60)
        logger.success("Traffic Signal Optimization Complete!")
        logger.success("=" * 60)
        
        return optimized_signals


def main():
    """Main function for signal optimization."""
    optimizer = SignalOptimizer()
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
