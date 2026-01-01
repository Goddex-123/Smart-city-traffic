"""
Traffic Simulator for Smart City Traffic System.
Simulates traffic flow and evaluates signal timing scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class TrafficSimulator:
    """Simulate traffic flow through signalized intersections."""
    
    def __init__(self):
        """Initialize the traffic simulator."""
        self.config = get_config()
        self.sim_config = self.config.get('optimization', {})
        
        self.simulation_duration = self.sim_config.get('simulation_duration', 3600)  # 1 hour
        self.vehicle_length = self.sim_config.get('vehicle_length', 4.5)  # meters
        self.safety_distance = self.sim_config.get('safety_distance', 2)  # meters
        
        logger.info("Traffic Simulator initialized")
        logger.info(f"Simulation duration: {self.simulation_duration} seconds")
    
    def simulate_intersection(self, green_ns: int, green_ew: int,
                             yellow: int, all_red: int,
                             volume_ns: float, volume_ew: float,
                             duration: int = None) -> Dict:
        """
        Simulate traffic flow through an intersection.
        
        Args:
            green_ns: North-South green time (seconds)
            green_ew: East-West green time (seconds)
            yellow: Yellow time (seconds)
            all_red: All-red time (seconds)
            volume_ns: North-South volume (vehicles/hour)
            volume_ew: East-West volume (vehicles/hour)
            duration: Simulation duration (seconds)
            
        Returns:
            Dictionary with simulation results
        """
        if duration is None:
            duration = self.simulation_duration
        
        # Calculate cycle time
        cycle_time = green_ns + green_ew + 2 * yellow + 2 * all_red
        num_cycles = int(duration / cycle_time)
        
        # Convert hourly volume to arrival rate (vehicles/second)
        arrival_rate_ns = volume_ns / 3600
        arrival_rate_ew = volume_ew / 3600
        
        # Saturation flow rate (vehicles/second that can pass during green)
        saturation_flow = 0.5  # vehicles/second/lane (assumed)
        
        # Simulation variables
        total_wait_time = 0
        total_vehicles_served = 0
        total_vehicles_arrived = 0
        queue_ns, queue_ew = 0, 0
        max_queue_ns, max_queue_ew = 0, 0
        total_stops = 0
        
        # Simulate each cycle
        for cycle in range(num_cycles):
            # Vehicles arriving during this cycle
            arrivals_ns = np.random.poisson(arrival_rate_ns * cycle_time)
            arrivals_ew = np.random.poisson(arrival_rate_ew * cycle_time)
            
            total_vehicles_arrived += arrivals_ns + arrivals_ew
            
            # Add to queues
            queue_ns += arrivals_ns
            queue_ew += arrivals_ew
            
            # North-South green phase
            vehicles_served_ns = min(queue_ns, int(green_ns * saturation_flow))
            queue_ns -= vehicles_served_ns
            total_vehicles_served += vehicles_served_ns
            
            # Those in queue wait
            total_wait_time += queue_ns * green_ns
            total_stops += vehicles_served_ns
            
            # East-West green phase
            vehicles_served_ew = min(queue_ew, int(green_ew * saturation_flow))
            queue_ew -= vehicles_served_ew
            total_vehicles_served += vehicles_served_ew
            
            # Those in queue wait
            total_wait_time += queue_ew * green_ew
            total_stops += vehicles_served_ew
            
            # Update max queues
            max_queue_ns = max(max_queue_ns, queue_ns)
            max_queue_ew = max(max_queue_ew, queue_ew)
        
        # Calculate metrics
        avg_wait_time = total_wait_time / (total_vehicles_served + 1e-6)  # seconds
        throughput = total_vehicles_served / duration * 3600  # vehicles/hour
        utilization = total_vehicles_served / (total_vehicles_arrived + 1e-6)
        
        # Queue metrics
        avg_queue_length = (max_queue_ns + max_queue_ew) / 2
        
        results = {
            'total_vehicles_arrived': total_vehicles_arrived,
            'total_vehicles_served': total_vehicles_served,
            'avg_wait_time_sec': avg_wait_time,
            'throughput_veh_per_hour': throughput,
            'utilization': utilization,
            'max_queue_length': avg_queue_length,
            'total_stops': total_stops,
            'num_cycles': num_cycles
        }
        
        return results
    
    def compare_scenarios(self, original_signals: pd.DataFrame,
                         optimized_signals: pd.DataFrame) -> Dict:
        """
        Compare baseline vs optimized signal scenarios.
        
        Args:
            original_signals: Original signal timings
            optimized_signals: Optimized signal timings
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing baseline vs optimized scenarios...")
        
        baseline_results = []
        optimized_results = []
        
        for _, signal in optimized_signals.iterrows():
            # Baseline scenario
            baseline = self.simulate_intersection(
                green_ns=signal['green_time_ns_original'],
                green_ew=signal['green_time_ew_original'],
                yellow=signal['yellow_time'],
                all_red=signal['all_red_time'],
                volume_ns=signal['traffic_volume_ns'],
                volume_ew=signal['traffic_volume_ew']
            )
            baseline['signal_id'] = signal['signal_id']
            baseline_results.append(baseline)
            
            # Optimized scenario
            optimized = self.simulate_intersection(
                green_ns=signal['green_time_ns'],
                green_ew=signal['green_time_ew'],
                yellow=signal['yellow_time'],
                all_red=signal['all_red_time'],
                volume_ns=signal['traffic_volume_ns'],
                volume_ew=signal['traffic_volume_ew']
            )
            optimized['signal_id'] = signal['signal_id']
            optimized_results.append(optimized)
        
        baseline_df = pd.DataFrame(baseline_results)
        optimized_df = pd.DataFrame(optimized_results)
        
        # Calculate improvements
        improvements = {
            'wait_time_reduction': (
                (baseline_df['avg_wait_time_sec'].mean() - 
                 optimized_df['avg_wait_time_sec'].mean()) /
                baseline_df['avg_wait_time_sec'].mean() * 100
            ),
            'throughput_increase': (
                (optimized_df['throughput_veh_per_hour'].mean() - 
                 baseline_df['throughput_veh_per_hour'].mean()) /
                baseline_df['throughput_veh_per_hour'].mean() * 100
            ),
            'queue_reduction': (
                (baseline_df['max_queue_length'].mean() - 
                 optimized_df['max_queue_length'].mean()) /
                baseline_df['max_queue_length'].mean() * 100
            ),
            'baseline_avg_wait': baseline_df['avg_wait_time_sec'].mean(),
            'optimized_avg_wait': optimized_df['avg_wait_time_sec'].mean(),
            'baseline_throughput': baseline_df['throughput_veh_per_hour'].mean(),
            'optimized_throughput': optimized_df['throughput_veh_per_hour'].mean()
        }
        
        logger.success("Simulation comparison complete:")
        logger.info(f"  Wait Time Reduction: {improvements['wait_time_reduction']:.2f}%")
        logger.info(f"  Throughput Increase: {improvements['throughput_increase']:.2f}%")
        logger.info(f"  Queue Length Reduction: {improvements['queue_reduction']:.2f}%")
        
        return {
            'baseline': baseline_df,
            'optimized': optimized_df,
            'improvements': improvements
        }
    
    def calculate_benefits(self, comparison: Dict) -> Dict:
        """
        Calculate environmental and economic benefits.
        
        Args:
            comparison: Comparison results from compare_scenarios
            
        Returns:
            Dictionary with benefit calculations
        """
        logger.info("Calculating benefits...")
        
        benefits_config = self.config.get('benefits', {})
        
        baseline = comparison['baseline']
        optimized = comparison['optimized']
        
        # Wait time savings
        total_baseline_wait = baseline['avg_wait_time_sec'].sum() * baseline['total_vehicles_served'].sum()
        total_optimized_wait = optimized['avg_wait_time_sec'].sum() * optimized['total_vehicles_served'].sum()
        wait_time_saved_hours = (total_baseline_wait - total_optimized_wait) / 3600
        
        # CO2 emissions
        co2_idle = benefits_config.get('co2_idle', 180)  # g/km while idling
        
        # Assume average idle time proportional to wait time
        baseline_idle_hours = total_baseline_wait / 3600
        optimized_idle_hours = total_optimized_wait / 3600
        
        co2_baseline = baseline_idle_hours * co2_idle  # kg
        co2_optimized = optimized_idle_hours * co2_idle  # kg
        co2_reduction = co2_baseline - co2_optimized
        co2_reduction_pct = (co2_reduction / co2_baseline) * 100 if co2_baseline > 0 else 0
        
        # Economic benefits
        time_value_per_hour = benefits_config.get('time_value_per_hour', 15)  # USD
        fuel_cost_per_liter = benefits_config.get('fuel_cost_per_liter', 1.5)
        fuel_consumption_idle = benefits_config.get('fuel_consumption_idle', 0.6)  # liters/hour
        
        time_savings_value = wait_time_saved_hours * time_value_per_hour
        fuel_savings_liters = (baseline_idle_hours - optimized_idle_hours) * fuel_consumption_idle
        fuel_savings_value = fuel_savings_liters * fuel_cost_per_liter
        
        total_economic_benefit = time_savings_value + fuel_savings_value
        
        benefits = {
            'wait_time_saved_hours': wait_time_saved_hours,
            'co2_reduction_kg': co2_reduction,
            'co2_reduction_pct': co2_reduction_pct,
            'time_savings_value_usd': time_savings_value,
            'fuel_savings_liters': fuel_savings_liters,
            'fuel_savings_value_usd': fuel_savings_value,
            'total_economic_benefit_usd': total_economic_benefit
        }
        
        logger.success("Benefits quantified:")
        logger.info(f"  Time Saved: {wait_time_saved_hours:.1f} hours")
        logger.info(f"  CO2 Reduction: {co2_reduction:.1f} kg ({co2_reduction_pct:.1f}%)")
        logger.info(f"  Economic Benefit: ${total_economic_benefit:.2f}")
        
        return benefits
    
    def save_simulation_results(self, comparison: Dict, benefits: Dict):
        """
        Save simulation results.
        
        Args:
            comparison: Comparison results
            benefits: Benefits calculation
        """
        logger.info("Saving simulation results...")
        
        processed_path = self.config.get_path('data_processed')
        
        # Save comparison dataframes
        comparison['baseline'].to_csv(processed_path / 'simulation_baseline.csv', index=False)
        comparison['optimized'].to_csv(processed_path / 'simulation_optimized.csv', index=False)
        
        # Save all results
        results = {
            'comparison': comparison,
            'benefits': benefits
        }
        
        with open(processed_path / 'simulation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        logger.success(f"Simulation results saved to {processed_path}")
    
    def run_simulation(self):
        """Run complete simulation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Traffic Simulation")
        logger.info("=" * 60)
        
        # Load optimized signals
        processed_path = self.config.get_path('data_processed')
        optimized_signals = pd.read_pickle(processed_path / 'optimized_signals.pkl')
        
        # Compare scenarios
        comparison = self.compare_scenarios(optimized_signals, optimized_signals)
        
        # Calculate benefits
        benefits = self.calculate_benefits(comparison)
        
        # Save results
        self.save_simulation_results(comparison, benefits)
        
        logger.success("=" * 60)
        logger.success("Traffic Simulation Complete!")
        logger.success("=" * 60)
        
        return comparison, benefits


def main():
    """Main function for traffic simulation."""
    simulator = TrafficSimulator()
    simulator.run_simulation()


if __name__ == "__main__":
    main()
