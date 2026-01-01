"""
Quick Start Script for Smart City Traffic System
Run this to execute the complete pipeline
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import get_logger

logger = get_logger(__name__)


def main():
    """Run complete pipeline."""
    
    logger.info("=" * 80)
    logger.info("SMART CITY TRAFFIC SYSTEM - COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Step 1: Generate Data
        logger.info("STEP 1/7: Generating synthetic traffic data...")
        from src.data.generator import TrafficDataGenerator
        generator = TrafficDataGenerator()
        generator.generate_all()
        logger.success("✓ Data generation complete\n")
        
        # Step 2: Process Data
        logger.info("STEP 2/7: Processing and engineering features...")
        from src.data.processor import TrafficDataProcessor
        processor = TrafficDataProcessor()
        processor.process_all()
        logger.success("✓ Data processing complete\n")
        
        # Step 3: Train Classification Models
        logger.info("STEP 3/7: Training classification models...")
        from src.models.classification import CongestionClassifier
        classifier = CongestionClassifier()
        classifier.train_all()
        logger.success("✓ Classification training complete\n")
        
        # Step 4: Train Forecasting Models
        logger.info("STEP 4/7: Training forecasting models...")
        from src.models.forecasting import TimeSeriesForecaster
        forecaster = TimeSeriesForecaster()
        forecaster.train_all()
        logger.success("✓ Forecasting training complete\n")
        
        # Step 5: Optimize Signals
        logger.info("STEP 5/7: Optimizing traffic signals...")
        from src.optimization.signal_optimizer import SignalOptimizer
        optimizer = SignalOptimizer()
        optimizer.run_optimization()
        logger.success("✓ Signal optimization complete\n")
        
        # Step 6: Run Simulation
        logger.info("STEP 6/7: Running traffic simulation...")
        from src.optimization.simulator import TrafficSimulator
        simulator = TrafficSimulator()
        simulator.run_simulation()
        logger.success("✓ Simulation complete\n")
        
        # Step 7: Evaluate Models
        logger.info("STEP 7/7: Evaluating models...")
        from src.models.evaluation import ModelEvaluator
        evaluator = ModelEvaluator()
        evaluator.evaluate_all()
        logger.success("✓ Evaluation complete\n")
        
        logger.info("=" * 80)
        logger.success("PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review reports in 'reports/' directory")
        logger.info("  2. Check model performance in 'data/models/'")
        logger.info("  3. Launch dashboard: streamlit run src/visualization/dashboard.py")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
