"""
Model Evaluation utilities for Smart City Traffic System.
Provides comprehensive evaluation metrics and visualization functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate and compare machine learning models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.config = get_config()
        self.figures_path = self.config.get_path('figures')
        
        logger.info("Model Evaluator initialized")
    
    def load_results(self) -> Tuple:
        """
        Load classification and forecasting results.
        
        Returns:
            Tuple of (classification_results, forecasting_results)
        """
        logger.info("Loading model results...")
        
        models_path = self.config.get_path('models')
        
        # Load classification results
        with open(models_path / 'classification_results.pkl', 'rb') as f:
            class_results = pickle.load(f)
        
        # Load forecasting results
        with open(models_path / 'forecasting_results.pkl', 'rb') as f:
            forecast_results = pickle.load(f)
        
        logger.success("Results loaded successfully")
        
        return class_results, forecast_results
    
    def plot_confusion_matrix(self, results: Dict, model_name: str):
        """
        Plot confusion matrix for a classification model.
        
        Args:
            results: Model results dictionary
            model_name: Name of the model
        """
        cm = results['confusion_matrix']
        labels = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig(self.figures_path / f'confusion_matrix_{model_name}.png', dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix saved for {model_name}")
    
    def plot_classification_comparison(self, class_results: Dict):
        """
        Plot comparison of classification models.
        
        Args:
            class_results: Dictionary with all classification results
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = list(class_results.keys())
        
        data = []
        for metric in metrics:
            for model in models:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': class_results[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model')
        plt.title('Classification Models Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.05)
        plt.legend(title='Model')
        plt.tight_layout()
        
        plt.savefig(self.figures_path / 'classification_comparison.png', dpi=300)
        plt.close()
        
        logger.info("Classification comparison plot saved")
    
    def plot_forecasting_comparison(self, forecast_results: Dict):
        """
        Plot comparison of forecasting models.
        
        Args:
            forecast_results: Dictionary with all forecasting results
        """
        metrics = ['rmse', 'mae', 'mape']
        models = list(forecast_results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            values = [forecast_results[model]['metrics'][metric] for model in models]
            
            axes[idx].bar(models, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Forecasting Models Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(self.figures_path / 'forecasting_comparison.png', dpi=300)
        plt.close()
        
        logger.info("Forecasting comparison plot saved")
    
    def create_performance_report(self, class_results: Dict, forecast_results: Dict) -> str:
        """
        Create comprehensive performance report.
        
        Args:
            class_results: Classification results
            forecast_results: Forecasting results
            
        Returns:
            Report string
        """
        logger.info("Creating performance report...")
        
        report = []
        report.append("=" * 80)
        report.append("SMART CITY TRAFFIC SYSTEM - MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Classification Results
        report.append("CLASSIFICATION MODELS")
        report.append("-" * 80)
        report.append(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        report.append("-" * 80)
        
        for model_name, results in class_results.items():
            report.append(
                f"{model_name:<20} "
                f"{results['accuracy']:<12.4f} "
                f"{results['precision']:<12.4f} "
                f"{results['recall']:<12.4f} "
                f"{results['f1_score']:<12.4f}"
            )
        
        report.append("")
        
        # Forecasting Results
        report.append("FORECASTING MODELS")
        report.append("-" * 80)
        report.append(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'MAPE (%)':<12}")
        report.append("-" * 80)
        
        for model_name, results in forecast_results.items():
            metrics = results['metrics']
            report.append(
                f"{model_name:<20} "
                f"{metrics['rmse']:<12.2f} "
                f"{metrics['mae']:<12.2f} "
                f"{metrics['mape']:<12.2f}"
            )
        
        report.append("")
        report.append("=" * 80)
        
        # Find best models
        best_classifier = max(class_results.items(), key=lambda x: x[1]['accuracy'])
        best_forecaster = min(forecast_results.items(), key=lambda x: x[1]['metrics']['rmse'])
        
        report.append("BEST PERFORMING MODELS")
        report.append("-" * 80)
        report.append(f"Best Classifier: {best_classifier[0]} (Accuracy: {best_classifier[1]['accuracy']:.4f})")
        report.append(f"Best Forecaster: {best_forecaster[0]} (RMSE: {best_forecaster[1]['metrics']['rmse']:.2f})")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        reports_path = self.config.get_path('reports')
        with open(reports_path / 'model_performance_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.success("Performance report created")
        
        return report_text
    
    def evaluate_all(self):
        """Run complete evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 60)
        
        # Load results
        class_results, forecast_results = self.load_results()
        
        # Plot confusion matrices
        for model_name in class_results.keys():
            self.plot_confusion_matrix(class_results[model_name], model_name)
        
        # Comparison plots
        self.plot_classification_comparison(class_results)
        self.plot_forecasting_comparison(forecast_results)
        
        # Performance report
        report = self.create_performance_report(class_results, forecast_results)
        print("\n" + report)
        
        logger.success("=" * 60)
        logger.success("Model Evaluation Complete!")
        logger.success("=" * 60)


def main():
    """Main function for model evaluation."""
    evaluator = ModelEvaluator()
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
