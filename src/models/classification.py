"""
Congestion Classification Models for Smart City Traffic System.
Implements multiple classifiers: Random Forest, XGBoost, and Neural Network.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Tuple
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class CongestionClassifier:
    """Train and evaluate congestion classification models."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.config = get_config()
        self.model_config = self.config.get('models', {})
        self.random_state = self.model_config.get('random_state', 42)
        
        self.models = {}
        self.results = {}
        
        logger.info("Congestion Classifier initialized")
    
    def load_processed_data(self) -> Dict:
        """
        Load processed and split data.
        
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Loading processed data...")
        
        processed_path = self.config.get_path('data_processed')
        
        with open(processed_path / 'data_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        logger.success(f"Loaded splits: Train={len(splits['X_train']):,}, "
                      f"Val={len(splits['X_val']):,}, Test={len(splits['X_test']):,}")
        
        return splits
    
    def train_random_forest(self, X_train, y_train, X_val, y_val) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest...")
        
        rf_config = self.model_config.get('random_forest', {})
        
        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 5),
            n_jobs=rf_config.get('n_jobs', -1),
            random_state=self.random_state,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        logger.success(f"Random Forest trained - Validation Accuracy: {val_acc:.4f}")
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost...")
        
        xgb_config = self.model_config.get('xgboost', {})
        
        model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 300),
            max_depth=xgb_config.get('max_depth', 8),
            learning_rate=xgb_config.get('learning_rate', 0.05),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # Evaluate on validation set
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        logger.success(f"XGBoost trained - Validation Accuracy: {val_acc:.4f}")
        
        self.models['xgboost'] = model
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val) -> keras.Model:
        """
        Train Neural Network classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        logger.info("Training Neural Network...")
        
        nn_config = self.model_config.get('neural_network', {
            'layers': [128, 64, 32],
            'dropout': 0.3,
            'epochs': 50,
            'batch_size': 64,
            'learning_rate': 0.001
        })
        
        # Build model
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(4, activation='softmax')  # 4 congestion classes
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_config.get('learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=nn_config.get('epochs', 50),
            batch_size=nn_config.get('batch_size', 64),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        logger.success(f"Neural Network trained - Validation Accuracy: {val_acc:.4f}")
        
        self.models['neural_network'] = model
        return model
    
    def evaluate_model(self, model_name: str, model, X_test, y_test) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name} on test set...")
        
        # Predictions
        if model_name == 'neural_network':
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Try to compute AUC (multi-class)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            target_names=['Free Flow', 'Moderate', 'Heavy', 'Severe'],
            output_dict=True
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.success(f"{model_name} Test Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        if auc:
            logger.info(f"  AUC:       {auc:.4f}")
        
        self.results[model_name] = results
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        model = self.models.get(model_name)
        
        if model_name in ['random_forest', 'xgboost']:
            importances = model.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\nTop 10 features for {model_name}:")
            for i, row in feature_imp.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return feature_imp
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return None
    
    def save_models(self):
        """Save trained models."""
        logger.info("Saving models...")
        
        models_path = self.config.get_path('models')
        
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(models_path / f'{name}_model.h5')
            else:
                joblib.dump(model, models_path / f'{name}_model.pkl')
        
        # Save results
        with open(models_path / 'classification_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.success(f"Models saved to {models_path}")
    
    def train_all(self):
        """Train all classification models."""
        logger.info("=" * 60)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 60)
        
        # Load data
        splits = self.load_processed_data()
        X_train, y_train = splits['X_train'], splits['y_train']
        X_val, y_val = splits['X_val'], splits['y_val']
        X_test, y_test = splits['X_test'], splits['y_test']
        
        # Train models
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_neural_network(X_train, y_train, X_val, y_val)
        
        # Evaluate all models
        for name, model in self.models.items():
            self.evaluate_model(name, model, X_test, y_test)
        
        # Feature importance
        with open(self.config.get_path('data_processed') / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        self.get_feature_importance('random_forest', feature_names)
        self.get_feature_importance('xgboost', feature_names)
        
        # Save models
        self.save_models()
        
        logger.success("=" * 60)
        logger.success("Model Training Complete!")
        logger.success("=" * 60)
        
        return self.models, self.results


def main():
    """Main function to train classification models."""
    classifier = CongestionClassifier()
    classifier.train_all()


if __name__ == "__main__":
    main()
