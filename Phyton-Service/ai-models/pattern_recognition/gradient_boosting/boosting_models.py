"""
Gradient Boosting Models Implementation
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class BoostingModels:
    """
    Gradient Boosting Models for Classification
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
    def create_model(self, **kwargs):
        """
        Create a gradient boosting model
        
        Args:
            **kwargs: Model parameters
        """
        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(**kwargs)
        elif self.model_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(**kwargs)
        elif self.model_type == 'catboost':
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(**kwargs)
        else:
            # Default to sklearn Gradient Boosting
            self.model = GradientBoostingClassifier(**kwargs)
            
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
        """
        if self.model is None:
            self.create_model()
            
        self.model.fit(X, y, **kwargs)
        self.is_trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            predictions: Model predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions
        
        Args:
            X: Input features
            
        Returns:
            probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.model.predict_proba(X)
        
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        self.model = joblib.load(filepath)
        self.is_trained = True

def create_boosting_model(model_type: str = 'xgboost', **kwargs) -> BoostingModels:
    """
    Create and return a Boosting model
    
    Args:
        model_type: Type of boosting model ('xgboost', 'lightgbm', 'catboost')
        **kwargs: Model parameters
        
    Returns:
        BoostingModels: Initialized boosting model
    """
    model = BoostingModels(model_type=model_type)
    model.create_model(**kwargs)
    
    return model