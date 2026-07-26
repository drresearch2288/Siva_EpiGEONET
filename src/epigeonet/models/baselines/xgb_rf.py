"""XGBoost and Random Forest Baselines."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import shap
import joblib
import json

def flatten_features(x_window):
    """
    Flattens a feature window [B, T, N, 24] or [T, N, 24] into a tabular format.
    If [T, N, 24], flattens to [N, T*24].
    """
    if x_window.ndim == 4:
        B, T, N, D = x_window.shape
        # Move N to front for tabular prediction: [B*N, T*D]
        x_reshaped = x_window.transpose(0, 2, 1, 3).reshape(B * N, T * D)
        return x_reshaped
    elif x_window.ndim == 3:
        T, N, D = x_window.shape
        x_reshaped = x_window.transpose(1, 0, 2).reshape(N, T * D)
        return x_reshaped
    return x_window

class XGBBaseline:
    def __init__(self, model_type='xgb', task='reg', kwargs=None):
        self.model_type = model_type
        self.task = task
        if kwargs is None:
            kwargs = {}
        
        if model_type == 'xgb':
            if task == 'reg':
                self.model = xgb.XGBRegressor(**kwargs)
            else:
                self.model = xgb.XGBClassifier(**kwargs)
        elif model_type == 'rf':
            if task == 'reg':
                self.model = RandomForestRegressor(**kwargs)
            else:
                self.model = RandomForestClassifier(**kwargs)
                
    def fit(self, X, y):
        X_flat = flatten_features(X)
        self.model.fit(X_flat, y)
        
    def predict(self, X):
        X_flat = flatten_features(X)
        return self.model.predict(X_flat)
        
    def predict_proba(self, X):
        if self.task != 'cls':
            raise ValueError("predict_proba only available for classification.")
        X_flat = flatten_features(X)
        return self.model.predict_proba(X_flat)
        
    def shap_values(self, X):
        X_flat = flatten_features(X)
        if self.model_type == 'xgb':
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X_flat)
        else:
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X_flat)
            
    def save(self, path):
        joblib.dump(self.model, path)
        
    @classmethod
    def load(cls, path, model_type='xgb', task='reg'):
        obj = cls(model_type=model_type, task=task)
        obj.model = joblib.load(path)
        return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed/features')
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    models_dir = out_dir / 'models'
    preds_dir = out_dir / 'predictions'
    models_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    print("XGB/RF baseline script ready.")

if __name__ == '__main__':
    main()
