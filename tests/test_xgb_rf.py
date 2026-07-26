"""Tests for XGBoost/RF baselines."""
import pytest
import numpy as np
from epigeonet.models.baselines.xgb_rf import flatten_features, XGBBaseline

def test_flatten_features():
    # [T, N, D]
    x_3d = np.random.randn(12, 6, 24)
    flat_3d = flatten_features(x_3d)
    assert flat_3d.shape == (6, 12 * 24)
    
    # [B, T, N, D]
    x_4d = np.random.randn(2, 12, 6, 24)
    flat_4d = flatten_features(x_4d)
    assert flat_4d.shape == (12, 12 * 24)
    
def test_xgb_reg():
    x = np.random.randn(12, 6, 24)
    y = np.random.randn(6)
    
    model = XGBBaseline(model_type='xgb', task='reg')
    model.fit(x, y)
    
    preds = model.predict(x)
    assert preds.shape == (6,)
    
def test_xgb_cls():
    x = np.random.randn(12, 6, 24)
    y = np.random.randint(0, 4, 6)
    
    model = XGBBaseline(model_type='xgb', task='cls')
    model.fit(x, y)
    
    preds = model.predict(x)
    assert preds.shape == (6,)
    
    probs = model.predict_proba(x)
    assert probs.shape == (6, 4)

def test_shap_values():
    x = np.random.randn(12, 6, 24)
    y = np.random.randn(6)
    
    model = XGBBaseline(model_type='xgb', task='reg')
    model.fit(x, y)
    
    shap_vals = model.shap_values(x)
    assert shap_vals.shape == (6, 12 * 24)
