"""Tests for EpiGeoNet evaluation metrics."""
import pytest
import numpy as np
from epigeonet.evaluation.metrics import (
    rmse,
    mape,
    mase,
    eval_classification,
    eval_early_warning,
    shap_fidelity,
    attribution_stability,
    attention_sparsity
)

def test_forecasting_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    y_naive = np.array([10.0, 15.0, 20.0])
    
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((4 + 4 + 0) / 3))
    assert np.isclose(mape(y_true, y_pred), (0.2 + 0.1 + 0.0) / 3)
    
    # MAE model = (2 + 2 + 0)/3 = 4/3
    # MAE naive = (0 + 5 + 10)/3 = 15/3 = 5
    # MASE = (4/3) / 5 = 4/15 = 0.2666...
    assert np.isclose(mase(y_true, y_pred, y_naive), 4/15)

def test_classification_metrics():
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 2, 2, 3, 1, 1])
    
    res = eval_classification(y_true, y_pred)
    assert res['accuracy'] == 4 / 6
    assert 'macro_f1' in res
    assert 'confusion_matrix' in res

def test_early_warning_metrics():
    y_true = np.array([0, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.8, 0.9, 0.2, 0.4])
    timestamps = np.array([1, 2, 3, 4, 5])
    
    res = eval_early_warning(y_true, y_pred, y_prob, timestamps)
    assert 'alert_f1' in res
    assert 'pr_auc' in res
    assert 'detection_rate_10far' in res
    
    # Lead time:
    # true_idx = [2, 4]
    # pred_idx = [1, 2]
    # For t=2 (surge 1): pred at 1, lead=1. pred at 2, lead=0. Last pred before/at 2 is 2 (lead 0)
    # For t=4 (surge 2): preds before/at 4 are 1, 2. Last is 2, lead=2
    # Wait, my logic takes last pred. For t=2, pred<=2 is [1, 2], last is 2 -> lead=0.
    # For t=4, pred<=4 is [1, 2], last is 2 -> lead=2.
    # Mean = 1.0
    assert res['avg_lead_time'] == 1.0

def test_explainability_metrics():
    assert np.isclose(shap_fidelity(0.9, 0.7), 0.2)
    
    sim = [1, 2, 3]
    dissim = [2, 3, 4]
    # Intersection = {2, 3} (2), Union = {1, 2, 3, 4} (4)
    assert attribution_stability(sim, dissim) == 0.5
    
    att = np.array([0.0001, 0.05, 0.1, 0.0005])
    assert attention_sparsity(att, 1e-3) == 0.5
