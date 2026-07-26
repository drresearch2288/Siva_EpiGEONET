"""Tests for label generation."""
import pytest
import numpy as np
import pandas as pd
import json
from epigeonet.features.labels import regression_labels, risk_class_labels, onset_alert_labels

def test_regression_labels():
    """Test t+1, t+2, t+4 shifting."""
    df = pd.DataFrame({
        'district': ['D1']*5,
        'cases': [10, 20, 30, 40, 50]
    })
    res = regression_labels(df, horizons=(1, 2))
    assert res['cases_t+1'].iloc[0] == 20
    assert res['cases_t+2'].iloc[0] == 30
    assert pd.isna(res['cases_t+1'].iloc[4])

def test_risk_class_labels(tmp_path):
    """Test percentiles and classification rules."""
    df = pd.DataFrame({
        'district': ['D1']*10,
        'cases': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    train_mask = pd.Series([True]*5 + [False]*5)
    
    thresh_path = tmp_path / "thresholds.json"
    res = risk_class_labels(df, train_mask, thresholds_path=str(thresh_path), horizons=(1,))
    
    with open(thresh_path, 'r') as f:
        thresh = json.load(f)
        
    assert thresh['D1']['P50'] == 3.0
    assert thresh['D1']['P75'] == 4.0
    assert thresh['D1']['P90'] == 4.6
    
    assert res['risk_t+1'].iloc[0] == 0  # val=2 <= 3
    assert res['risk_t+1'].iloc[2] == 1  # val=4 <= 4
    assert res['risk_t+1'].iloc[4] == 3  # val=6 > 4.6

def test_onset_alert_labels():
    """Test 1.5 std trailing heuristic."""
    df = pd.DataFrame({
        'district': ['D1']*10,
        'cases': [10, 10, 10, 10, 10, 10, 10, 10, 10, 20]
    })
    
    res = onset_alert_labels(df, horizons=(1,))
    
    # At index 8 (9th row), t+1 cases = 20.
    # Trailing 8-week mean for index 8 = 10, std = 0. Threshold = 10.
    # 20 > 10, so alert_t+1 should be 1.0 at index 8.
    assert res['alert_t+1'].iloc[8] == 1.0
    
    # At index 7, t+1 cases = 10.
    # Threshold = 10. 10 is not > 10. Alert = 0.
    assert res['alert_t+1'].iloc[7] == 0.0
