"""Tests for feature engineering."""
import pytest
import numpy as np
import pandas as pd
import joblib
from epigeonet.features.feature_engineering import (
    add_climate_lags, add_case_history, add_static_density, normalise, assemble_feature_matrix
)

def test_add_climate_lags():
    """Test 1/2/3/4 week lags."""
    df = pd.DataFrame({
        'district': ['D1']*5,
        'temperature': [10, 11, 12, 13, 14],
        'precipitation': [0, 0, 0, 0, 0],
        'lai': [1, 1, 1, 1, 1]
    })
    lagged = add_climate_lags(df)
    assert lagged['temperature_lag1'].iloc[4] == 13
    assert lagged['temperature_lag2'].iloc[4] == 12
    assert 'temperature_lag4' in lagged.columns

def test_add_case_history():
    """Test moving averages and rate of change."""
    df = pd.DataFrame({
        'district': ['D1']*5,
        'cases': [2, 4, 6, 8, 10]
    })
    hist = add_case_history(df)
    assert hist['cases_ma4'].iloc[3] == 5.0
    assert hist['cases_ma4'].iloc[4] == 7.0
    assert hist['cases_roc4'].iloc[4] == 2.0

def test_normalise_train_leakage(tmp_path):
    """Test scaler stats are fit on TRAIN ONLY."""
    df = pd.DataFrame({
        'district': ['D1', 'D1', 'D1', 'D1'],
        'feat1': [10.0, 20.0, 100.0, 200.0],
        'cases': [1.0, 1.0, 1.0, 1.0]
    })
    train_mask = pd.Series([True, True, False, False])
    
    scaler_path = tmp_path / "scalers.joblib"
    norm_df = normalise(
        df, train_mask, continuous_cols=['feat1'], log1p_cols=['cases'],
        scaler_path=str(scaler_path)
    )
    
    stats = joblib.load(scaler_path)
    assert 'D1' in stats
    assert stats['D1']['feat1']['mean'] == 15.0
    
    assert norm_df['feat1'].iloc[2] > 10.0

def test_assemble_feature_matrix(tmp_path):
    """Test feature order and 24-D vector creation."""
    df = pd.DataFrame({'district': ['D1']})
    df, order = assemble_feature_matrix(df, out_path=str(tmp_path / "order.json"))
    
    assert len(order) == 24
    assert 'mobility' in order
    assert df['mobility'].iloc[0] == 0.0
    
    for feat in order:
        assert feat in df.columns
