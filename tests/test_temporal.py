"""Tests for temporal features."""
import pytest
import numpy as np
import pandas as pd
from epigeonet.features.temporal import resample_epiweeks, add_seasonal_features, make_windows

def test_resample_epiweeks():
    """Test gap filling logic."""
    dates = pd.date_range("2020-01-01", periods=5, freq='W-MON')
    df = pd.DataFrame({
        'date': dates,
        'district': ['D1'] * 5,
        'val': [1.0, np.nan, np.nan, np.nan, 5.0]
    })
    
    df_missing = df.drop([1, 2, 3])
    
    resampled = resample_epiweeks(df_missing, date_col='date', district_col='district')
    
    assert len(resampled) == 5
    # ffill(2) creates 1.0, 1.0, 1.0, NaN, 5.0
    # linear interpolate on index 3 fills with (1.0 + 5.0) / 2 = 3.0
    assert np.allclose(resampled['val'].values, [1.0, 1.0, 1.0, 3.0, 5.0])

def test_add_seasonal_features():
    """Test week-of-year encoding and monsoon indicator."""
    dates = pd.date_range("2020-01-01", periods=52, freq='W-MON')
    df = pd.DataFrame({'date': dates})
    
    feat_df = add_seasonal_features(df, date_col='date')
    
    assert 'week_sin' in feat_df.columns
    assert 'week_cos' in feat_df.columns
    assert 'monsoon' in feat_df.columns
    
    epi_week = feat_df['date'].dt.isocalendar().week
    expected_monsoon = ((epi_week >= 22) & (epi_week <= 39)).astype(float)
    assert np.allclose(feat_df['monsoon'], expected_monsoon)

def test_make_windows(tmp_path):
    """Test window generation and split boundaries."""
    n_weeks = 20
    df = pd.DataFrame({
        'district': ['D1'] * n_weeks,
        'f1': np.arange(n_weeks, dtype=float),
        'target': np.arange(n_weeks, dtype=float) * 2
    })
    
    split_mask = pd.Series(['train'] * 10 + ['val'] * 10)
    
    shapes, manifest = make_windows(
        df, split_mask, feature_cols=['f1'], target_col='target',
        input_len=4, horizons=(1, 2), save_dir=tmp_path
    )
    
    assert shapes['train'] == (5, 4, 1)
    assert shapes['val'] == (5, 4, 1)
    
    npz = np.load(tmp_path / "train.npz")
    X = npz['X']
    y1 = npz['y1']
    y2 = npz['y2']
    
    assert np.allclose(X[0].flatten(), [0, 1, 2, 3])
    # t + 1 horizon is index 4 => 4 * 2 = 8
    assert y1[0] == 8
    # t + 2 horizon is index 5 => 5 * 2 = 10
    assert y2[0] == 10
