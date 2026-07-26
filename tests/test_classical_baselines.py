"""Tests for classical baselines B1 & B2."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from epigeonet.models.baselines.sarima import fit_predict_sarima
from epigeonet.models.baselines.prophet_model import fit_predict_prophet

@pytest.fixture
def synthetic_series():
    # 2 years of weekly data
    dates = pd.date_range(start='2018-01-01', periods=104, freq='W')
    # Seasonal sine wave + noise
    cases = 50 + 40 * np.sin(2 * np.pi * np.arange(104) / 52) + np.random.normal(0, 5, 104)
    return pd.Series(cases, index=dates)
    
@pytest.fixture
def synthetic_df(synthetic_series):
    df = pd.DataFrame({
        'ds': synthetic_series.index,
        'y': synthetic_series.values,
        'weekly_precipitation': np.random.uniform(0, 100, 104),
        'weekly_mean_temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(104) / 52)
    })
    return df

def test_sarima_baseline(synthetic_series):
    preds = fit_predict_sarima(synthetic_series, horizons=[1, 2, 4])
    assert 'h_1' in preds
    assert 'h_2' in preds
    assert 'h_4' in preds
    assert all(v >= 0 for v in preds.values())

def test_sarima_fallback():
    # Too short series should trigger fallback
    short_series = pd.Series([10.0, 15.0])
    preds = fit_predict_sarima(short_series, horizons=[1])
    assert preds['h_1'] == 15.0 # fallback to last value since < 52 weeks

def test_prophet_baseline(synthetic_df):
    preds = fit_predict_prophet(synthetic_df, horizons=[1, 2, 4])
    assert 'h_1' in preds
    assert 'h_2' in preds
    assert 'h_4' in preds
    assert all(v >= 0 for v in preds.values())

def test_prophet_fallback():
    short_df = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=2, freq='W'), 'y': [10.0, 15.0]})
    # Prophet requires at least 2 non-NaN rows to run, but if it fails it should fallback
    # Let's make it fail by giving it 1 row.
    very_short_df = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=1, freq='W'), 'y': [10.0]})
    preds = fit_predict_prophet(very_short_df, horizons=[1])
    assert preds['h_1'] == 10.0
