"""Tests for splits and leakage prevention."""
import pytest
import pandas as pd
import json
from epigeonet.features.splits import make_split_masks, recompute_scalers_per_split, write_split_manifest

def test_make_split_masks():
    """Assert disjoint year ranges and no overlap."""
    dates = pd.date_range("2009-01-01", "2022-12-31", freq='W-MON')
    df = pd.DataFrame({'date': dates})
    
    mask = make_split_masks(df)
    
    train_dates = df.loc[mask == 'train', 'date'].dt.year
    val_dates = df.loc[mask == 'val', 'date'].dt.year
    test_dates = df.loc[mask == 'test', 'date'].dt.year
    
    assert train_dates.min() >= 2009 and train_dates.max() <= 2018
    assert val_dates.min() == 2019 and val_dates.max() == 2019
    assert test_dates.min() >= 2020 and test_dates.max() <= 2022
    
    assert len(set(train_dates).intersection(set(val_dates))) == 0
    assert len(set(val_dates).intersection(set(test_dates))) == 0

def test_recompute_scalers_per_split(tmp_path):
    """Test scaler presence validation."""
    s_path = tmp_path / "scalers.joblib"
    t_path = tmp_path / "thresholds.json"
    
    with pytest.raises(FileNotFoundError):
        recompute_scalers_per_split(str(s_path), str(t_path))
        
    s_path.touch()
    t_path.touch()
    
    assert recompute_scalers_per_split(str(s_path), str(t_path))

def test_write_split_manifest(tmp_path):
    """Test leakage report generation."""
    dates = pd.date_range("2018-12-01", "2020-01-31", freq='W-MON')
    df = pd.DataFrame({'date': dates})
    mask = make_split_masks(df)
    
    out_path = tmp_path / "splits.json"
    manifest = write_split_manifest(df, mask, out_path=str(out_path))
    
    assert manifest['leakage_report']['status'] == 'PASS'
    assert not manifest['leakage_report']['train_val_overlap']
    
    with open(out_path, 'r') as f:
        data = json.load(f)
        assert 'train' in data
        assert 'val' in data
        assert 'test' in data
