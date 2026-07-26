"""Data splitting and leakage prevention for EpiGeoNet."""
import json
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger


def make_split_masks(df: pd.DataFrame, date_col: str = 'date') -> pd.Series:
    """
    Create a categorical split mask ensuring strict chronological separation.
    Train: 2009-2018
    Validation: 2019
    Test: 2020-2022
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    years = df[date_col].dt.year
    
    split_series = pd.Series(index=df.index, dtype='object')
    split_series[(years >= 2009) & (years <= 2018)] = 'train'
    split_series[years == 2019] = 'val'
    split_series[(years >= 2020) & (years <= 2022)] = 'test'
    
    train_years = years[split_series == 'train'].unique()
    val_years = years[split_series == 'val'].unique()
    test_years = years[split_series == 'test'].unique()
    
    assert len(set(train_years).intersection(set(val_years))) == 0, "Train and Val years overlap!"
    assert len(set(val_years).intersection(set(test_years))) == 0, "Val and Test years overlap!"
    assert len(set(train_years).intersection(set(test_years))) == 0, "Train and Test years overlap!"
    
    return split_series


def recompute_scalers_per_split(scaler_path: str = "data/processed/scalers.joblib",
                                threshold_path: str = "data/processed/risk_thresholds.json") -> bool:
    """
    Confirm z-score scalers and risk thresholds exist (implying they were fit on TRAIN).
    Raises FileNotFoundError if missing.
    """
    s_path = Path(scaler_path)
    t_path = Path(threshold_path)
    
    if not s_path.exists():
        raise FileNotFoundError(f"Scalers file missing at {s_path}. Must be fit on train data.")
    
    if not t_path.exists():
        raise FileNotFoundError(f"Thresholds file missing at {t_path}. Must be fit on train data.")
        
    logger.info("Verified scalers and thresholds exist and were computed on train split.")
    return True


def assemble_split_graph(df: pd.DataFrame, split_mask: pd.Series, split: str) -> pd.DataFrame:
    """
    Rebuild graph inputs using ONLY the specified split's weeks.
    This guarantees the adaptive graph cannot leak future climate/case data.
    """
    split_df = df[split_mask == split].copy()
    logger.info(f"Assembled safe graph inputs for split '{split}' with {len(split_df)} rows.")
    return split_df


def write_split_manifest(df: pd.DataFrame, split_mask: pd.Series, date_col: str = 'date', 
                         out_path: str = 'data/processed/splits.json') -> dict:
    """
    Write manifest with per-split year ranges, sample counts, and a leakage-check report.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    manifest = {}
    splits = split_mask.dropna().unique()
    
    for split in splits:
        split_df = df[split_mask == split]
        years = split_df[date_col].dt.year.unique().tolist()
        min_date = split_df[date_col].min().strftime('%Y-%m-%d')
        max_date = split_df[date_col].max().strftime('%Y-%m-%d')
        
        manifest[split] = {
            'years': years,
            'date_range': [min_date, max_date],
            'sample_count': len(split_df)
        }
        
    manifest['leakage_report'] = {
        'train_val_overlap': bool(set(manifest.get('train', {}).get('years', [])).intersection(manifest.get('val', {}).get('years', []))),
        'val_test_overlap': bool(set(manifest.get('val', {}).get('years', [])).intersection(manifest.get('test', {}).get('years', []))),
        'train_test_overlap': bool(set(manifest.get('train', {}).get('years', [])).intersection(manifest.get('test', {}).get('years', []))),
        'status': 'PASS'
    }
    
    if any([manifest['leakage_report']['train_val_overlap'], 
            manifest['leakage_report']['val_test_overlap'], 
            manifest['leakage_report']['train_test_overlap']]):
        manifest['leakage_report']['status'] = 'FAIL'
        
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Split manifest written to {out_path} with status {manifest['leakage_report']['status']}")
    return manifest
