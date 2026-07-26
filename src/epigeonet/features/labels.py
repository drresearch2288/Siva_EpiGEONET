"""Multi-task label generation for EpiGeoNet."""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def regression_labels(df: pd.DataFrame, group_col: str = 'district', case_col: str = 'cases', horizons: Tuple[int, ...] = (1, 2, 4)) -> pd.DataFrame:
    """Generate regression labels (total cases) at future horizons t+1, t+2, t+4."""
    df = df.copy()
    for h in horizons:
        df[f'cases_t+{h}'] = df.groupby(group_col)[case_col].shift(-h)
    return df


def risk_class_labels(df: pd.DataFrame, train_mask: pd.Series, group_col: str = 'district', case_col: str = 'cases',
                      thresholds_path: str = "data/processed/risk_thresholds.json",
                      horizons: Tuple[int, ...] = (1, 2, 4)) -> pd.DataFrame:
    """
    Generate 4-class risk labels (Low/Moderate/High/Severe) based on district-specific
    historical percentiles (P50, P75, P90). Thresholds are computed on TRAIN ONLY.
    """
    df = df.copy()
    
    train_df = df[train_mask]
    thresholds = {}
    
    for district, group in train_df.groupby(group_col):
        vals = group[case_col].dropna()
        if len(vals) > 0:
            p50, p75, p90 = np.percentile(vals, [50, 75, 90])
        else:
            p50, p75, p90 = 0.0, 0.0, 0.0
            
        thresholds[str(district)] = {
            'P50': float(p50),
            'P75': float(p75),
            'P90': float(p90)
        }
        
    out_path = Path(thresholds_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
        
    def map_to_class(val, dist_thresh):
        if pd.isna(val): return np.nan
        if val <= dist_thresh['P50']: return 0
        elif val <= dist_thresh['P75']: return 1
        elif val <= dist_thresh['P90']: return 2
        else: return 3
        
    for h in horizons:
        future_col = f'cases_t+{h}'
        risk_col = f'risk_t+{h}'
        
        if future_col not in df.columns:
            df[future_col] = df.groupby(group_col)[case_col].shift(-h)
            
        def apply_risk(row):
            dist = str(row[group_col])
            val = row[future_col]
            if dist in thresholds:
                return map_to_class(val, thresholds[dist])
            return np.nan
            
        df[risk_col] = df.apply(apply_risk, axis=1)
        
    return df


def onset_alert_labels(df: pd.DataFrame, group_col: str = 'district', case_col: str = 'cases',
                       horizons: Tuple[int, ...] = (1, 2, 4)) -> pd.DataFrame:
    """
    Generate binary onset alert labels: 1 if forecast-week cases > trailing 8-week mean + 1.5 std.
    """
    df = df.copy()
    
    ma8 = df.groupby(group_col)[case_col].transform(lambda x: x.rolling(8, min_periods=1).mean())
    std8 = df.groupby(group_col)[case_col].transform(lambda x: x.rolling(8, min_periods=1).std().fillna(0))
    
    threshold = ma8 + 1.5 * std8
    
    for h in horizons:
        future_col = f'cases_t+{h}'
        alert_col = f'alert_t+{h}'
        
        if future_col not in df.columns:
            df[future_col] = df.groupby(group_col)[case_col].shift(-h)
            
        df[alert_col] = (df[future_col] > threshold).astype(float)
        df.loc[df[future_col].isna(), alert_col] = np.nan
        
        n_alerts = (df[alert_col] == 1).sum()
        total = df[alert_col].notna().sum()
        if total > 0:
            logger.info(f"Horizon {h}: {n_alerts} alerts out of {total} valid samples ({n_alerts/total*100:.2f}% positive class)")
        
    return df


def save_labels(df: pd.DataFrame, split_mask: pd.Series, group_col: str = 'district',
                horizons: Tuple[int, ...] = (1, 2, 4), out_dir: str = "data/processed/labels/") -> None:
    """
    Save labels to data/processed/labels/{split}.npz
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    splits = split_mask.dropna().unique()
    
    for split in splits:
        split_df = df[split_mask == split]
        if len(split_df) == 0: continue
        
        ids = split_df[group_col].values
        save_dict = {'district_id': ids}
        
        for h in horizons:
            if f'cases_t+{h}' in split_df.columns:
                save_dict[f'cases_t+{h}'] = split_df[f'cases_t+{h}'].values
            if f'risk_t+{h}' in split_df.columns:
                save_dict[f'risk_t+{h}'] = split_df[f'risk_t+{h}'].values
            if f'alert_t+{h}' in split_df.columns:
                save_dict[f'alert_t+{h}'] = split_df[f'alert_t+{h}'].values
            
        np.savez_compressed(path / f"{split}.npz", **save_dict)
