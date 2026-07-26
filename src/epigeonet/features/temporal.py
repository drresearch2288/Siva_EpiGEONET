"""Temporal feature engineering and windowing for EpiGeoNet."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

def resample_epiweeks(df: pd.DataFrame, date_col: str = 'date', district_col: str = 'district', disease_col: str = 'disease') -> pd.DataFrame:
    """
    Reindex each (district, disease) to a continuous epi-week grid.
    Forward-fill gaps <= 2 weeks, else linear interpolate. Log gap statistics.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    group_cols = [district_col]
    if disease_col in df.columns:
        group_cols.append(disease_col)
        
    resampled_dfs = []
    total_gaps = 0
    total_ffill = 0
    total_interp = 0
    
    for name, group in df.groupby(group_cols):
        group = group.sort_values(date_col)
        group = group.set_index(date_col)
        
        # Resample to weekly starting on Monday
        resampled = group.resample('W-MON').asfreq()
        
        missing_mask = resampled.iloc[:, 0].isna()
        total_gaps += missing_mask.sum()
        
        # Forward fill gaps <= 2
        resampled_ffill = resampled.ffill(limit=2)
        ffill_filled = missing_mask.sum() - resampled_ffill.iloc[:, 0].isna().sum()
        total_ffill += ffill_filled
        
        # Linear interpolate remaining
        # Setting numeric_only=True prevents interpolating string columns
        resampled_interp = resampled_ffill.copy()
        numeric_cols = resampled_interp.select_dtypes(include=[np.number]).columns
        resampled_interp[numeric_cols] = resampled_interp[numeric_cols].interpolate(method='linear')
        
        # We check one column to see how many gaps were filled by interpolate
        if len(numeric_cols) > 0:
            interp_filled = resampled_ffill[numeric_cols[0]].isna().sum() - resampled_interp[numeric_cols[0]].isna().sum()
            total_interp += interp_filled
        
        # Restore grouping columns
        if isinstance(name, tuple):
            resampled_interp[district_col] = name[0]
            if disease_col in df.columns and len(name) > 1:
                resampled_interp[disease_col] = name[1]
        else:
            resampled_interp[district_col] = name
            
        resampled_dfs.append(resampled_interp.reset_index())
        
    logger.info(f"Resampling stats: {total_gaps} total gaps. "
                f"Filled {total_ffill} via ffill (<=2 weeks), "
                f"{total_interp} via linear interpolation.")
                
    result_df = pd.concat(resampled_dfs, ignore_index=True)
    return result_df

def add_seasonal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add week-of-year cyclically encoded (sin/cos) and a monsoon indicator.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    epi_week = df[date_col].dt.isocalendar().week.astype(float)
    
    df['week_sin'] = np.sin(2 * np.pi * epi_week / 52.0)
    df['week_cos'] = np.cos(2 * np.pi * epi_week / 52.0)
    
    df['monsoon'] = ((epi_week >= 22) & (epi_week <= 39)).astype(float)
    
    return df

def make_windows(
    df: pd.DataFrame,
    split_mask: pd.Series,
    feature_cols: List[str],
    target_col: str,
    district_col: str = 'district',
    input_len: int = 12,
    horizons: Tuple[int, ...] = (1, 2, 4),
    save_dir: str = 'data/processed/sequences/'
) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, Any]]:
    """
    Produce per-district sliding windows and save as .npz shards.
    """
    max_horizon = max(horizons)
    window_size = input_len + max_horizon
    
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    splits = split_mask.unique()
    
    windows = {s: {'district_id': [], 'week_t': [], 'X': [], 'y1': [], 'y2': [], 'y4': []} for s in splits}
    
    for district, group in df.groupby(district_col):
        group = group.reset_index(drop=True)
        split_mask_group = split_mask.iloc[group.index].reset_index(drop=True)
        n = len(group)
        
        for i in range(n - window_size + 1):
            input_idx = slice(i, i + input_len)
            
            # Ensure window does not cross split boundaries
            window_splits = split_mask_group.iloc[i : i + window_size]
            if len(window_splits.unique()) > 1:
                continue
                
            split = window_splits.iloc[0]
            
            X_win = group[feature_cols].iloc[input_idx].values
            
            y1 = group[target_col].iloc[i + input_len - 1 + 1] if 1 in horizons else np.nan
            y2 = group[target_col].iloc[i + input_len - 1 + 2] if 2 in horizons else np.nan
            y4 = group[target_col].iloc[i + input_len - 1 + 4] if 4 in horizons else np.nan
            
            week_t = i + input_len - 1
            
            windows[split]['district_id'].append(district)
            windows[split]['week_t'].append(week_t)
            windows[split]['X'].append(X_win)
            windows[split]['y1'].append(y1)
            windows[split]['y2'].append(y2)
            windows[split]['y4'].append(y4)
            
    shapes = {}
    n_samples = {}
    
    for split in splits:
        X_arr = np.array(windows[split]['X'])
        if len(X_arr) == 0:
            continue
            
        y1_arr = np.array(windows[split]['y1'])
        y2_arr = np.array(windows[split]['y2'])
        y4_arr = np.array(windows[split]['y4'])
        district_arr = np.array(windows[split]['district_id'])
        week_t_arr = np.array(windows[split]['week_t'])
        
        shapes[split] = X_arr.shape
        n_samples[split] = len(X_arr)
        
        np.savez_compressed(
            out_dir / f"{split}.npz",
            district_id=district_arr,
            week_t=week_t_arr,
            X=X_arr,
            y1=y1_arr,
            y2=y2_arr,
            y4=y4_arr
        )
        
    manifest = {
        'n_samples': n_samples,
        'feature_order': feature_cols,
        'horizons': list(horizons),
        'input_len': input_len
    }
    
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    return shapes, manifest
