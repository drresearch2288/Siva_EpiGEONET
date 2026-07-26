"""Feature engineering and normalization for EpiGeoNet."""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger


def add_climate_lags(df: pd.DataFrame, group_col: str = 'district', vars: List[str] = ['temperature', 'precipitation', 'lai']) -> pd.DataFrame:
    """Add 1/2/3/4-week lags for climate variables."""
    df = df.copy()
    for var in vars:
        if var in df.columns:
            for lag in [1, 2, 3, 4]:
                col_name = f"{var}_lag{lag}"
                df[col_name] = df.groupby(group_col)[var].shift(lag)
    return df


def add_case_history(df: pd.DataFrame, group_col: str = 'district', case_col: str = 'cases') -> pd.DataFrame:
    """Add 4-week and 8-week moving averages and their rate-of-change of cases."""
    df = df.copy()
    if case_col not in df.columns:
        return df
        
    df['cases_ma4'] = df.groupby(group_col)[case_col].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df['cases_ma8'] = df.groupby(group_col)[case_col].transform(lambda x: x.rolling(8, min_periods=1).mean())
    
    df['cases_roc4'] = df.groupby(group_col)['cases_ma4'].diff().fillna(0)
    df['cases_roc8'] = df.groupby(group_col)['cases_ma8'].diff().fillna(0)
    
    return df


def add_static_density(df: pd.DataFrame, density_df: pd.DataFrame, group_col: str = 'district') -> pd.DataFrame:
    """Join log1p(population_density) and broadcast across weeks."""
    df = df.copy()
    density_df = density_df.copy()
    
    if 'population_density' in density_df.columns:
        density_df['log_density'] = np.log1p(density_df['population_density'])
        
    df = df.merge(density_df[[group_col, 'log_density']], on=group_col, how='left')
    return df


def normalise(df: pd.DataFrame, train_mask: pd.Series, group_col: str = 'district', 
              continuous_cols: Optional[List[str]] = None,
              log1p_cols: Optional[List[str]] = None,
              scaler_path: str = "data/processed/scalers.joblib") -> pd.DataFrame:
    """
    Apply log1p to specified columns, then per-district z-score for continuous covariates.
    Fit stats ONLY on train_mask.
    """
    df = df.copy()
    if log1p_cols is None:
        log1p_cols = ['cases', 'log_density']
        
    if continuous_cols is None:
        continuous_cols = []
        
    # Apply log1p BEFORE scaling
    for col in log1p_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
            
    # Calculate per-district stats on TRAIN ONLY
    train_df = df[train_mask]
    
    scaler_stats = {}
    for district, group in train_df.groupby(group_col):
        stats = {}
        for col in continuous_cols:
            if col in group.columns:
                stats[col] = {'mean': group[col].mean(), 'std': group[col].std()}
        scaler_stats[district] = stats
        
    out_path = Path(scaler_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_stats, out_path)
    
    def scale_group(group):
        dist = group.name
        res = group.copy()
        dist_stats = scaler_stats.get(dist, {})
        for col in continuous_cols:
            if col in res.columns:
                mean = dist_stats.get(col, {}).get('mean', 0.0)
                std = dist_stats.get(col, {}).get('std', 1.0)
                if pd.isna(mean): mean = 0.0
                if pd.isna(std) or std == 0.0: std = 1.0
                res[col] = (res[col] - mean) / std
        return res
        
    df = df.groupby(group_col, group_keys=False).apply(scale_group)
    return df


def assemble_feature_matrix(df: pd.DataFrame, out_path: str = "data/processed/feature_order.json") -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble the 24-D feature vector.
    """
    df = df.copy()
    
    base_weather = ['temperature', 'precipitation', 'lai']
    lags = [f"{var}_lag{l}" for var in base_weather for l in [1, 2, 3, 4]]
    history = ['cases_ma4', 'cases_ma8', 'cases_roc4', 'cases_roc8']
    density = ['log_density']
    seasonal = ['week_sin', 'week_cos', 'monsoon']
    
    if 'mobility' not in df.columns:
        df['mobility'] = 0.0
    mobility = ['mobility']
    
    feature_order = base_weather + lags + history + density + seasonal + mobility
    
    assert len(feature_order) == 24, f"Feature vector is {len(feature_order)}-D instead of 24-D"
    
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(feature_order, f, indent=2)
        
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0
            
    return df, feature_order
