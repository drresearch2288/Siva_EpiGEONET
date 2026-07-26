"""Spatial evaluation metrics for EpiGeoNet."""
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from sklearn.metrics import mean_squared_error
try:
    from epigeonet.xai.plausibility import climatic_plausibility as cp
except ImportError:
    # Dummy if not implemented yet
    cp = None

def morans_i_residuals(residuals, weights):
    """
    Computes global Moran's I on model residuals.
    residuals: 1D array of residuals for N districts (e.g. mean residual over time).
    weights: libpysal W object representing the spatial graph.
    Returns: (I, p_value)
    """
    m = Moran(residuals, weights)
    return m.I, m.p_sim

def local_morans(residuals, weights):
    """
    Computes Local Moran's I (LISA) for cluster mapping.
    residuals: 1D array of residuals for N districts.
    weights: libpysal W object.
    Returns: list of cluster labels ('HH', 'LL', 'HL', 'LH', 'ns')
    """
    lm = Moran_Local(residuals, weights)
    
    # Classify clusters based on p-value and quadrants
    sig = lm.p_sim < 0.05
    clusters = []
    for i in range(len(residuals)):
        if not sig[i]:
            clusters.append('ns')
        elif lm.q[i] == 1:
            clusters.append('HH')
        elif lm.q[i] == 2:
            clusters.append('LH')
        elif lm.q[i] == 3:
            clusters.append('LL')
        elif lm.q[i] == 4:
            clusters.append('HL')
        else:
            clusters.append('ns')
    return clusters

def cross_district_rmse_variance(pred, true, district_ids):
    """
    Computes variance of per-district RMSE to measure spatial fairness.
    pred, true: 1D arrays of predictions and targets.
    district_ids: 1D array of district IDs corresponding to pred/true.
    """
    df = pd.DataFrame({'pred': pred, 'true': true, 'district_id': district_ids})
    
    def rmse(group):
        return np.sqrt(mean_squared_error(group['true'], group['pred']))
        
    rmse_per_district = df.groupby('district_id').apply(rmse, include_groups=False)
    return rmse_per_district.var()

def climatic_plausibility(*args, **kwargs):
    """Wrapper for climatic plausibility for Table 3."""
    if cp is not None:
        return cp(*args, **kwargs)
    return 0.0
