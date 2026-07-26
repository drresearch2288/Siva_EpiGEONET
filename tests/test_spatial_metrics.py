"""Tests for spatial metrics."""
import pytest
import numpy as np
import libpysal
from epigeonet.evaluation.spatial_metrics import (
    morans_i_residuals,
    local_morans,
    cross_district_rmse_variance
)

@pytest.fixture
def dummy_weights():
    # 4x4 regular lattice queen weights
    w = libpysal.weights.lat2W(4, 4, rook=False)
    w.transform = 'r'
    return w

def test_morans_i_random(dummy_weights):
    # Random residuals should have Moran's I close to 0 (or expected value -1/(n-1))
    np.random.seed(42)
    res_random = np.random.randn(16)
    
    I, p = morans_i_residuals(res_random, dummy_weights)
    assert abs(I) < 0.4 # Random should be small

def test_morans_i_spatial_gradient(dummy_weights):
    # Strong spatial gradient should have positive Moran's I
    # A 4x4 grid, values increasing along x and y
    res_gradient = np.array([
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6
    ], dtype=float)
    
    I, p = morans_i_residuals(res_gradient, dummy_weights)
    assert I > 0.3 # Gradient should be highly positive correlated

def test_cross_district_rmse_variance():
    pred = np.array([1, 2, 3, 4, 5, 6])
    true = np.array([1, 2, 5, 4, 10, 6])
    # District 0: pred=[1, 2], true=[1, 2] -> RMSE=0
    # District 1: pred=[3, 4], true=[5, 4] -> MSE=2/2=1 -> RMSE=1
    # District 2: pred=[5, 6], true=[10, 6] -> MSE=25/2=12.5 -> RMSE=3.53
    district_ids = np.array([0, 0, 1, 1, 2, 2])
    
    var = cross_district_rmse_variance(pred, true, district_ids)
    assert var > 0
