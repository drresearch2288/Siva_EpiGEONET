"""Tests for base graph utilities."""
import os
import json
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.sparse import csr_matrix
from epigeonet.graph.base_graph import (
    build_queen_adjacency,
    centroid_distance_matrix,
    climate_similarity,
    case_synchrony,
    save_static
)


@pytest.fixture
def toy_geometry():
    """Create a 4-district toy geometry (a 2x2 grid)."""
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    p3 = Polygon([(0, -1), (1, -1), (1, 0), (0, 0)])
    p4 = Polygon([(1, -1), (2, -1), (2, 0), (1, 0)])
    
    gdf = gpd.GeoDataFrame(
        {'district': ['D1', 'D2', 'D3', 'D4']},
        geometry=[p1, p2, p3, p4]
    )
    # Generic geographic CRS
    gdf.set_crs(epsg=4326, inplace=True)
    gdf.set_index('district', inplace=True)
    return gdf


def test_build_queen_adjacency(toy_geometry, tmp_path, monkeypatch):
    """Test queen adjacency and node index generation."""
    # Mock the save path so we don't write to the real data folder during tests
    import epigeonet.graph.base_graph
    monkeypatch.setattr(epigeonet.graph.base_graph, 'Path', lambda x: tmp_path / "node_index.json" if "node_index.json" in str(x) else tmp_path / x)
    
    A_geo, node_index = build_queen_adjacency(toy_geometry)
    
    assert isinstance(A_geo, csr_matrix)
    assert np.allclose(A_geo.toarray(), A_geo.toarray().T)
    assert list(node_index.keys()) == ['D1', 'D2', 'D3', 'D4']


def test_centroid_distance_matrix(toy_geometry):
    """Test distance weight matrix."""
    D_ij, w_dist = centroid_distance_matrix(toy_geometry)
    
    assert np.all(w_dist > 0)
    assert np.all(w_dist <= 1.0)
    assert np.allclose(np.diag(w_dist), 1.0)


def test_climate_similarity():
    """Test cosine similarity in [-1, 1]."""
    data = np.random.randn(4, 12)
    df = pd.DataFrame(data)
    
    sim = climate_similarity(df)
    
    assert np.all(sim >= -1.0 - 1e-6)
    assert np.all(sim <= 1.0 + 1e-6)
    assert np.allclose(np.diag(sim), 1.0)


def test_case_synchrony():
    """Test case synchrony (Pearson corr)."""
    data = np.random.randn(4, 8)
    df = pd.DataFrame(data)
    
    corr = case_synchrony(df)
    
    assert np.all(corr >= -1.0 - 1e-6)
    assert np.all(corr <= 1.0 + 1e-6)
    assert np.allclose(np.diag(corr), 1.0)
