"""Base graph generation utilities for EpiGeoNet."""
import json
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.distance import great_circle
from libpysal.weights import Queen
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def build_queen_adjacency(gdf: gpd.GeoDataFrame) -> Tuple[sparse.csr_matrix, Dict[str, int]]:
    """
    Build Queen adjacency matrix and node index mapping from a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame containing district polygons.
             Assumes the index is the district identifier.
             
    Returns:
        A_geo: Scipy sparse CSR matrix (binary).
        node_index: Dictionary mapping district name to node index.
    """
    districts = gdf.index.tolist()
    node_index = {str(d): i for i, d in enumerate(districts)}
    
    w = Queen.from_dataframe(gdf, use_index=False)
    A_geo = w.sparse
    
    out_path = Path("data/processed/node_index.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(node_index, f)
        
    return A_geo, node_index


def centroid_distance_matrix(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate geodesic distance matrix and normalized distance weights.
    
    Args:
        gdf: GeoDataFrame of districts.
        
    Returns:
        D_ij: Geodesic distance matrix.
        w_dist: Normalized distance weights 1/(1 + D_ij/D_max).
    """
    gdf_wgs84 = gdf.to_crs(epsg=4326) if gdf.crs else gdf
    
    # Suppress UserWarning about centroid for geographic CRS
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centroids = gdf_wgs84.geometry.centroid
        
    coords = [(geom.y, geom.x) for geom in centroids]
    n = len(coords)
    D_ij = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                D_ij[i, j] = great_circle(coords[i], coords[j]).kilometers
                
    D_max = D_ij.max() if D_ij.max() > 0 else 1.0
    w_dist = 1 / (1 + D_ij / D_max)
    
    return D_ij, w_dist


def climate_similarity(cov_window: pd.DataFrame) -> np.ndarray:
    """
    Calculate climate similarity (cosine similarity) of a 4-week covariate window.
    
    Args:
        cov_window: DataFrame with shape (num_districts, features).
                    
    Returns:
        cosine_sim: D x D cosine similarity matrix.
    """
    cosine_sim = cosine_similarity(cov_window.values)
    return cosine_sim


def case_synchrony(case_window: pd.DataFrame) -> np.ndarray:
    """
    Calculate Pearson correlation of a trailing 8-week case series.
    
    Args:
        case_window: DataFrame with shape (num_districts, time_steps).
                     
    Returns:
        corr_matrix: D x D Pearson correlation matrix.
    """
    corr_matrix = np.corrcoef(case_window.values)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def save_static(path: str, A_geo: sparse.csr_matrix, w_dist: np.ndarray, node_index: Dict[str, int]) -> None:
    """
    Save static graph components (A_geo, w_dist, node_index).
    
    Args:
        path: Path to save the .npz file (e.g., 'data/processed/graph_static.npz').
        A_geo: Sparse binary adjacency matrix.
        w_dist: Distance weight matrix.
        node_index: Node index mapping.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    node_index_json = json.dumps(node_index)
    
    np.savez_compressed(
        out_path,
        A_geo_data=A_geo.data,
        A_geo_indices=A_geo.indices,
        A_geo_indptr=A_geo.indptr,
        A_geo_shape=A_geo.shape,
        w_dist=w_dist,
        node_index=np.array([node_index_json])
    )
