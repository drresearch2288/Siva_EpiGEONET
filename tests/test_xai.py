"""Tests for the XAI-GeoExplain Module."""
import pytest
import torch
import numpy as np
from epigeonet.models.epigeonet import EpiGeoNet
from epigeonet.xai.spatial_explain import explain_spatial
from epigeonet.xai.temporal_explain import explain_temporal
from epigeonet.xai.feature_explain import explain_features

@pytest.fixture
def dummy_model_batch():
    """Create a tiny trained-then-frozen model and a synthetic batch."""
    N, B, T, in_dim = 10, 2, 12, 24
    model = EpiGeoNet(n_nodes=N, in_dim=in_dim, explain=True)
    model.eval() # freeze model
    
    batch = {
        'x': torch.randn(B, T, N, in_dim),
        'static': torch.randn(N, 1),
        'w_clim': torch.rand(B, T, N, N),
        'w_case': torch.rand(B, T, N, N),
        'graph_static': {
            'A_geo': torch.eye(N),
            'w_dist': torch.rand(N, N)
        }
    }
    return model, batch

def test_spatial_explain(dummy_model_batch):
    """Smoke-test spatial attention attribution and ASCGC weights."""
    model, batch = dummy_model_batch
    df, fusion = explain_spatial(model, batch, target_node=0, target_week=11)
    
    assert 'source_node' in df.columns
    assert 'attention_weight' in df.columns
    assert len(fusion) == 4
    assert 'geographic' in fusion

def test_temporal_explain(dummy_model_batch):
    """Smoke-test temporal lag-saliency vectors."""
    model, batch = dummy_model_batch
    lag_saliency = explain_temporal(model, batch)
    
    assert lag_saliency.shape == (10, 12) # N, T
    # Attention sums to roughly 1 over time
    assert np.allclose(lag_saliency.sum(axis=1), 1.0, atol=1e-4)

def test_feature_explain(dummy_model_batch, tmp_path):
    """Smoke-test Integrated Gradients and SHAP caching."""
    model, batch = dummy_model_batch
    out_dir = tmp_path / 'predictions'
    
    attr_ig = explain_features(model, batch, out_dir=str(out_dir))
    
    assert attr_ig.shape == batch['x'].shape
    assert (out_dir / 'shap_values.npz').exists()
