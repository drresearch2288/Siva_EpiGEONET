"""Tests for bulletin and plausibility modules."""
import pytest
import os
from pathlib import Path
import yaml
from epigeonet.xai.bulletin import generate_bulletin
from epigeonet.xai.plausibility import climatic_plausibility_score

@pytest.fixture
def mock_yaml(tmp_path):
    yaml_path = tmp_path / "climate_thresholds.yaml"
    data = {
        'temperature': {'min': 20.0, 'max': 30.0},
        'humidity': {'min': 80.0, 'max': 100.0}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return str(yaml_path)

def test_generate_bulletin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    spatial_expl = {'top_component': 'case synchrony', 'influential_neighbors': 2}
    temporal_expl = {'top_lag_week': 3}
    feature_expl = [{'name': 'rainfall', 'value': 150.0, 'importance': 0.9}]
    
    bulletin = generate_bulletin(
        district='D1',
        week=12,
        prediction='HIGH',
        spatial_expl=spatial_expl,
        temporal_expl=temporal_expl,
        feature_expl=feature_expl
    )
    
    expected = "Risk raised to HIGH, driven primarily by a 3-week rainfall anomaly and elevated case synchrony with 2 adjoining districts."
    assert bulletin == expected
    
    # Check if file was saved
    assert (tmp_path / 'reports' / 'bulletins' / 'bulletin_D1_w12.txt').exists()

def test_climatic_plausibility_score(mock_yaml):
    feature_expl = [
        {'name': 'temperature', 'value': 25.0, 'importance': 0.8}, # plausible
        {'name': 'temperature', 'value': 35.0, 'importance': 0.9}, # implausible
        {'name': 'humidity', 'value': 85.0, 'importance': 0.7},    # plausible
        {'name': 'rainfall', 'value': 100.0, 'importance': 0.5}    # not in yaml, ignored
    ]
    
    score = climatic_plausibility_score(feature_expl, mock_yaml)
    
    assert 0 <= score <= 100
    assert abs(score - 66.666) < 0.1
