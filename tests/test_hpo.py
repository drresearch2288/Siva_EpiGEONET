"""Tests for HPO."""
import pytest
from epigeonet.training.hpo import run_hpo
from pathlib import Path

def test_hpo_runs():
    study = run_hpo(n_trials=2, study_name="test-hpo")
    assert len(study.trials) == 2
    
    best_config_path = Path('config/model_best.yaml')
    assert best_config_path.exists()
