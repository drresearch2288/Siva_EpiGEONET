"""Tests for statistical methods."""
import pytest
import numpy as np
from epigeonet.evaluation.stats_tests import paired_ttest, bootstrap_ci, bonferroni_correct

def test_paired_ttest():
    proposed = [0.1, 0.12, 0.11, 0.13, 0.1]
    baseline = [0.2, 0.22, 0.21, 0.23, 0.2]
    
    pval = paired_ttest(proposed, baseline)
    assert pval < 0.05
    
    # Identical arrays
    pval_id = paired_ttest(proposed, proposed)
    assert pval_id == 1.0

def test_bootstrap_ci():
    preds = np.random.randn(100)
    labels = preds + 0.1 
    
    def dummy_metric(l, p):
        return np.mean((l - p)**2)
        
    mean_val, lo, hi = bootstrap_ci(dummy_metric, preds, labels, n=100)
    assert lo <= mean_val <= hi
    assert mean_val > 0

def test_bonferroni_correct():
    pvals = [0.01, 0.04, 0.5]
    corrected = bonferroni_correct(pvals)
    assert np.allclose(corrected, [0.03, 0.12, 1.0])
