"""Statistical tests for EpiGeoNet evaluation."""
import numpy as np
from scipy import stats

def paired_ttest(proposed_scores, baseline_scores):
    """Computes paired t-test between proposed and baseline scores."""
    # Ensure they are arrays
    proposed = np.asarray(proposed_scores)
    baseline = np.asarray(baseline_scores)
    
    # If standard deviations are zero or arrays identical, return 1.0
    if np.allclose(proposed, baseline):
        return 1.0
        
    stat, p_val = stats.ttest_rel(proposed, baseline)
    if np.isnan(p_val):
        return 1.0
    return float(p_val)

def bootstrap_ci(metric_fn, preds, labels, n=1000, ci=95):
    """
    Computes bootstrap confidence interval for a metric function.
    preds and labels are 1D arrays of equal length.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    n_samples = len(preds)
    
    scores = []
    # Set random seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        scores.append(metric_fn(labels[idx], preds[idx]))
        
    scores = np.array(scores)
    mean_val = np.mean(scores)
    alpha = 100 - ci
    lo = np.percentile(scores, alpha / 2)
    hi = np.percentile(scores, 100 - alpha / 2)
    return float(mean_val), float(lo), float(hi)

def bonferroni_correct(pvals):
    """Bonferroni correction for multiple hypothesis testing."""
    n = len(pvals)
    return np.minimum(np.array(pvals) * n, 1.0).tolist()
