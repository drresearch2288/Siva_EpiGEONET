"""Evaluation metrics for EpiGeoNet and baselines."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    roc_curve
)
import time
import torch

# -----------------
# 5.1 Forecasting
# -----------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def mase(y_true, y_pred, y_naive):
    """
    Mean Absolute Scaled Error.
    y_naive represents the predictions of a seasonal naive model on the test set.
    """
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_true, y_naive)
    if mae_naive == 0:
        return np.inf
    return mae_model / mae_naive

# -----------------
# 5.2 Classification
# -----------------
def eval_classification(y_true, y_pred, y_prob=None):
    res = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    if y_prob is not None:
        try:
            res['auc_roc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            res['auc_roc_ovr'] = np.nan
    return res

# -----------------
# 5.3 Early Warning
# -----------------
def detection_rate_at_far(y_true, y_prob, far_target=0.1):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= far_target)[0][-1]
    return tpr[idx]

def average_lead_time(y_true, y_pred, timestamps):
    """
    Computes average lead time between a predicted alert and an actual surge.
    y_true: binary surge indicator
    y_pred: binary alert indicator
    timestamps: temporal index
    """
    # Simplified lead time: time from first alert in a window to first surge
    # For a robust implementation, we group by outbreak events.
    # In this mock implementation for testing, we just compute distance of nearest matches.
    true_idx = np.where(y_true == 1)[0]
    pred_idx = np.where(y_pred == 1)[0]
    
    if len(true_idx) == 0 or len(pred_idx) == 0:
        return 0.0
        
    lead_times = []
    for t in true_idx:
        # Find first pred before or at t
        valid_preds = pred_idx[pred_idx <= t]
        if len(valid_preds) > 0:
            lead = t - valid_preds[-1]
            lead_times.append(lead)
            
    if not lead_times:
        return 0.0
    return np.mean(lead_times)

def eval_early_warning(y_true, y_pred, y_prob, timestamps=None):
    return {
        'alert_f1': f1_score(y_true, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_true, y_prob),
        'detection_rate_10far': detection_rate_at_far(y_true, y_prob, 0.1),
        'avg_lead_time': average_lead_time(y_true, y_pred, timestamps) if timestamps is not None else 0.0
    }

# -----------------
# 5.5 Explainability
# -----------------
def shap_fidelity(perf_original, perf_masked):
    """Performance drop when top-k SHAP features are masked."""
    return perf_original - perf_masked

def attribution_stability(top_k_similar, top_k_dissimilar):
    """Jaccard similarity of top-K SHAP features."""
    set1 = set(top_k_similar)
    set2 = set(top_k_dissimilar)
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def attention_sparsity(attention_weights, threshold=1e-3):
    """Effective # of non-trivial attentions."""
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    return np.mean(attention_weights > threshold)

# -----------------
# 5.6 Efficiency
# -----------------
def efficiency_metrics(train_report):
    return {
        'train_wall_clock': train_report.get('wall_clock', 0.0),
        'peak_memory_mb': train_report.get('peak_memory_mb', 0.0),
        'model_size_mb': train_report.get('model_size_mb', 0.0)
    }

def measure_inference_latency(model, dummy_batch, num_runs=10):
    """Measures average inference latency in ms."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            model(dummy_batch)
            times.append(time.time() - start)
    return np.mean(times) * 1000 # ms
