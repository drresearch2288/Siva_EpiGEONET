"""Plausibility metrics for XAI explanations."""
import yaml
from pathlib import Path

def load_thresholds(yaml_path='data/external/climate_thresholds.yaml'):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def climatic_plausibility_score(feature_expl: list, thresholds_path: str = 'data/external/climate_thresholds.yaml') -> float:
    """
    Returns the % of top-attributed climate features whose actual values 
    fall within literature vector-breeding ranges.
    
    feature_expl format expected for testing:
    [
        {'name': 'temperature', 'value': 25.5, 'importance': 0.8},
        {'name': 'humidity', 'value': 85.0, 'importance': 0.5}
    ]
    """
    try:
        thresholds = load_thresholds(thresholds_path)
    except FileNotFoundError:
        # Fallback for tests if not provided via mock
        return 0.0
    
    plausible_count = 0
    total_climate_features = 0
    
    for feat in feature_expl:
        name = feat.get('name')
        val = feat.get('value')
        
        if name in thresholds:
            total_climate_features += 1
            t_min = thresholds[name].get('min', float('-inf'))
            t_max = thresholds[name].get('max', float('inf'))
            
            if t_min <= val <= t_max:
                plausible_count += 1
                
    if total_climate_features == 0:
        return 0.0
        
    return (plausible_count / total_climate_features) * 100.0

if __name__ == '__main__':
    print("Plausibility check ready.")
