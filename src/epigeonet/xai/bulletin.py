"""Generates IDSP-style natural-language bulletins from explanations."""
import argparse
from pathlib import Path

def generate_bulletin(district: str, week: int, prediction: str, 
                      spatial_expl: dict, temporal_expl: dict, feature_expl: list) -> str:
    """
    Generates a structured text bulletin.
    Example: 'Risk raised to HIGH, driven primarily by a 3-week rainfall anomaly and elevated case synchrony with 2 adjoining districts.'
    """
    
    # Extract details
    top_feature = feature_expl[0]['name'] if feature_expl else 'unknown factors'
    lag_weeks = temporal_expl.get('top_lag_week', 1)
    
    spatial_driver = spatial_expl.get('top_component', 'geographic proximity')
    adjoining_count = spatial_expl.get('influential_neighbors', 0)
    
    bulletin = (f"Risk raised to {prediction.upper()}, driven primarily by a "
                f"{lag_weeks}-week {top_feature} anomaly and elevated {spatial_driver} "
                f"with {adjoining_count} adjoining districts.")
                
    # Save report
    out_dir = Path('reports/bulletins')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bulletin_{district}_w{week}.txt"
    
    with open(out_path, 'w') as f:
        f.write(bulletin)
        
    return bulletin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--district', type=str, default='D1')
    args = parser.parse_args()
    print("Bulletin generator ready.")

if __name__ == '__main__':
    main()
