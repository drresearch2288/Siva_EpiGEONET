import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zones = ['Tropical Wet', 'Tropical Dry', 'Sub-tropical', 'Montane']
    
    # Mock error data per zone
    np.random.seed(42)
    data = [
        np.random.normal(1.2, 0.2, 50),
        np.random.normal(1.1, 0.15, 60),
        np.random.normal(1.3, 0.25, 40),
        np.random.normal(1.0, 0.1, 30)
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(data, tick_labels=zones, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_hatch('//')
        
    ax.set_ylabel('Forecast Error (RMSE)')
    ax.set_title('Model Generalisation Across Agro-Climatic Zones')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig15_zone_generalisation.png', dpi=300)
    fig.savefig(out_dir / 'fig15_zone_generalisation.pdf')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig15_zone_generalisation'}")

if __name__ == '__main__':
    main()
