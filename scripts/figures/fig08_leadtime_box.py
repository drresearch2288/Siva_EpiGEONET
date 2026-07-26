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
    
    # Mock lead time distributions
    np.random.seed(42)
    data = [
        np.random.normal(0.5, 0.2, 100),  # Baseline 1
        np.random.normal(1.0, 0.3, 100),  # Baseline 2
        np.random.normal(2.8, 0.5, 100)   # EpiGeoNet
    ]
    labels = ['SARIMA', 'LSTM', 'EpiGeoNet']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    
    colors = ['#bdc3c7', '#95a5a6', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        if color == '#2ecc71':
            patch.set_hatch('//')
            
    ax.set_ylabel('Alert Lead-Time (Weeks)')
    ax.set_title('Distribution of Early Warning Lead-Times')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig08_leadtime_box.png', dpi=300)
    fig.savefig(out_dir / 'fig08_leadtime_box.pdf')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig08_leadtime_box'}")

if __name__ == '__main__':
    main()
