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
    
    classes = ['Low', 'Moderate', 'High', 'Severe']
    models = ['Baseline (LSTM)', 'EpiGeoNet']
    
    # Mock data
    acc_base = [0.90, 0.75, 0.60, 0.45]
    acc_epi = [0.95, 0.88, 0.82, 0.78]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(x - width/2, acc_base, width, label='Baseline (LSTM)', color='#95a5a6', edgecolor='black')
    ax.bar(x + width/2, acc_epi, width, label='EpiGeoNet', color='#3498db', edgecolor='black', hatch='//')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Risk Class Accuracy Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig06_class_accuracy.png', dpi=300)
    fig.savefig(out_dir / 'fig06_class_accuracy.pdf')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig06_class_accuracy'}")

if __name__ == '__main__':
    main()
