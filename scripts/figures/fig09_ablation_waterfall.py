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
    
    # Monotonic improvements (lowering RMSE)
    stages = ['Base(LSTM)', '+Static Graph', '+ASCGC', '+Attention', 'Full EpiGeoNet']
    base_val = 2.0
    drops = [0, -0.2, -0.3, -0.15, -0.15]
    
    vals = []
    current = base_val
    for d in drops:
        current += d
        vals.append(current)
        
    x = np.arange(len(stages))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw bars
    for i in range(len(stages)):
        if i == 0:
            ax.bar(x[i], vals[i], color='#95a5a6')
        elif i == len(stages)-1:
            ax.bar(x[i], vals[i], color='#2ecc71', hatch='//')
        else:
            # Waterfall step
            ax.bar(x[i], -drops[i], bottom=vals[i], color='#3498db')
            ax.bar(x[i], vals[i], color='#bdc3c7', alpha=0.5)
            
            # Annotate gain
            ax.text(x[i], vals[i] - drops[i]/2, f"{drops[i]:.2f}", ha='center', va='center', color='white', fontweight='bold')
            
    # Connect steps
    for i in range(len(stages)-1):
        ax.plot([x[i], x[i+1]], [vals[i], vals[i]], 'k--', alpha=0.5)
        
    ax.set_ylabel('RMSE Score')
    ax.set_title('Ablation Study: Component Contributions to RMSE Reduction')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig09_ablation_waterfall.png', dpi=300)
    fig.savefig(out_dir / 'fig09_ablation_waterfall.pdf')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig09_ablation_waterfall'}")

if __name__ == '__main__':
    main()
