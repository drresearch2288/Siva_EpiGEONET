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
    
    hardware = ['CPU-Only (M5 Pro)', 'MPS (M5 Pro)', 'Discrete GPU (A100)']
    train_time = [45.2, 10.5, 4.2]
    inf_time = [65.0, 15.0, 8.5]
    
    x = np.arange(len(hardware))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    
    rects1 = ax1.bar(x - width/2, train_time, width, label='Train Time (min)', color='#34495e')
    rects2 = ax2.bar(x + width/2, inf_time, width, label='Inference (ms)', color='#e67e22')
    
    ax1.set_ylabel('Training Time (minutes)', color='#34495e')
    ax2.set_ylabel('Inference Time (ms)', color='#e67e22')
    
    ax1.set_title('Hardware Acceleration Benchmark')
    ax1.set_xticks(x)
    ax1.set_xticklabels(hardware)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig10_hardware_benchmark.png', dpi=300)
    fig.savefig(out_dir / 'fig10_hardware_benchmark.pdf')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig10_hardware_benchmark'}")

if __name__ == '__main__':
    main()
