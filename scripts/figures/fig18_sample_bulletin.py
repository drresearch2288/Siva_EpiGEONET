import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 7))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))
    ax2.axis('off')
    
    # 1. Forecast Chart
    weeks = np.arange(10)
    history = np.random.poisson(20, 7)
    forecast = np.random.poisson(50, 3)
    
    ax1.plot(weeks[:7], history, label='Historical', color='#34495e', marker='o')
    ax1.plot(weeks[6:], [history[-1]] + list(forecast), label='Forecast', color='#e74c3c', marker='x', linestyle='--')
    ax1.fill_between(weeks[6:], [history[-1]-5] + list(forecast-5), [history[-1]+5] + list(forecast+5), color='#e74c3c', alpha=0.2)
    ax1.axvline(6, color='gray', linestyle=':')
    ax1.set_title('EpiGeoNet Forecast (District A)', fontsize=14)
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Dengue Cases')
    ax1.legend()
    
    # 2. Bulletin Panel
    box = FancyBboxPatch((0, 0.1), 1, 0.8, boxstyle="round,pad=0.05,rounding_size=0.05", 
                         ec="black", fc='#f9f9f9', lw=1.5)
    ax2.add_patch(box)
    
    bulletin_text = (
        "EPIGEONET ALERT BULLETIN\n\n"
        "Location: District A\n"
        "Risk Level: SEVERE\n\n"
        "Forecast:\n"
        "- Surge expected in 2 weeks.\n"
        "- Peak est. 50 cases/wk.\n\n"
        "Key Drivers (SHAP):\n"
        "1. Temp Max (t-4) > 31 C\n"
        "2. Neighboring outbreak\n"
        "   (District B, weight: 0.35)\n"
        "3. High humidity (82%)\n\n"
        "Action: Intensify fogging."
    )
    
    ax2.text(0.05, 0.85, bulletin_text, va='top', ha='left', fontsize=12, family='monospace', wrap=True)
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig18_sample_bulletin.png', dpi=300)
    fig.savefig(out_dir / 'fig18_sample_bulletin.pdf')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig18_sample_bulletin'}")

if __name__ == '__main__':
    main()
