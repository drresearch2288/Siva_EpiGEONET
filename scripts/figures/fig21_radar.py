import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_radar_data():
    # Axes: RMSE(inv), Macro-F1, Alert-F1, Lead Time, Moran(inv), SHAP, Plausibility, Speed
    axes = ['RMSE (inv)', 'Macro-F1', 'Alert-F1', 'Lead Time', "Moran's I (inv)", 'SHAP Fidelity', 'Plausibility', 'Inference Speed']
    
    # Mock normalized data [0, 1], higher is better
    models = {
        'SARIMA': [0.1, 0.2, 0.1, 0.4, 0.0, 0.0, 0.2, 0.8],
        'Prophet': [0.3, 0.3, 0.2, 0.5, 0.1, 0.0, 0.3, 0.9],
        'LSTM': [0.6, 0.6, 0.5, 0.6, 0.2, 0.3, 0.4, 0.7],
        'ST-GCN': [0.7, 0.7, 0.7, 0.7, 0.4, 0.5, 0.5, 0.6],
        'DCRNN': [0.75, 0.75, 0.75, 0.75, 0.5, 0.6, 0.6, 0.5],
        'Transformer': [0.8, 0.8, 0.7, 0.8, 0.3, 0.7, 0.5, 0.6],
        'GAT': [0.85, 0.8, 0.8, 0.8, 0.6, 0.8, 0.7, 0.7],
        'EpiGeoNet': [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.9]
    }
    return axes, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels, data = get_radar_data()
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one polygon per method
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)-1))
    
    color_idx = 0
    for name, values in data.items():
        vals = values + values[:1]
        if name == 'EpiGeoNet':
            ax.plot(angles, vals, color='#e74c3c', linewidth=3, label=name)
            ax.fill(angles, vals, color='#e74c3c', alpha=0.15)
        else:
            ax.plot(angles, vals, color=colors[color_idx], linewidth=1, linestyle='--', label=name)
            color_idx += 1
            
    # Fix axis labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
    
    ax.set_title('Comprehensive 8-Axis Model Evaluation', pad=20, fontsize=16, fontweight='bold')
    
    # Put legend outside
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig21_radar.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig21_radar.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig21_radar'}")

if __name__ == '__main__':
    main()
