import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import argparse
from pathlib import Path

def draw_box(ax, x, y, width, height, text, color, text_color='white'):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1,rounding_size=0.1", 
                         ec="none", fc=color)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            color=text_color, fontweight='bold', wrap=True)
    return x + width/2, y + height/2

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=20, 
                            color='gray', linewidth=2)
    ax.add_patch(arrow)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', va='bottom', fontsize=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    
    c_mdl = '#34495e'
    c_attr = '#e74c3c'
    c_bull = '#3498db'
    
    xm, ym = draw_box(ax, 1, 3, 2, 2, "Trained\nEpiGeoNet", c_mdl)
    
    x1, y1 = draw_box(ax, 5, 5, 2, 1, "Spatial\nAttention (GAT)", c_attr)
    x2, y2 = draw_box(ax, 5, 3, 2, 1, "Temporal\nAttention (STCA)", c_attr)
    x3, y3 = draw_box(ax, 5, 1, 2, 1, "Feature\nSHAP Values", c_attr)
    
    xb, yb = draw_box(ax, 9, 3, 2.5, 2, "GeoExplain\nEpidemiological\nBulletin", c_bull)
    
    draw_arrow(ax, xm+1, ym+0.5, x1-1, y1)
    draw_arrow(ax, xm+1, ym, x2-1, y2)
    draw_arrow(ax, xm+1, ym-0.5, x3-1, y3)
    
    draw_arrow(ax, x1+1, y1, xb-1.25, yb+0.5)
    draw_arrow(ax, x2+1, y2, xb-1.25, yb)
    draw_arrow(ax, x3+1, y3, xb-1.25, yb-0.5)
    
    plt.title("XAI-GeoExplain Module Schematic", fontsize=16, fontweight='bold')
    
    fig.savefig(out_dir / 'fig04_xai.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig04_xai.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig04_xai'}")

if __name__ == '__main__':
    main()
