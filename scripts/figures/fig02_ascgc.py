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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    c_comp = '#3498db'
    c_fuse = '#e67e22'
    
    x1, y1 = draw_box(ax, 1, 6, 2, 1, "Geographic\n(Queen)", c_comp)
    x2, y2 = draw_box(ax, 1, 4.5, 2, 1, "Distance\nDecay", c_comp)
    x3, y3 = draw_box(ax, 1, 3, 2, 1, "Climate\nSimilarity", c_comp)
    x4, y4 = draw_box(ax, 1, 1.5, 2, 1, "Case\nSynchrony", c_comp)
    
    xf, yf = draw_box(ax, 6, 3.75, 2, 1, "Learned\nSoftmax Fusion", c_fuse)
    
    xa, ya = draw_box(ax, 9, 3.75, 1, 1, "A_t\nGraph", '#2ecc71')
    
    draw_arrow(ax, x1+1, y1, xf-1, yf+0.3, "w_geo (0.42)")
    draw_arrow(ax, x2+1, y2, xf-1, yf+0.1, "w_dist (0.15)")
    draw_arrow(ax, x3+1, y3, xf-1, yf-0.1, "w_clim (0.28)")
    draw_arrow(ax, x4+1, y4, xf-1, yf-0.3, "w_case (0.15)")
    
    draw_arrow(ax, xf+1, yf, xa-0.5, ya)
    
    plt.title("ASCGC: Adaptive Spatiotemporal Graph Construction", fontsize=16, fontweight='bold')
    
    fig.savefig(out_dir / 'fig02_ascgc.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig02_ascgc.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig02_ascgc'}")

if __name__ == '__main__':
    main()
