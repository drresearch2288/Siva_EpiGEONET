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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    c_in = '#95a5a6'
    c_sp = '#9b59b6'
    c_te = '#2ecc71'
    c_stca = '#f1c40f'
    
    xi, yi = draw_box(ax, 1, 3, 1.5, 1, "Input\nX_t", c_in)
    
    xs, ys = draw_box(ax, 4, 4.5, 2, 1, "Spatial Stream\n(GATv2)", c_sp)
    xt, yt = draw_box(ax, 4, 1.5, 2, 1, "Temporal Stream\n(TCN)", c_te)
    
    xf, yf = draw_box(ax, 8, 3, 1.5, 1, "STCA\nCross-Attention", c_stca, text_color='black')
    
    draw_arrow(ax, xi+0.75, yi+0.2, xs-1, ys)
    draw_arrow(ax, xi+0.75, yi-0.2, xt-1, yt)
    
    draw_arrow(ax, xs+1, ys, xf-0.75, yf+0.2, "H_S")
    draw_arrow(ax, xt+1, yt, xf-0.75, yf-0.2, "H_T")
    
    plt.title("Hybrid Parallel Streams & STCA Fusion", fontsize=16, fontweight='bold')
    
    fig.savefig(out_dir / 'fig03_hybrid.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig03_hybrid.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig03_hybrid'}")

if __name__ == '__main__':
    main()
