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

def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=20, 
                            color='gray', linewidth=2)
    ax.add_patch(arrow)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    
    # Colors
    c_data = '#3498db'
    c_graph = '#e67e22'
    c_spatial = '#9b59b6'
    c_temp = '#2ecc71'
    c_fusion = '#f1c40f'
    c_head = '#e74c3c'
    c_xai = '#34495e'
    
    # Draw blocks
    x0, y0 = draw_box(ax, 0.5, 2, 1.5, 1, "Multi-Source\nData", c_data)
    x1, y1 = draw_box(ax, 2.5, 2, 1.5, 1, "ASCGC\nGraph Const.", c_graph)
    
    x2_s, y2_s = draw_box(ax, 4.5, 3, 1.5, 1, "GATv2\nSpatial Encoder", c_spatial)
    x2_t, y2_t = draw_box(ax, 4.5, 1, 1.5, 1, "TCN\nTemporal Encoder", c_temp)
    
    x3, y3 = draw_box(ax, 6.5, 2, 1.5, 1, "STCA\nFusion Gate", c_fusion, text_color='black')
    
    x4, y4 = draw_box(ax, 8.5, 3, 1.2, 1, "Multi-Task\nHeads", c_head)
    x5, y5 = draw_box(ax, 8.5, 1, 1.2, 1, "XAI-GeoExplain\nBulletin", c_xai)
    
    # Draw arrows
    draw_arrow(ax, x0+0.75, y0, x1-0.75, y1)
    
    draw_arrow(ax, x1+0.75, y1+0.2, x2_s-0.75, y2_s)
    draw_arrow(ax, x1+0.75, y1-0.2, x2_t-0.75, y2_t)
    
    draw_arrow(ax, x2_s+0.75, y2_s, x3-0.75, y3+0.2)
    draw_arrow(ax, x2_t+0.75, y2_t, x3-0.75, y3-0.2)
    
    draw_arrow(ax, x3+0.75, y3+0.2, x4-0.6, y4)
    draw_arrow(ax, x3+0.75, y3-0.2, x5-0.6, y5)
    
    plt.title("EpiGeoNet End-to-End Pipeline", fontsize=16, fontweight='bold', pad=20)
    
    fig.savefig(out_dir / 'fig01_pipeline.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig01_pipeline.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig01_pipeline'}")

if __name__ == '__main__':
    main()
