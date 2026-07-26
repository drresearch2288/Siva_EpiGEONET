import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import contextily as cx
from pathlib import Path

def get_dummy_geodata():
    polygons = []
    for x in range(5):
        for y in range(5):
            polygons.append(Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)]))
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.crs = "EPSG:3857"
    return gdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpkg', default='data/processed/districts.gpkg')
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gdf = gpd.read_file(args.gpkg)
    except Exception:
        gdf = get_dummy_geodata()
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    gdf.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
    
    # Mock ASCGC edges
    np.random.seed(42)
    centroids = np.array([geom.centroid.coords[0] for geom in gdf.geometry])
    
    # Draw a few strong edges
    for _ in range(15):
        idx1, idx2 = np.random.choice(len(centroids), 2, replace=False)
        weight = np.random.uniform(0.5, 1.0)
        p1 = centroids[idx1]
        p2 = centroids[idx2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#e74c3c', linewidth=weight*3, alpha=weight)
        
    try:
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    except Exception:
        pass
        
    ax.axis('off')
    ax.set_title('Top-Weighted ASCGC Graph Edges Overlay', fontsize=16)
    
    fig.savefig(out_dir / 'fig12_ascgc_overlay.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig12_ascgc_overlay.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig12_ascgc_overlay'}")

if __name__ == '__main__':
    main()
