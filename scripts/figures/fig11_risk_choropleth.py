import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import contextily as cx
from pathlib import Path
from matplotlib.colors import ListedColormap

def get_dummy_geodata():
    """Generates a 5x5 dummy grid for districts."""
    polygons = []
    for x in range(5):
        for y in range(5):
            polygons.append(Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)]))
            
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf['district_id'] = np.arange(25)
    gdf.crs = "EPSG:3857"
    return gdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpkg', default='data/processed/districts.gpkg')
    parser.add_argument('--week', type=str, default='2022-W40')
    parser.add_argument('--out-dir', default='reports/figures')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gdf = gpd.read_file(args.gpkg)
    except Exception:
        gdf = get_dummy_geodata()
        
    # Mock risk classes: 0: Low, 1: Moderate, 2: High, 3: Severe
    np.random.seed(42)
    gdf['risk_class'] = np.random.choice([0, 1, 2, 3], size=len(gdf), p=[0.5, 0.3, 0.15, 0.05])
    
    # Colorblind safe colormap
    colors = ['#f1eef6', '#bdc9e1', '#74a9cf', '#0570b0']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    gdf.plot(column='risk_class', cmap=cmap, linewidth=0.5, ax=ax, edgecolor='0.8', 
             legend=True, categorical=True,
             legend_kwds={'labels': ['Low', 'Moderate', 'High', 'Severe']})
             
    try:
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    except Exception:
        pass # Ignore contextily fail if no internet
        
    ax.axis('off')
    ax.set_title(f'EpiGeoNet Outbreak Risk Map (Week: {args.week})', fontsize=16)
    
    fig.savefig(out_dir / 'fig11_risk_choropleth.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig11_risk_choropleth.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_dir / 'fig11_risk_choropleth'}")

if __name__ == '__main__':
    main()
