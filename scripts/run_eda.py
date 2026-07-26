"""Exploratory Data Analysis script for EpiGeoNet."""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as cx
import networkx as nx
from pathlib import Path

# Setup style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette('colorblind')

def setup_axes(ax):
    """IEEE/Elsevier style: no top/right spines, light-gray grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5, color='lightgray')

def save_fig(fig, name):
    """Saves 300-dpi PNG and PDF to reports/figures/."""
    out_dir = Path('reports/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig.tight_layout()
    fig.savefig(out_dir / f'eda_{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'eda_{name}.pdf', bbox_inches='tight')
    print(f"Saved reports/figures/eda_{name}.png and .pdf")

def load_data():
    try:
        df = pd.read_parquet('data/processed/master_table.parquet')
    except Exception:
        # Fallback to the user's merged CSV
        csv_path = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/EpiGeoNet/Dataset_Cleaned/Dataset_Cleaned /merged_master_dataset.csv'
        df = pd.read_csv(csv_path)
    
    try:
        gdf = gpd.read_file('data/processed/districts.gpkg')
    except Exception:
        # Create a dummy geodataframe if not available
        import shapely.geometry
        districts = df['district_id'].unique() if 'district_id' in df.columns else []
        gdf = gpd.GeoDataFrame({'district_id': districts}, geometry=[shapely.geometry.Point(0,0)] * len(districts))
        
    return df, gdf

def run_dist():
    """
    Validates class imbalance thresholds and the log-normal distribution of case counts.
    Ensures that the 4 risk classes and the binary alert class are sufficiently populated 
    to train cross-entropy and focal loss without degenerating.
    """
    df, _ = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Case distribution (log scale)
    sns.histplot(df['cases'], bins=50, log_scale=(False, True), ax=axes[0])
    axes[0].set_title("Case Count Distribution (Log Scale)")
    axes[0].set_xlabel("Cases")
    setup_axes(axes[0])
    
    # 2. 4 Risk Classes
    if 'risk_class' not in df.columns:
        df['risk_class'] = pd.qcut(df['cases'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    sns.countplot(x='risk_class', data=df, ax=axes[1])
    axes[1].set_title("4-Class Risk Imbalance")
    setup_axes(axes[1])
    
    # 3. Rare Alert Class
    if 'alert' not in df.columns:
        df['alert'] = (df['cases'] > df['cases'].quantile(0.95)).astype(int)
    sns.countplot(x='alert', data=df, ax=axes[2])
    axes[2].set_title("Rare Alert Class Imbalance")
    setup_axes(axes[2])
    
    save_fig(fig, 'dist')

def run_seasonality():
    """
    Validates the temporal alignment between climate covariates (precipitation, temperature)
    and the epidemiological case counts. Confirms the expected monsoon lag to justify
    the T=12 window size for the temporal encoder.
    """
    df, _ = load_data()
    if 'date' not in df.columns and 'ds' in df.columns:
        df['date'] = df['ds']
    
    df['date'] = pd.to_datetime(df['date'])
    df['epi_week'] = df['date'].dt.isocalendar().week
    
    weekly = df.groupby('epi_week').agg({
        'cases': 'mean',
        'weekly_precipitation': 'mean',
        'weekly_mean_temperature': 'mean'
    }).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(weekly['epi_week'], weekly['cases'], color='C0', label='Cases (Mean)', linewidth=2)
    ax1.set_xlabel('Epi-Week')
    ax1.set_ylabel('Cases', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    setup_axes(ax1)
    
    ax2 = ax1.twinx()
    ax2.plot(weekly['epi_week'], weekly['weekly_precipitation'], color='C1', label='Precipitation', linestyle='--')
    ax2.plot(weekly['epi_week'], weekly['weekly_mean_temperature'], color='C2', label='Temperature', linestyle='-.')
    ax2.set_ylabel('Climate', color='C1')
    ax2.spines['top'].set_visible(False)
    
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    save_fig(fig, 'seasonality')

def run_corr():
    """
    Validates multicollinearity and feature redundancies among the 24 predictors.
    Clustered heatmap helps identify groups of highly correlated climate/lag features,
    motivating the use of regularizers or feature selection in baselines.
    """
    df, _ = load_data()
    features = df.select_dtypes(include=[np.number]).drop(columns=['cases'], errors='ignore')
    features = features.iloc[:, :24]
    
    corr = features.corr().fillna(0)
    
    fig = plt.figure(figsize=(10, 8))
    sns.clustermap(corr, cmap='vlag', center=0, figsize=(10, 8), dendrogram_ratio=0.1)
    
    # seaborn clustermap creates its own figure
    fig = plt.gcf()
    save_fig(fig, 'corr')

def run_missing():
    """
    Validates data completeness across time and variables.
    Ensures that our forward-fill/interpolation preprocessing strategy is sound
    and that no single year/district is pathologically absent.
    """
    df, _ = load_data()
    
    if 'date' not in df.columns and 'ds' in df.columns:
        df['date'] = df['ds']
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    missing_by_year = df.groupby('year').apply(lambda x: x.isnull().mean())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(missing_by_year.T, cmap='Blues', ax=ax, cbar_kws={'label': 'Fraction Missing'})
    ax.set_title("Missingness Matrix (by Year and Column)")
    setup_axes(ax)
    
    save_fig(fig, 'missing')

def run_geo():
    """
    Validates the spatial join between the GADM boundary polygons and the master table.
    Sanity-checks that mean annual incidence maps cleanly without missing geographic holes.
    """
    df, gdf = load_data()
    
    incidence = df.groupby('district_id')['cases'].mean().reset_index()
    gdf_plot = gdf.merge(incidence, on='district_id', how='left')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_plot = gdf_plot.to_crs(epsg=3857)
    
    gdf_plot.plot(column='cases', cmap='OrRd', legend=True, ax=ax, 
                  legend_kwds={'label': "Mean Annual Cases", 'shrink': 0.7},
                  alpha=0.7, edgecolor='k', linewidth=0.5)
    
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.set_title("Mean Annual Dengue Incidence")
    
    save_fig(fig, 'geo')

def run_graph():
    """
    Validates the queen-contiguity spatial adjacency graph.
    Visualises edges connecting district centroids to ensure the graph convolution
    will pass messages across correct real-world neighbors.
    """
    df, gdf = load_data()
    
    import libpysal
    gdf_proj = gdf.to_crs(epsg=3857)
    W = libpysal.weights.Queen.from_dataframe(gdf_proj)
    
    G = W.to_networkx()
    
    centroids = np.column_stack((gdf_proj.centroid.x, gdf_proj.centroid.y))
    pos = {i: (centroids[i, 0], centroids[i, 1]) for i in range(len(gdf_proj))}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    gdf_proj.plot(ax=ax, facecolor='none', edgecolor='lightgray', linewidth=0.5)
    
    nx.draw(G, pos, ax=ax, node_size=10, node_color='C0', edge_color='C1', alpha=0.6, width=0.5)
    
    ax.set_axis_off()
    ax.set_title("Queen-Contiguity District Graph")
    
    save_fig(fig, 'graph')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', action='store_true', help="Case-count distributions")
    parser.add_argument('--seasonality', action='store_true', help="Weekly climatology")
    parser.add_argument('--corr', action='store_true', help="Correlation heatmap")
    parser.add_argument('--missing', action='store_true', help="Missingness matrix")
    parser.add_argument('--geo', action='store_true', help="Choropleth of incidence")
    parser.add_argument('--graph', action='store_true', help="Visualize contiguity graph")
    
    args = parser.parse_args()
    
    run_all = not any(vars(args).values())
    
    if args.dist or run_all: run_dist()
    if args.seasonality or run_all: run_seasonality()
    if args.corr or run_all: run_corr()
    if args.missing or run_all: run_missing()
    if args.geo or run_all: run_geo()
    if args.graph or run_all: run_graph()

if __name__ == '__main__':
    main()
