import geopandas as gpd
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

shp_path = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/District boundary shapefile/gadm/gadm41_IND_2.shp'
out_dir = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/District boundary shapefile_Cleand'
os.makedirs(out_dir, exist_ok=True)

# Load shapefile
print("Loading shapefile...")
gdf = gpd.read_file(shp_path)

# Extract and standardize keys
print("Extracting and standardizing features...")
gdf['state'] = gdf['NAME_1'].astype(str).str.strip().str.upper()
gdf['district'] = gdf['NAME_2'].astype(str).str.strip().str.upper()

# Calculate centroid lat/lon
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Centroids
centroids = gdf.geometry.centroid
gdf['centroid_lon'] = centroids.x
gdf['centroid_lat'] = centroids.y

# Select desired columns
columns_to_keep = ['state', 'district', 'centroid_lat', 'centroid_lon', 'geometry']
cleaned_gdf = gdf[columns_to_keep]

# Save to GeoJSON
out_path = os.path.join(out_dir, 'cleaned_district_boundaries.geojson')
print(f"Saving cleaned boundaries to {out_path}...")
cleaned_gdf.to_file(out_path, driver='GeoJSON')
print("Done.")
