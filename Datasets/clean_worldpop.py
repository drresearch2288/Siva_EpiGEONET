import pandas as pd
import geopandas as gpd
import zipfile
import os

zip_path = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/WorldPop India population /ind_pd_2011_1km_UNadj_ASCII_XYZ.zip'
shp_path = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/District boundary shapefile/gadm/gadm41_IND_2.shp'
out_dir = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/WorldPop India population _ Cleaned'
csv_name = 'ind_pd_2011_1km_UNadj_ASCII_XYZ.csv'

os.makedirs(out_dir, exist_ok=True)

print("Loading shapefile...")
districts = gpd.read_file(shp_path)
# Ensure same CRS
districts = districts.to_crs("EPSG:4326")

print("Loading WorldPop CSV...")
with zipfile.ZipFile(zip_path) as z:
    with z.open(csv_name) as f:
        df_pop = pd.read_csv(f)

print(f"Loaded {len(df_pop)} points.")
print("Converting to GeoDataFrame...")
gdf_pop = gpd.GeoDataFrame(
    df_pop, geometry=gpd.points_from_xy(df_pop.X, df_pop.Y), crs="EPSG:4326"
)

del df_pop

print("Performing Spatial Join...")
joined = gpd.sjoin(gdf_pop, districts, how='inner', predicate='intersects')

print("Calculating zonal means...")
zonal_stats = joined.groupby(['NAME_1', 'NAME_2']).agg(
    zonal_mean_pop_density=('Z', 'mean')
).reset_index()

zonal_stats.rename(columns={'NAME_1': 'state', 'NAME_2': 'district'}, inplace=True)
zonal_stats['state'] = zonal_stats['state'].str.strip().str.upper()
zonal_stats['district'] = zonal_stats['district'].str.strip().str.upper()

out_path = os.path.join(out_dir, 'cleaned_worldpop_2011.csv')
zonal_stats.to_csv(out_path, index=False)
print(f"Saved cleaned data to {out_path}")
