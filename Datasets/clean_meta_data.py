import os
import zipfile
import pandas as pd
from tqdm import tqdm

input_zip_path = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Meta/movement-range-data-2020-03-01-2020-12-31.zip"
internal_file = "movement-range-data-2020-03-01--2020-12-31.txt"
output_dir = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Meta_Cleaned"
output_file = os.path.join(output_dir, "cleaned_meta_movement_2020.csv")

os.makedirs(output_dir, exist_ok=True)

# Define the columns we want to read to save memory
cols_to_use = [
    'country', 'polygon_name', 'ds', 
    'all_day_bing_tiles_visited_relative_change', 
    'all_day_ratio_single_tile_users'
]

print("Reading and processing data...")
chunks = []
with zipfile.ZipFile(input_zip_path) as z:
    with z.open(internal_file) as f:
        # Use chunking for memory safety
        for chunk in tqdm(pd.read_csv(f, sep='\t', usecols=cols_to_use, chunksize=500000)):
            # Filter for India
            chunk = chunk[chunk['country'] == 'IND'].copy()
            
            if len(chunk) > 0:
                # Rename columns
                chunk = chunk.rename(columns={'polygon_name': 'district', 'ds': 'date'})
                
                # Convert date to datetime
                chunk['date'] = pd.to_datetime(chunk['date'])
                
                # Calculate epi_week (Year-W##)
                iso_cal = chunk['date'].dt.isocalendar()
                chunk['epi_week'] = iso_cal.year.astype(str) + '-W' + iso_cal.week.astype(str).str.zfill(2)
                
                # Drop country as we don't need it in the final output
                chunk = chunk.drop(columns=['country'])
                
                chunks.append(chunk)

print("Concatenating chunks...")
final_df = pd.concat(chunks, ignore_index=True)

print("Sorting by district and date...")
final_df = final_df.sort_values(by=['district', 'date'])

print(f"Saving to {output_file}...")
final_df.to_csv(output_file, index=False)

print("Done! Data shape:", final_df.shape)
print("Columns:", final_df.columns.tolist())
print(final_df.head())
