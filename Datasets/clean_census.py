import os
import glob
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

TARGET_DIR = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Census of India 2011"
OUTPUT_DIR = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Census of India 2011_Cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all village and town amenities files
village_files = glob.glob(os.path.join(TARGET_DIR, "* - Village Amenities*"))
town_files = glob.glob(os.path.join(TARGET_DIR, "* - Town Amenities*"))

data_records = []

# Process Village Files
for vf in village_files:
    try:
        df = pd.read_excel(vf)
        if 'State Name' in df.columns and 'District Name' in df.columns:
            # We need 'Total Geographical Area (in Hectares)' and 'Total Population of Village'
            area_col = [c for c in df.columns if 'Geographical Area' in c and 'Hectares' in c]
            pop_col = [c for c in df.columns if 'Population of Village' in c and 'Total' in c and 'Male' not in c and 'Female' not in c]
            
            if area_col and pop_col:
                area_col = area_col[0]
                pop_col = pop_col[0]
                
                grouped = df.groupby(['State Name', 'District Name']).agg({
                    area_col: 'sum',
                    pop_col: 'sum'
                }).reset_index()
                
                # Convert Hectares to sq km
                grouped['area_sq_km'] = grouped[area_col] / 100.0
                grouped.rename(columns={pop_col: 'total_population'}, inplace=True)
                
                data_records.append(grouped[['State Name', 'District Name', 'total_population', 'area_sq_km']])
    except Exception as e:
        print(f"Error reading {vf}: {e}")

# Process Town Files
for tf in town_files:
    try:
        df = pd.read_excel(tf)
        if 'State Name' in df.columns and 'District Name' in df.columns:
            # We need 'Area (sq. km.)' and 'Total Population of Town'
            area_col = [c for c in df.columns if 'Area' in c and 'sq. km' in c]
            pop_col = [c for c in df.columns if 'Population of Town' in c and 'Total' in c and 'Male' not in c and 'Female' not in c]
            
            if area_col and pop_col:
                area_col = area_col[0]
                pop_col = pop_col[0]
                
                grouped = df.groupby(['State Name', 'District Name']).agg({
                    area_col: 'sum',
                    pop_col: 'sum'
                }).reset_index()
                
                grouped.rename(columns={area_col: 'area_sq_km', pop_col: 'total_population'}, inplace=True)
                
                data_records.append(grouped[['State Name', 'District Name', 'total_population', 'area_sq_km']])
    except Exception as e:
        print(f"Error reading {tf}: {e}")

if data_records:
    combined_df = pd.concat(data_records, ignore_index=True)
    
    # Clean up state/district strings (strip spaces, lowercase for join logic if needed)
    combined_df['state'] = combined_df['State Name'].astype(str).str.strip().str.upper()
    combined_df['district'] = combined_df['District Name'].astype(str).str.strip().str.upper()
    
    # Force convert to numeric to handle any string entries
    combined_df['total_population'] = pd.to_numeric(combined_df['total_population'], errors='coerce')
    combined_df['area_sq_km'] = pd.to_numeric(combined_df['area_sq_km'], errors='coerce')
    
    final_grouped = combined_df.groupby(['state', 'district']).agg({
        'total_population': 'sum',
        'area_sq_km': 'sum'
    }).reset_index()
    
    # Prevent division by zero or strings
    import numpy as np
    final_grouped['area_sq_km'] = final_grouped['area_sq_km'].replace(0, np.nan)
    
    final_grouped['population_density'] = final_grouped['total_population'] / final_grouped['area_sq_km']
    
    output_path = os.path.join(OUTPUT_DIR, "cleaned_census_2011.csv")
    final_grouped.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
else:
    print("No data extracted.")
