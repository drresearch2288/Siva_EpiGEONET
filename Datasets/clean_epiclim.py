import zipfile
import pandas as pd
import os

zip_path = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/EpiClim/14580510.zip'
output_dir = '/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Epiclim_Cleaned data'
output_file = os.path.join(output_dir, 'cleaned_epiclim.csv')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Desired columns and their mappings from the original data
column_mapping = {
    'state_ut': 'state',
    'district': 'district',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'year': 'year',
    'week_of_outbreak': 'epi_week',
    'Disease': 'disease_type',
    'Cases': 'cases',
    'Deaths': 'deaths',
    'preci': 'weekly_precipitation',
    'Temp': 'weekly_mean_temperature',
    'LAI': 'LAI'
}

join_keys = ['state', 'district', 'year', 'epi_week']

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open('Final_data.csv') as f:
        print("Loading data...")
        df = pd.read_csv(f)
        
        print("Original columns:", df.columns.tolist())
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Select the desired columns
        features_to_extract = list(column_mapping.values())
        
        # Check if all required features are present
        missing_cols = [col for col in features_to_extract if col not in df.columns]
        if missing_cols:
            print(f"Warning: The following columns are missing and will be skipped: {missing_cols}")
            actual_features = [col for col in features_to_extract if col in df.columns]
        else:
            actual_features = features_to_extract
            
        df_cleaned = df[actual_features].copy()
        
        # We can sort by the join keys if present
        actual_join_keys = [col for col in join_keys if col in df_cleaned.columns]
        if actual_join_keys:
            df_cleaned.sort_values(by=actual_join_keys, inplace=True)

            
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
