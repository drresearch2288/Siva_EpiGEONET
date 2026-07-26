import json
import pandas as pd
import re
from thefuzz import process, fuzz
import os

base_dir = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Dataset_Cleaned "
files = {
    "boundaries": base_dir + "/District boundary shapefile_Cleand/cleaned_district_boundaries.geojson",
    "meta": base_dir + "/Meta_Cleaned/cleaned_meta_movement_2020.csv",
    "worldpop": base_dir + "/WorldPop India population _ Cleaned/cleaned_worldpop_2011.csv",
    "epiclim": base_dir + "/Epiclim_Cleaned data/cleaned_epiclim.csv",
    "census": base_dir + "/Census of India 2011_Cleaned/cleaned_census_2011.csv"
}

alias_file = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/alias_dict.json"
with open(alias_file) as f:
    alias_dict = json.load(f)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).upper()
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_alias(district):
    return alias_dict.get(district, district)

# 1. Load reference boundaries
with open(files["boundaries"]) as f:
    data = json.load(f)
    
ref_districts = set()
state_district_map = {}  # for meta to infer state
for feat in data.get("features", []):
    props = feat["properties"]
    s = clean_text(props.get("state"))
    d = clean_text(props.get("district"))
    ref_districts.add((s, d))
    if d not in state_district_map:
        state_district_map[d] = s

print(f"Loaded {len(ref_districts)} master districts from boundaries.")

ref_district_list = list(set([d for s, d in ref_districts]))

def fuzzy_match_district(d, limit=1, score_cutoff=90):
    match = process.extractOne(d, ref_district_list, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
    if match:
        return match[0]
    return d

unmapped_total = set()
results = {}

for name, path in files.items():
    if name == "boundaries": continue
    print(f"\nProcessing {name}...")
    
    df = pd.read_csv(path)
    if "state" not in df.columns:
        df["state"] = ""
        
    df["clean_state"] = df["state"].apply(clean_text)
    df["clean_dist"] = df["district"].apply(clean_text)
    
    if name == "meta":
        # Meta only has district. Infer state from boundaries if possible.
        df["clean_state"] = df["clean_dist"].map(lambda x: state_district_map.get(x, ""))
        
    df["clean_dist"] = df["clean_dist"].apply(apply_alias)
    
    # Try exact match first
    matched = 0
    total = len(df)
    
    for idx, row in df.iterrows():
        s = row["clean_state"]
        d = row["clean_dist"]
        if (s, d) in ref_districts or (name == "meta" and any(ref_d == d for ref_s, ref_d in ref_districts)):
            matched += 1
        else:
            # try fuzzy matching district
            fuzzy_d = fuzzy_match_district(d)
            if (s, fuzzy_d) in ref_districts or (name == "meta" and any(ref_d == fuzzy_d for ref_s, ref_d in ref_districts)):
                matched += 1
            else:
                unmapped_total.add(d)

    coverage = matched / total * 100 if total > 0 else 0
    results[name] = coverage
    print(f"{name}: {coverage:.2f}% matched ({matched}/{total})")

print(f"\nFound {len(unmapped_total)} unmapped unique district names across all datasets.")
unmapped_file = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/unmapped_districts.csv"
pd.DataFrame({"unmapped_district": list(unmapped_total)}).to_csv(unmapped_file, index=False)
print(f"Saved unmapped districts to {unmapped_file}")

# --- Actual Join Logic ---
print("\n--- Performing Spatial Join ---")
# 1. Create a boundary dataframe with centroid_lat and centroid_lon
boundary_rows = []
for feat in data.get("features", []):
    props = feat["properties"]
    s = clean_text(props.get("state"))
    d = clean_text(props.get("district"))
    lat = props.get("centroid_lat")
    lon = props.get("centroid_lon")
    boundary_rows.append({"clean_state": s, "clean_dist": d, "centroid_lat": lat, "centroid_lon": lon})
df_bounds = pd.DataFrame(boundary_rows).drop_duplicates(subset=["clean_state", "clean_dist"])

def load_and_clean(name, path):
    df = pd.read_csv(path)
    if "state" not in df.columns:
        df["state"] = ""
    df["clean_state"] = df["state"].apply(clean_text)
    df["clean_dist"] = df["district"].apply(clean_text)
    if name == "meta":
        df["clean_state"] = df["clean_dist"].map(lambda x: state_district_map.get(x, ""))
    df["clean_dist"] = df["clean_dist"].apply(apply_alias)
    
    # Apply fuzzy matching
    def match(row):
        s = row["clean_state"]
        d = row["clean_dist"]
        if (s, d) in ref_districts or (name == "meta" and any(ref_d == d for ref_s, ref_d in ref_districts)):
            return d
        fuzzy_d = fuzzy_match_district(d)
        if (s, fuzzy_d) in ref_districts or (name == "meta" and any(ref_d == fuzzy_d for ref_s, ref_d in ref_districts)):
            return fuzzy_d
        return None
        
    df["mapped_dist"] = df.apply(match, axis=1)
    # Drop rows that couldn't be mapped
    df = df.dropna(subset=["mapped_dist"])
    df["clean_dist"] = df["mapped_dist"]
    df = df.drop(columns=["mapped_dist"])
    return df

print("Loading cleaned subsets...")
df_meta = load_and_clean("meta", files["meta"])
df_worldpop = load_and_clean("worldpop", files["worldpop"])
df_epiclim = load_and_clean("epiclim", files["epiclim"])
df_census = load_and_clean("census", files["census"])

print("Merging epiclim (target) with boundaries...")
merged = pd.merge(df_epiclim, df_bounds, on=["clean_state", "clean_dist"], how="left")

print("Merging with worldpop...")
merged = pd.merge(merged, df_worldpop.drop(columns=["state", "district"], errors="ignore"), on=["clean_state", "clean_dist"], how="left", suffixes=("", "_worldpop"))

print("Merging with census...")
merged = pd.merge(merged, df_census.drop(columns=["state", "district"], errors="ignore"), on=["clean_state", "clean_dist"], how="left", suffixes=("", "_census"))

print("Merging with meta (on district & epi_week)...")
# For meta, we join on clean_state, clean_dist, and epi_week
merged = pd.merge(merged, df_meta.drop(columns=["state", "district", "date"], errors="ignore"), on=["clean_state", "clean_dist", "epi_week"], how="left", suffixes=("", "_meta"))

output_merge_path = os.path.join(base_dir, "merged_master_dataset.csv")
merged.to_csv(output_merge_path, index=False)
print(f"Successfully joined {len(merged)} records. Saved to {output_merge_path}")

