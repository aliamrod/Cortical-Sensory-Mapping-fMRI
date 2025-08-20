"""
00_split.py

1. Load master CSV -> DataFrame
2. Normalize diagnosis labels
3. Resolve relative -> absolute paths using base_dir
4. Verify files exist, print full paths (no truncation)
5. Split & export per-condition CSVs + path-only lists
"""

# Import relevant libraries
import os
import pandas as pd 
import glob, re

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

# Load input data
master_csv = "/home/yyang/yang/sensory/sensory_ABIDE.csv"
base_dir = "/var/datasets/fmri_prep/abide_fmri/fmri_prep_rerun"

master_df = pd.read_csv(master_csv)

# Ensure expected columns exist
assert {"subject_id", "diagnosis", "path_surface_left", "path_surface_right"}.issubset(master_df.columns), "Rename columns to match."

# Normalize and strip `diagnosis`
master_df['diagnosis'] = master_df['diagnosis'].apply(lambda x: x.lower() if isinstance(x, str) else x)

master_df = master_df.rename(columns = {
    "path_surface_left": "lh_path",
    "path_surface_right": "rh_path"
})


# Resolve relative -> absolute paths using base_dir
def resolve_path(p, base_dir):
    if p is None:
        return None
    p = os.path.expanduser(os.path.expandvars(str(p).strip()))

def resolve_path(p, base_dir):
    if p is None: 
        return None
    p = os.path.expanduser(os.path.expandvars(str(p).strip()))
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    return os.path.normpath(p)

master_df["lh_path"] = master_df["lh_path"].apply(lambda p: resolve_path(p, base_dir))
master_df["rh_path"] = master_df["rh_path"].apply(lambda p: resolve_path(p, base_dir))

# Verify files exists
master_df["lh_exists"] = master_df["lh_path"].apply(lambda p: os.path.exists(p) if p else False)
master_df["rh_exists"] = master_df["rh_path"].apply(lambda p: os.path.exists(p) if p else False)
missing = master_df[(~master_df["lh_exists"])] | (~master_df["rh_exists"])
if not missing.empty:
    missing.to_csv("missing_paths.csv", index=False)


# Split into per-condition DataFrames
au_df = master_df[master_df["diagnosis"] == "autism"].reset_index(drop=True).copy()
hc_df = master_df[master_df["diagnosis"] == "healthy"].reset_index(drop=True).copy()
print(f"\nSplit: autism={len(au_df)} | control={len(hc_df)}")

# Print full paths clearly 

#"sub-0001/L.func.gii" + base_dir="/data/ABIDE" → "/data/ABIDE/sub-0001/L.func.gii".


# Split and export by `diagnosis`
au_df.to_csv("autism_split.csv", index=False)
hc_df.to_csv("healthy-control_split.csv", index=False)

def write_list(paths, fname):
    with open(fname, "w") as f:
        for p in paths:
            if p: f.write(p + "\n")
write_list(au_df["lh_path"].tolist(), "autism_lh_paths.txt")
write_list(au_df["rh_path"].tolist(), "autism_rh_paths.txt")
write_list(hc_df["lh_path"].tolist(), "healthy-control_lh_paths.txt")
write_list(hc_df["rh_path"].tolist(), "healthy-control_rh_paths.txt")

print("\nFile summary:  ")
print(master_df[["diagnosis", "lh_exists", "rh_exists"]].value_counts())
if not missing.empty: 
    print(f"\nMissing paths logged to missing_paths.csv (n={len(missing)})")
