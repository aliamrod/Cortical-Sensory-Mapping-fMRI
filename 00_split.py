"""
00_split.py

1. Load master CSV -> DataFrame
2. Normalize diagnosis labels
3. Resolve relative -> absolute paths using base_dir
4. Verify files exist, print full paths (no truncation)
5. Split & export per-condition CSVs + path-only lists
"""

import os
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

# ---- Inputs ----
master_csv = "/home/yyang/yang/sensory/sensory_ABIDE.csv"
base_dir   = "/var/datasets/fmri_prep/abide_fmri/fmri_prep_rerun"

# ---- Load ----
master_df = pd.read_csv(master_csv)

# ---- Ensure expected columns ----
required = {"subject_id", "diagnosis", "path_surface_left", "path_surface_right"}
assert required.issubset(master_df.columns), f"Missing columns. Need {required}"

# ---- Rename path columns ----
master_df = master_df.rename(columns={
    "path_surface_left": "lh_path",
    "path_surface_right": "rh_path"
})

# ---- Normalize diagnosis to {'autism','control'} ----
def norm_dx(x):
    x = x.strip().lower() if isinstance(x, str) else x
    if x in {"asd", "aut", "autism"}:
        return "autism"
    if x in {"control", "hc", "td", "healthy", "healthy control", "typical"}:
        return "control"
    return x

master_df["diagnosis"] = master_df["diagnosis"].apply(norm_dx)

# ---- Resolve relative -> absolute paths ----
def resolve_path(p, base):
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    p = os.path.expanduser(os.path.expandvars(str(p).strip()))
    if not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.normpath(p)

master_df["lh_path"] = master_df["lh_path"].apply(lambda p: resolve_path(p, base_dir))
master_df["rh_path"] = master_df["rh_path"].apply(lambda p: resolve_path(p, base_dir))

# ---- Verify files exist ----
master_df["lh_exists"] = master_df["lh_path"].apply(lambda p: os.path.exists(p) if p else False)
master_df["rh_exists"] = master_df["rh_path"].apply(lambda p: os.path.exists(p) if p else False)

missing = master_df[(~master_df["lh_exists"]) | (~master_df["rh_exists"])]
if not missing.empty:
    missing.to_csv("missing_paths.csv", index=False)

# ---- Split ----
au_df = master_df[master_df["diagnosis"] == "autism"].reset_index(drop=True).copy()
hc_df = master_df[master_df["diagnosis"] == "control"].reset_index(drop=True).copy()

print(f"\nSplit: autism={len(au_df)} | control={len(hc_df)}")

# ---- Print full paths by group (no truncation) ----
def print_paths(df, group):
    print(f"\n--- {group.upper()} LH paths ---")
    print(df["lh_path"].to_string(index=False))
    print(f"\n--- {group.upper()} RH paths ---")
    print(df["rh_path"].to_string(index=False))

print_paths(au_df, "autism")
print_paths(hc_df, "control")

# ---- Exports: CSVs with metadata ----
au_df.to_csv("autism_split.csv", index=False)
hc_df.to_csv("control_split.csv", index=False)

# ---- Exports: plain path lists ----
def write_list(paths, fname):
    with open(fname, "w") as f:
        for p in paths:
            if p: f.write(p + "\n")

write_list(au_df["lh_path"].tolist(), "autism_lh_paths.txt")
write_list(au_df["rh_path"].tolist(), "autism_rh_paths.txt")
write_list(hc_df["lh_path"].tolist(), "control_lh_paths.txt")
write_list(hc_df["rh_path"].tolist(), "control_rh_paths.txt")

# ---- Summary ----
print("\nFile existence summary:")
print(master_df[["diagnosis","lh_exists","rh_exists"]].value_counts())
if not missing.empty:
    print(f"\nMissing paths logged to missing_paths.csv (n={len(missing)})")
