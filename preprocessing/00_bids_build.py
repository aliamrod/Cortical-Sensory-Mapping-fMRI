# 1. Does the SOURCE file exist?
ls -lh /home/yyang/yang/map_master_fmri/fMRI_All_master_file_V6.csv

# 2. Create destination folder to preferred path
DEST=/home/amahama/PROJECTS/1_sensory/data/manifests
mkdir -p "$DEST"

# 3. Copy here with last date of update in the name (verbose)
cp -v /home/yyang/yang/map_master_fmri/fMRI_All_master_file_V6.csv \
      "$DEST/fMRI_All_master_file_V7_$(date +%F).csv"
ls -lh "$DEST"

# 4. Inspect contents of *.csv
cd /home/amahama/PROJECTS/1_sensory/data/manifests

head -n 10 fMRI_All_master_file_V7_2025-09-08.csv | column -t -s,
head -n 1 fMRI_All_master_file_V7_2025-09-08.csv | tr, ',' '\n' | nl

# Print unique *string values for fMRI pass column (pass/fail)
csvcut -c preprocessing_failed_fmriprep_stable fMRI_All_master_file_V6.csv \
| tail -n +2 | sed 's/\r$//'


csvcut -c preprocessing_failed_fmriprep_stable fMRI_All_master_file_V6.csv \
| tail -n +2 | sed 's/\r$//' | awk '{print tolower($0)}' | sed 's/^[ \t]*//;s/[ \t]*$//' \
| sed 's/^$/<empty>/' | sort | uniq -c | sort -nr
# 46999 ok
# 1543 failed


# 5. fMRI Prep Preprocessing
grep -c "fsaverage5" fMRI_All_master_file_V7_2025-09-08.csv
# 0 count.

grep -c "MNI" fMRI_All_master_file_V7_2025-09-08.csv
# 46,999 count.



# MNI is a 3D voxel template (e.g., MNI152NLin2009cAsym) used for volumetric normalization; this is the default output of fMRIPrep as it often gives one these
# NifTI outputs by default. *However, the sensory-integration method is performed on the cortical surface, vertex-by-vertex, not in MNI volume. So MNI outputs 
# are not sufficient for our beta estimation. 

# FreeSurfer's surface template(s), fsaverage5 specifically, is a downsampled FreeSurfer surface with 10,242 vertices per hemisphere (~20,000k vertices total).
# We require surface time series on a consistent mesh to do vertex-wise regression (i.e., we use these '.func.gii' time series). 

# Exclude the failed subjects first, then proceed to convert the successful runs from MNI Space -> fsaverage5. 
(head -n 1 fMRI_All_master_file_V7_2025-09-09.csv && tail -n +2 fMRI_All_master_file_V7_2025-09-08.csv | grep -Eiv "Failed") > fMRI_master_file_MNI_pass.csv


# Now, we re-run fMRIPrep for only subjects that lack surfaces and ask it to output fsaverage5 surface time-series (*space-fsaverage5_hemi-*.func.gii).
# Those provide accurate, subject-specific vertex data (in order to regress each vertex's time series onto the 3 seed time series (V1/S1/A1); that requires surface time series (fsaverage, .func.gii) and surface atlas (Glasser on fsaverage5 .annot).

# Extract site name(s); if multi-site, will run BIDS check multiple times. BIDS specification does not explicity cover studies with data coming from multiple sites 
# or multiple centers. 
# 1) Treat each site/center as a separate dataset. 
# 2) Combining sites/centers into one dataset: (2a): Collate sites at subject level--> identify which site each subjects comes from you can add a `site` column in the `participants.tsv` file indicating the source site. This solution allows
# ... one to analyze all subjects together in one dataset. One caveat is that subjects from all sites will have to have unique labels. To enforce that and improve readability you can use
# ... a subject label prefix identifying the site (i.e., sub-NUY001, sub-NUY002, sub-NUY003, etc.). 
# OR
# (2b): Use different sessions for different sites. In case of studies such as "Traveling Human Phantom" it is possible to incorporate site within session label (i.e., sub-human1/ses-NUY, sub-human1/ses-MIT, sub-phantom1/ses-NUY, sub-phantom1/ses-MIT, etc.). 

# Inspect sites
head -n 1 fMRI_master_file_MNI_pass.csv | column -t -s,
#   subject_id  session_id  run  age  sex  site  scanner_id  diagnosis  path_fmri  path_fmriprep  preprocessing_failed_fmriprep_stable  uid  uid2
cut -d, -f7 fMRI_master_file_MNI_pass.csv | sort | uniq


cut -d, -f7 fMRI_master_file_MNI_pass.csv | sort | uniq | wc -l
# 234

for site in $(cut -d, -f7 fMRI_master_file_MNI_pass.csv | sort | uniq); do
    grep "$site" fMRI_master_file_MNI_pass.csv > "${site}_subjects.csv"
done


################################## Create BIDS Structure ##################################
"""
'00_bids_build.py':
Convert master CSV manifest of fMRI files into proper BIDS (Brain Imaging Data Structure)-compliant folder structure.
- Reads fMRI_master_file_MNI_pass.csv
- Generates sub-/ses-/func folders
- Symlinks each bold file into the new tree
- Writes minimal JSON sidescars (TaskName)
- Creates a `participants.tsv` with age, sex, diagnosis, site
"""

import csv, os, re, json
from pathlib import Path

MASTER = "/home/amahama/PROJECTS/1_sensory/data/manifests/fMRI_master_file_MNI_pass.csv"
BIDS = "/home/amahama/PROJECTS/1_sensory/bids_dir"
LOG_MISSING = "/home/amahama/PROJECTS/1_sensory/logs/missing_files.txt"

Path(BIDS).mkdir(parents=True, exist_ok=True)
Path(LOG_MISSING).parent.mkdir(parents=True, exist_ok=True)
(open(LOG_MISSING, "w")).write("")  # truncate log

# BIDS metadata
(Path(BIDS) / "dataset_description.json").write_text(
    '{ "Name":"Sensory Mapping BIDS", "BIDSVersion":"1.8.0", "DatasetType":"raw" }\n'
)

# Initialize `participants.tsv`
ptsv = Path(BIDS) / "participants.tsv"
if not ptsv.exists():
    with open(ptsv, "w", newline="") as f:
        f.write("participant_id\tsite\tdiagnosis\tage\tsex\n")

seen = set()  # track which participant rows we already wrote

# --- helpers ---------------------------------------------------------------

def norm_ses(x: str) -> str:
    """Normalize session string to 'ses-<label>'."""
    if not x:
        return "ses-1"  # sensible default if absent
    x = str(x).strip()
    x = re.sub(r"^ses-", "", x, flags=re.IGNORECASE)
    return f"ses-{x}"

def norm_run(x: str) -> str:
    """Normalize run string to 'run-<index>' (no zero-padding assumptions)."""
    if not x:
        return "run-1"
    x = str(x).strip()
    x = re.sub(r"^run-", "", x, flags=re.IGNORECASE)
    return f"run-{x}"

def guess_task(src_path: str, row_task: str | None) -> str:
    """Pick TaskName from CSV column if present; otherwise regex from filename; else 'rest'."""
    if row_task and str(row_task).strip():
        # make sure it's BIDS-safe (alnum/underscore only)
        t = re.sub(r"[^A-Za-z0-9_]", "", str(row_task).strip())
        return t or "rest"
    if src_path:
        m = re.search(r"task-([A-Za-z0-9_]+)", os.path.basename(src_path))
        if m:
            return m.group(1)
    return "rest"

def get_col(row: dict, *candidates, default=None):
    """Robust column accessor: tries multiple names (case-insensitive)."""
    lower_map = {k.lower(): k for k in row.keys()}
    for cand in candidates:
        if cand is None:
            continue
        k = lower_map.get(cand.lower())
        if k is not None:
            return row[k]
    return default

def preprocessing_ok(row: dict) -> bool:
    """Decide whether to keep the row based on 'preprocessing_failed_*' style column."""
    # Try several variants seen in your notes/logs
    flag = get_col(
        row,
        "preprocessing_failed_fmriprep_stable",
        "preprocessing_failed_fmri_prep_stable",
        "preprocessing_failed_fmriprep",
        "preprocessing_failed",
        default=None,
    )
    if flag is None:
        # If there is no column, be permissive and keep the row.
        return True
    return str(flag).strip().lower() == "ok"

def safe_symlink(src: Path, dst: Path):
    """Create/refresh a symlink (idempotent)."""
    if dst.is_symlink() or dst.exists():
        try:
            if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
                return  # already correct
            dst.unlink()
        except FileNotFoundError:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)

# --- main loop -------------------------------------------------------------

missing = []

with open(MASTER, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not preprocessing_ok(row):
            continue

        # Pull fields (robust to name variants)
        subj = get_col(row, "subject_id", "subject", "sub", "participant_id")
        ses = get_col(row, "session_id", "session", "ses")
        run = get_col(row, "run", "run_id")
        site = get_col(row, "site", "site_id")
        dx = get_col(row, "diagnosis", "dx", "label")
        age = get_col(row, "age", "participant_age")
        sex = get_col(row, "sex", "gender")
        path_fmri = get_col(row, "path_fmri", "bold_path", "nii_path", "filepath")

        if not subj or not path_fmri:
            # can't proceed without ID and file path
            continue

        participant_id = f"sub-{str(subj).strip()}"
        ses_label = norm_ses(ses)
        run_label = norm_run(run)
        task = guess_task(path_fmri, get_col(row, "task", "task_name"))

        # Participants file: write once per participant
        if participant_id not in seen:
            with open(ptsv, "a", newline="") as pf:
                pf.write(
                    f"{participant_id}\t{site or 'n/a'}\t{dx or 'n/a'}\t{age or 'n/a'}\t{sex or 'n/a'}\n"
                )
            seen.add(participant_id)

        # Build BIDS paths
        func_dir = Path(BIDS) / participant_id / ses_label / "func"
        func_dir.mkdir(parents=True, exist_ok=True)

        # BIDS filename (minimal set: sub, ses, task, run, suffix)
        bids_stem = f"{participant_id}_{ses_label}_task-{task}_{run_label}"
        bids_nii = func_dir / f"{bids_stem}_bold.nii.gz"
        bids_json = func_dir / f"{bids_stem}_bold.json"

        # Symlink source -> BIDS
        src_path = Path(path_fmri)
        if src_path.exists():
            safe_symlink(src_path, bids_nii)
        else:
            missing.append(path_fmri)
            continue  # skip sidecar if file missing

        # Minimal sidecar JSON
        sidecar = {"TaskName": task}
        bids_json.write_text(json.dumps(sidecar, indent=2) + "\n")

# Log missing files (if any)
if missing:
    with open(LOG_MISSING, "a") as mf:
        mf.write("\n".join(missing) + "\n")
