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
