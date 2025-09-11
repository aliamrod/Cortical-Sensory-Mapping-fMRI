#!/usr/bin/env python3
"""
CSV → BIDS (fMRIPrep-ready)

- Strictly uses path_fmri as-is (no remaps/searches).
- Creates sub-*/ses-*/func for *all* subjects that pass the failure filter,
  so the directory is complete even if some files are missing.
- Symlinks bold files only when they exist.
- Writes minimal JSON sidecars (TaskName).
- Writes participants.tsv.
- Emits subject lists for fMRIPrep:
    - subjects_with_bold.txt (with 'sub-' prefix)
    - subjects_with_bold_noprefix.txt (bare labels for --participant-label)
"""

import csv, os, re, json
from pathlib import Path

# ---------- config ----------
MASTER = "/home/amahama/PROJECTS/1_sensory/data/manifests/fMRI_master_file_MNI_pass.csv"
BIDS_DIR = "/home/amahama/PROJECTS/1_sensory/bids_dir03"
LOGS_DIR = "/home/amahama/PROJECTS/1_sensory/logs"
USE_SYMLINKS = True
DATASET_NAME = "Sensory Mapping BIDS"
BIDS_VERSION = "1.8.0"
# ----------------------------

Path(BIDS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
LOG_MISSING = str(Path(LOGS_DIR) / "missing_files.txt")
open(LOG_MISSING, "w").write("")  

# dataset_description.json
(Path(BIDS_DIR) / "dataset_description.json").write_text(
    json.dumps({"Name": DATASET_NAME, "BIDSVersion": BIDS_VERSION, "DatasetType": "raw"}, indent=2) + "\n"
)

# participants.tsv
ptsv = Path(BIDS_DIR) / "participants.tsv"
if not ptsv.exists():
    with open(ptsv, "w") as f:
        f.write("participant_id\tsite\tdiagnosis\tage\tsex\n")

seen_subjects = set()
made_subject_dirs = set()
subjects_with_bold = set()
missing = []

# ---- helpers ----
def digits_only(s: str, default="1") -> str:
    if not s: return default
    d = re.sub(r"[^0-9]", "", str(s).strip())
    return d or default

def norm_ses(s: str) -> str:
    s = re.sub(r"(?i)^ses-", "", str(s).strip()) if s else ""
    return f"ses-{digits_only(s)}"

def norm_run(s: str) -> str:
    s = re.sub(r"(?i)^run-", "", str(s).strip()) if s else ""
    return f"run-{digits_only(s)}"

def task_from_path(path: str) -> str:
    m = re.search(r"task-([A-Za-z0-9_]+)", os.path.basename(path or ""))
    return m.group(1) if m else "rest"

def link_or_copy(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        try:
            if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
                return
            dst.unlink()
        except FileNotFoundError:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    if USE_SYMLINKS:
        os.symlink(src, dst)
    else:
        import shutil
        shutil.copy2(src, dst)

def failed(flag: str) -> bool:
    s = (flag or "").strip().lower()
    return s in ("failed", "true", "1", "yes", "fail")

# ---- main ----
c_total=c_skip_failed=c_keep=c_haspath=c_exist=c_link=0
subjects_in_csv=set()

with open(MASTER, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        c_total += 1

        flag = row.get("preprocessing_failed_fmriprep_stable","")
        if failed(flag):
            c_skip_failed += 1
            continue
        c_keep += 1

        subj_raw = (row.get("subject_id") or "").strip()
        ses      = norm_ses(row.get("session_id",""))
        run      = norm_run(row.get("run",""))
        site     = (row.get("site","") or "").strip()
        dx       = (row.get("diagnosis","") or "").strip()
        age      = (row.get("age","") or "").strip()
        sex      = (row.get("sex","") or "").strip()
        path     = (row.get("path_fmri","") or "").strip()

        if not subj_raw:
            continue
        subjects_in_csv.add(subj_raw)

        participant_id = subj_raw if subj_raw.startswith("sub-") else f"sub-{subj_raw}"
        subj_root = Path(BIDS_DIR) / participant_id / ses

        # Always ensure dirs + participants row exist (even if file missing)
        if participant_id not in made_subject_dirs:
            (subj_root/"func").mkdir(parents=True, exist_ok=True)
            made_subject_dirs.add(participant_id)
        if participant_id not in seen_subjects:
            with open(ptsv, "a") as pf:
                pf.write(f"{participant_id}\t{site or 'n/a'}\t{dx or 'n/a'}\t{age or 'n/a'}\t{sex or 'n/a'}\n")
            seen_subjects.add(participant_id)

        # If no path, skip file ops
        if not path:
            continue
        c_haspath += 1

        task = task_from_path(path)
        func_dir = subj_root/"func"
        stem = f"{participant_id}_{ses}_task-{task}_{run}"
        dst_nii  = func_dir/f"{stem}_bold.nii.gz"
        dst_json = func_dir/f"{stem}_bold.json"

        src = Path(path)
        if src.exists():
            c_exist += 1
            link_or_copy(src, dst_nii)
            dst_json.write_text(json.dumps({"TaskName": task}, indent=2) + "\n")
            c_link += 1
            subjects_with_bold.add(participant_id)
        else:
            missing.append(path)

# Write missing paths log
if missing:
    with open(LOG_MISSING, "a") as mf:
        mf.write("\n".join(missing) + "\n")

# Write subject lists for fMRIPrep
subjects_with_bold = sorted(subjects_with_bold)
with open(Path(LOGS_DIR)/"subjects_with_bold.txt","w") as f:
    for s in subjects_with_bold:
        f.write(s+"\n")
with open(Path(LOGS_DIR)/"subjects_with_bold_noprefix.txt","w") as f:
    for s in subjects_with_bold:
        f.write(s.replace("sub-","",1)+"\n")

# Summary
summary = {
    "rows_total": c_total,
    "rows_skipped_failed": c_skip_failed,
    "rows_kept_after_flag": c_keep,
    "rows_with_path_field": c_haspath,
    "rows_where_path_exists": c_exist,
    "files_linked": c_link,
    "unique_subjects_in_csv": len(subjects_in_csv),
    "unique_subject_dirs_created": len(made_subject_dirs),
    "unique_subjects_with_bold": len(subjects_with_bold),
    "bids_dir": str(BIDS_DIR),
    "missing_log": LOG_MISSING,
    "subjects_list": str(Path(LOGS_DIR)/"subjects_with_bold_noprefix.txt"),
}
print(json.dumps(summary, indent=2))
print("PROCESS HAS BEEN COMPLETED.")
