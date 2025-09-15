# Before running BIDS validator. 
cd /home/amahama/PROJECTS/1_sensory/bids_dir05

# NIfTIs
find sub-* -type f -name "*_task-rest_run_run-*_bold.nii.gz" \
  -exec bash -c 'for f; do mv "$f" "${f/_task-rest_run_run-/_task-rest_run-}"; done' _ {} +

# JSON sidecars
find sub-* -type f -name "*_task-rest_run_run-*_bold.json" \
  -exec bash -c 'for f; do mv "$f" "${f/_task-rest_run_run-/_task-rest_run-}"; done' _ {} +

# Fix ..._run_run-XX_ → ..._run-XX_  (NIfTI + JSON)
find sub-* -type f -name "*_run_run-*_bold.nii.gz" \
  -exec bash -c 'for f; do mv "$f" "${f/_run_run-/_run-}"; done' _ {} +
find sub-* -type f -name "*_run_run-*_bold.json" \
  -exec bash -c 'for f; do mv "$f" "${f/_run_run-/_run-}"; done' _ {} +



# Sanitize paths with duplicate run tags (e.g., '_run-01_run-01_)
python3 - <<'PY'
from pathlib import Path
import re, os, json

root = Path(".")
pat = re.compile(r"""
    (?P<prefix>.+?_task-(?P<task>[A-Za-z0-9]+)_)
    run-(?P<run>\d+)
    _bold\.(?P<ext>nii\.gz|json)
""", re.X)

def pad_run(r): 
    return f"{int(r):02d}"

def rename(p: Path, newname: str):
    if p.name != newname:
        p.rename(p.with_name(newname))
        return True
    return False

# pass 1: normalize names, pad run
changed = 0
for p in list(root.rglob("*_bold.nii.gz")) + list(root.rglob("*_bold.json")):
    m = pat.match(p.name)
    if not m: 
        continue
    pref, task, run, ext = m.group("prefix","task","run","ext")
    task = re.sub(r"[^A-Za-z0-9]","",task) or "rest"
    new = f"{pref.split('_task-')[0]}_task-{task}_run-{pad_run(run)}_bold.{ext}"
    if rename(p, new): 
        changed += 1
print(f"Renamed {changed} files")

# pass 2: ensure json/nii pairs exist & TaskName is clean
fixed_json = 0
for nii in root.rglob("*_bold.nii.gz"):
    base = nii.name[:-7]  # strip '.nii.gz'
    j = nii.with_name(base + ".json")
    if j.exists():
        try:
            d = json.loads(j.read_text())
        except Exception:
            d = {}
        # get task from filename
        m = re.search(r"_task-([A-Za-z0-9]+)_", nii.name)
        task = m.group(1) if m else "rest"
        d["TaskName"] = task
        j.write_text(json.dumps(d, indent=2) + "\n")
        fixed_json += 1
print(f"Updated TaskName in {fixed_json} JSONs")

# pass 3: remove orphan JSONs without matching NIfTI
removed = 0
for j in root.rglob("*_bold.json"):
    nii = j.with_name(j.name.replace("_bold.json","_bold.nii.gz"))
    if not nii.exists():
        j.unlink()
        removed += 1
print(f"Removed {removed} orphan JSONs")
PY


###########################################################################################

module load apptainer 2>/dev/null || true
module load singularity 2>/dev/null || true
module load freesurfer

# ---Detect---
if command -v apptainer >/dev/null 2>&1; then
  export CTL=apptainer
elif command -v singularity >/dev/null 2>&1; then
  export CTL=singularity
else
  echo "Neither apptainer nor singularity found. Ask admin to enable one." >&2
  return 2 2>/dev/null || exit 2
fi

echo "Using: $CTL"

# Set paths for project.
BIDS=/home/amahama/PROJECTS/1_sensory/bids_dir04
DERIV=$BIDS/derivatives/fmriprep
WORK=$BIDS/work_fmriprep
LOGS=$BIDS/logs
mkdir -p "$DERIV" "$WORK" "$LOGS"

# BIDS validation, containerized.
# try central image path; otherwise pull locally
VALIDATOR_IMG=/mnt/lmod/software/singularity/images/bids_validator:1.14.5.simg
[ -f "$VALIDATOR_IMG" ] || VALIDATOR_IMG=$BIDS/bids_validator_1.14.5.sif
[ -f "$VALIDATOR_IMG" ] || $CTL pull "$VALIDATOR_IMG" docker://bids/validator:1.14.5

# run it
$CTL run "$VALIDATOR_IMG" "$BIDS"

