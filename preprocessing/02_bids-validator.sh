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

