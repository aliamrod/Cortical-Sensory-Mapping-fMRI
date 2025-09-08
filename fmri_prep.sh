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


################################## fMRIPrep SBATCH ##################################
module load fMRIprep
module load freesurfer
module load singularity

cd ~/PROJECTS/1_sensory/data/manifests

# Master CSV + known column indices
CSV=fMRI_master_file_MNI_pass.csv
COL_SITE=7
COL_SUBJECT=2
COL_PATH_FMRI=10

# fMRIPrep / FS locations
FMRIPREP_IMG="/mnt/lmod/software/singularity/images/fmriprep:23.2.1.simg"
FS_LICENSE="$FREESURFER_HOME/license.txt"
FS_SUBJECTS_DIR=""

module load singularity 2>dev/null || true
