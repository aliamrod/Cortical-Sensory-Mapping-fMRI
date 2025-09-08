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

head -n 10 fMRI_All_master_file_V7_2025-09-08.csv | uid -t -s,


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


# fMRI PREP
# BIDS CHECK
module avail fMRIprep
module avail freesurfer
module avail singularity 

module load fMRIprep
module load freesurfer
module load singularity

export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1


# Create project layout
mkdir -p ~/PROJECTS/1_sensory/{logs,work,derivatives,code}


