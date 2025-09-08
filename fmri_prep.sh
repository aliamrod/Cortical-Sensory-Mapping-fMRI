# fMRI Prep 

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

# Excluse the failed subjects first, then proceed to convert the successful runs from MNI Space -> fsaverage5. 
head -n 1 fMRI_All_master_file_V7_2025-09-08.csv > fMRI_All_master_file_MNIonly.csv
grep -Ev "Failed"|"failed" fMRI_All_master_file_V7_2025_09-08.csv >> fMRI_All_master_file_pass_MNI.csv




