# fMRI Prep 

# 1. Does the SOURCE file exist?
ls -lh /home/yyang/yang/map_master_fmri/fMRI_All_master_file_V6.csv

# 2. Create destination folder to preferred path
DEST=/home/amahama/PROJECTS/1_sensory/data/manifests
mkdir -p "$DEST"

# 3. Copy here with last date of update in the name (verbose)
STAMP=${date +%F}
cp -v /home/yyang/yang/map_master_fmri/fMRI_All_master_file_V6.csv \
    "$DEST/fMRI_All_master_file_V7_${STAMP}.csv"

# 4. Verify file has been copied and print out
ls -lh "$DEST"v/null | head -n 20
