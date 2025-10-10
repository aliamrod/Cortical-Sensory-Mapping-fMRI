cd /home/amahama/PROJECTS/1_sensory/data/manifests

# Inspect headers 
csvcut -n fMRI_master_file_MNI_pass.csv
csvcut -n fMRI_surface.csv

# fMRI_master_file_MNI_pass.csv columns
#  1: 
#  2: subject_id
#  3: session_id
#  4: run
#  5: age
#  6: sex
#  7: site
#  8: scanner_id
#  9: diagnosis
# 10: path_fmri
# 11: path_fmriprep
# 12: preprocessing_failed_fmriprep_stable
# 13: uid
# 14: uid2

# fMRI_surface.csv columns
#  1: subject_id
#  2: session_id
#  3: run
#  4: age
#  5: sex
#  6: site
#  7: diagnosis
#  8: path_fmri
#  9: path_fmriprep
# 10: preprocessing_failed_fmriprep_stable
# 11: uid
# 12: uid2
# 13: dataset
# 14: tr
# 15: time_len
# 16: uid_tail

# Extract unique subject lists
csvcut -c subject_id fMRI_master_file_MNI_pass.csv | tail -n +2 | sort -u > master.ids
csvcut -c subject_id fMRI_surface.csv | tail -n +2 | sort -u > surface.ids

# Intersection: in both MASTER and SURFACE/PROCESSED
comm -12 master.ids surface.ids > processed.ids

# select * from master where subject_id NOT IN (select subject_id from surface)
comm -23 master.ids surface.ids > unprocessed.ids

{
  echo "subject_id,status"
  awk '{print $0",processed"}'   processed.ids
  awk '{print $0",unprocessed"}' unprocessed.ids
} > status_map.csv

csvgrep -c subject_id -f processed.ids fMRI_master_file_MNI_pass.csv > master_processed01.csv
csvgrep -c subject_id -f unprocessed.ids fMRI_master_file_MNI_pass.csv > master_unprocessed.csv

# Make ONE master with a status column attached
csvjoin -c subject_id fMRI_master_file_MNI_pass.csv status_map.csv > master_with_status.csv


# Counts
echo "Processed (in both):   $(wc -l < processed.ids)"
echo "Unprocessed (in master only): $(wc -l < unprocessed.ids)"
echo "Surface-only (not in master, FYI): $(wc -l < extra_surface_only.ids)"

cd /home/amahama/PROJECTS/1_sensory/data/manifests
csvcut -C 1 -d ',' fMRI_master_file_MNI_pass.csv > master.clean.csv
mv master_with_status.csv master_with_status.preclean.csv 2>/dev/null || true
csvjoin -c subject_id -d ',' master.clean.csv status_map.csv > master_with_status.csv
mv master_processed01.csv master_processed.preclean.csv 2>/dev/null || true
csvgrep -c subject_id -f processed.ids -d ',' master.clean.csv > master_processed.csv
csvgrep -c subject_id -f unprocessed.ids -d ',' master.clean.csv > master_unprocessed.csv

# Generate run list for BIDS building
# subject_id, session_id, run, path_fmri (from master, cleaned version)
csvcut -c subject_id,session_id,run,path_fmri -d ',' master_unprocessed.csv > \ unprocessed_minimal.csv
