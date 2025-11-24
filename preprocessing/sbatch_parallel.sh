for i in $(seq -w 00 19); do
    echo "Running chunk ${i}..."
    MANIFEST="/home/amahama/PROJECTS/1_sensory/data/manifests/1_PREPROCESS/abide_fs5/fMRI_surface_ABIDE2_chunk${i}.csv"

    python3 /home/amahama/PROJECTS/1_sensory/scripts/1_sensory_mapping_preprocess.py \
      --manifest "${MANIFEST}" \
      --lh_annot /home/amahama/PROJECTS/1_sensory/data/manifests/templates/lh.HCP-MMP1.fsaverage5.annot \
      --rh_annot /home/amahama/PROJECTS/1_sensory/data/manifests/templates/rh.HCP-MMP1.fsaverage5.annot \
      --outdir /home/amahama/PROJECTS/1_sensory/data/manifests/OUTPUT &
done

wait
