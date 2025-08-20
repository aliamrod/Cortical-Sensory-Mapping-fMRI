# Cortical-Sensory-Mapping-fMRI
Pipeline for mapping cortical sensory integration from fMRI using the HCP-MMP atlas, with downsampling to fsaverage5, non-negative regression, and HSV visualization.



**A. Script 1**

The script implements sensory-integration mapping on **fsaverage5** space (10,242 vertices per hemisphere; 20,484 total). "Seeds" are V1, S1 (areas 3a/3b/1/2), and A1 from HCP-MMP1 annotations. For each subject, it:

(1) builds seed masks from `.annot`, 

(2) extracts seed mean time series from LH/RH functional GIFTIs resampled to fsaverage5,

(3) performs non-negative regression (beta ≥ 0) per vertex using the three seed series as predictors, 

(4) converts betas to HSV/RGB color encodings, 

(5) aggregates to group level with circular stats for hue.


