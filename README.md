# Cortical-Sensory-Mapping-fMRI
Pipeline for mapping cortical sensory integration from fMRI using the HCP-MMP atlas, with downsampling to fsaverage5, non-negative regression, and HSV visualization.



**A. Script 1**

Objective: (1) Build V1/S1/A1 masks on fsaverage5 from HCP-MMP1.annot files.

(2) For each subject, time series within V1, S1, and A1 is averaged -> used these three traces as predictors. 
(3) Regress every vertex's time series on those preddictors with non-negative coefficients. 
(4) Betas -> R (V1), G (S1), B (A1); R² → A (alpha).
(5) Convert per-vertex RGB betas → HSV hue/“strength” for visualization.

