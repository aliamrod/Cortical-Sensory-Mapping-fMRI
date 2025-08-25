# Cortical-Sensory-Mapping-fMRI
Pipeline for mapping cortical sensory integration from fMRI using the HCP-MMP atlas, with downsampling to fsaverage5, non-negative regression, and HSV visualization.



**A. 00_split.py**

The script implements sensory-integration mapping on **fsaverage5** space (10,242 vertices per hemisphere; 20,484 total). "Seeds" are V1, S1 (areas 3a/3b/1/2), and A1 from HCP-MMP1 annotations. For each subject, it:

(1) builds seed masks from `.annot`, 

(2) extracts seed mean time series from LH/RH functional GIFTIs resampled to fsaverage5,

(3) performs non-negative regression (beta ≥ 0) per vertex using the three seed series as predictors, 

(4) converts betas to HSV/RGB color encodings, 

(5) aggregates to group level with circular stats for hue.

Imports 

```python
numpy
scipy
nibabel
sklearn.linear_model.LinearRegression(positive=True)
matplotlib.colors # for HSV <-> RGB mapping
tqdm # progress bars
pingouin # circular means (group-level hue aggregation)

```

**B. 01_sensory_mapping_preprocess.py**

To run, verify deps are installed


```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy scikit-learn nibabel pingouin tqdm matplotlib 
```


**C. 02_statistic_anglecomp_perm.py**


