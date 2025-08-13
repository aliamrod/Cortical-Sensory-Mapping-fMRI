
# 01_sensory_mapping_preprocess.py
# Robust fsaverage5 implementation of V1/S1/A1-based sensory integration mapping.
# Includes: .annot name-based masks, non-negative regression, HSV mapping, and group aggregation.

# Set up environment
# python3 -m pip install --upgrade pip
# python3 -m pip install numpy scipy scikit-learn nibabel pingouin tqdm matplotlib


# Imports
import os
import numpy as np
import scipy.stats as ss
from scipy.optimize import nnls
import nibabel as nib
from nibabel.freesurfer.io import read_annot
from sklearn.linear_model import LinearRegression as LReg
from matplotlib import colors as color
from tqdm import tqdm
import pingouin as pg

"""
# Required inputs 
# fsaverage5 HCP-MMP1 annotation files (vertex count 10,242 per hemisphere) 
#   - {"$SUBJECTS_DIR/fsaverage5/label/lh.HCP-MMP1.fsaverage5.annot"}
#   - {"$SUBJECTS_DIR/fsaverage5/label/rh.HCP-MMP1.fsaverage5.annot"}
# For each subject: LH/RH GIFTI functional files resampled to fsaverage5 (same T per hemi), e.g.:
#   - .../sub-XX.L.func.gii
#   - .../sub-XX.R.func.gii
# General assumption: vertex order is LH (10,242) and RH (10,242). 
"""
lh_annot = "/home/amahama/PROJECTS/1_sensory/data/HCP-MMP1.fsaverage/lh.HCP-MMP1.fsaverage5.annot"
rh_annot = "/home/amahama/PROJECTS/1_sensory/data/HCP-MMP1.fsaverage/rh.HCP-MMP1.fsaverage5.annot"

# -------------------- Utilities --------------------
def _b2s(x):
    """bytes -> str (for .annot name tables)."""
    return x.decode() if isinstance(x, (bytes, bytearray)) else x

def _norm_name(n):
    """Normalize HCP-MMP names e.g. 'L_V1_ROI' -> 'V1'."""
    n = _b2s(n)
    if n.startswith(("L_", "R_")):
        n = n[2:]
    if n.endswith("_ROI"):
        n = n[:-4]
    return n

def _ids_for(names_array, targets):
    """Return label indices whose normalized name is in targets (case-insensitive exact match)."""
    targets = {t.lower() for t in targets}
    ids = []
    for i, n in enumerate(names_array):
        if _norm_name(n).lower() in targets:
            ids.append(i)
    return ids

# -------------------- Mask building --------------------

def build_primary_masks_fs5(lh_annot_path, rh_annot_path):
    """
    Build boolean masks for V1, S1 (3a/3b/1/2), and A1 from fsaverage5 .annot files.
    Assumes vertex order [LH(10242), RH(10242)] concatenated -> V=20484.
    """
    lh_labels, _, lh_names = read_annot(os.path.expanduser(lh_annot_path))
    rh_labels, _, rh_names = read_annot(os.path.expanduser(rh_annot_path))
    lh_names = [_b2s(n) for n in lh_names]
    rh_names = [_b2s(n) for n in rh_names]

    V1 = ["V1"]
    S1 = ["3a", "3b", "1", "2"]
    A1 = ["A1"]

    lh_V1_ids = _ids_for(lh_names, V1); rh_V1_ids = _ids_for(rh_names, V1)
    lh_S1_ids = _ids_for(lh_names, S1); rh_S1_ids = _ids_for(rh_names, S1)
    lh_A1_ids = _ids_for(lh_names, A1); rh_A1_ids = _ids_for(rh_names, A1)

    if len(lh_V1_ids + rh_V1_ids) == 0:
        raise RuntimeError("V1 not found in annot names. Inspect/adjust targets.")
    if len(lh_S1_ids + rh_S1_ids) == 0:
        raise RuntimeError("S1 (3a/3b/1/2) not found in annot names. Adjust targets.")
    if len(lh_A1_ids + rh_A1_ids) == 0:
        raise RuntimeError("A1 not found in annot names. Inspect/adjust targets.")

    lh_V1 = np.isin(lh_labels, lh_V1_ids); rh_V1 = np.isin(rh_labels, rh_V1_ids)
    lh_S1 = np.isin(lh_labels, lh_S1_ids); rh_S1 = np.isin(rh_labels, rh_S1_ids)
    lh_A1 = np.isin(lh_labels, lh_A1_ids); rh_A1 = np.isin(rh_labels, rh_A1_ids)

    V1_mask = np.concatenate([lh_V1, rh_V1])
    S1_mask = np.concatenate([lh_S1, rh_S1])
    A1_mask = np.concatenate([lh_A1, rh_A1])
    return {"V1": V1_mask, "S1": S1_mask, "A1": A1_mask}

# -------------------- Data loading helpers --------------------

def load_subject_ts_fs5(lh_func_gii, rh_func_gii):
    """
    Load LH/RH GIFTI functional time series and concatenate LH->RH.
    Returns ts: [V x T] with V=20484.
    """
    lh = nib.load(os.path.expanduser(lh_func_gii))
    rh = nib.load(os.path.expanduser(rh_func_gii))
    lh_d = np.column_stack([da.data for da in lh.darrays])  # [10242 x T]
    rh_d = np.column_stack([da.data for da in rh.darrays])  # [10242 x T]
    ts = np.vstack([lh_d, rh_d]).astype(float)              # [20484 x T]
    return ts

# -------------------- Core mapping --------------------

def v_ts_nnls_vertex_fs5(ts, masks):
    """
    Non-negative regression per vertex using V1/S1/A1 mean time series as predictors.
    ts: [V x T] (V=20484 for fsaverage5; LH first then RH)
    masks: dict {"V1": bool[V], "S1": bool[V], "A1": bool[V]}
    Returns rgba [4 x V]: betas (R,G,B) and R^2 (A).

    Row 0 -> beta weight for V1 predictor (visual seed)
    Row 1 -> beta weight for S1 predictor (somatosensory seed)
    Row 2 -> beta weight for A1 predictor (auditory seed)
    Row 3 -> R^2 value (variance explained at that vertex)
    """
    ts = np.asarray(ts, dtype=float)
    V, T = ts.shape
    if V != 20484:
        # Not hard failing; warn via assertion for correctness.
        assert V % 2 == 0, "Expected LH+RH concatenation for fsaverage5."
    for k in ("V1", "S1", "A1"):
        if masks[k].shape[0] != V:
            raise ValueError(f"Mask '{k}' length {masks[k].shape[0]} != ts vertices {V}")

    # NaN-tolerant seed means
    ts_v1 = np.nanmean(ts[masks["V1"], :], axis=0)
    ts_s1 = np.nanmean(ts[masks["S1"], :], axis=0)
    ts_a1 = np.nanmean(ts[masks["A1"], :], axis=0)

    X = np.vstack((ts_v1, ts_s1, ts_a1)).T  # [T x 3]
    Y = ts.T                                 # [T x V]

    # Check predictors are not constant
    if np.any(np.nanstd(X, axis=0) == 0):
        raise ValueError("One or more seed predictors are constant—check masks and ts.")

    # Fast path: sklearn multi-output with non-negative coefficients
    coefs = np.zeros((3, V), dtype=float)
    Y_hat = np.zeros_like(Y)
    try:
        reg = LReg(positive=True)
        reg.fit(X, Y)
        coefs = reg.coef_.T                  # [3 x V]
        Y_hat = reg.predict(X)               # [T x V]
    except TypeError:
        # Fallback to strict NNLS per vertex
        for v in range(V):
            b, _ = nnls(X, Y[:, v])
            coefs[:, v] = b
            Y_hat[:, v] = X @ b

    rgba = np.zeros((4, V), dtype=float)
    rgba[:3, :] = coefs

    y_bar = np.nanmean(Y, axis=0, keepdims=True)
    ss_total = np.nansum((Y - y_bar) ** 2, axis=0)
    ss_exp   = np.nansum((Y_hat - y_bar) ** 2, axis=0)
    eps = np.finfo(float).eps
    ss_total = np.where(ss_total == 0, eps, ss_total)
    rgba[3, :] = ss_exp / ss_total
    return rgba

# -------------------- Color models --------------------

def v_rgba2hue(rgba):
    """Hue (degrees) from RGB betas (rows 0–2), wrapped to [0, 360)."""
    rgbv = rgba[:3, :]
    H = np.zeros(rgbv.shape[1], dtype=float)
    delta = np.max(rgbv, axis=0) - np.min(rgbv, axis=0)
    del_0 = (delta == 0)
    H[del_0] = 0.0
    if np.any(~del_0):
        rgbv_pos = rgbv[:, ~del_0]
        del_pos = delta[~del_0]
        ind_max = np.argmax(rgbv_pos, axis=0)
        Hp = np.zeros(rgbv_pos.shape[1], dtype=float)
        sel = (ind_max == 0)
        if np.any(sel):
            Hp[sel] = 60 * ((rgbv_pos[1, sel] - rgbv_pos[2, sel]) / del_pos[sel])
        sel = (ind_max == 1)
        if np.any(sel):
            Hp[sel] = 60 * ((rgbv_pos[2, sel] - rgbv_pos[0, sel]) / del_pos[sel] + 2)
        sel = (ind_max == 2)
        if np.any(sel):
            Hp[sel] = 60 * ((rgbv_pos[0, sel] - rgbv_pos[1, sel]) / del_pos[sel] + 4)
        H[~del_0] = Hp % 360.0
    return H

def v_hsv_model_rgba(rgba):
    """RGBA -> (HSV, RGB, theta_rad, ranked_R2) for one subject."""
    hue = v_rgba2hue(rgba)  # degrees
    rd = ss.rankdata(rgba[3, :])
    denom = (rd.max() - rd.min())
    rd = (rd - rd.min()) / (denom if denom != 0 else 1.0)

    hsv = np.zeros((rgba.shape[1], 3), dtype=float)
    hsv[:, 0] = hue / 360.0
    hsv[:, 1] = rd
    hsv[:, 2] = 0.86
    rgb   = color.hsv_to_rgb(hsv)
    theta = 2 * np.pi * (hue / 360.0)
    return hsv, rgb, theta, rd

def v_hsv_model_rgba_indiv(rgba):
    """
    rgba: [S x 4 x V] (subjects, channels, vertices)
    Returns: th_indiv[S x V], rd_indiv[S x V], th_group[V], rd_group[V], rgb_group[V x 3]
    """
    n_sub, _, n_ver = rgba.shape
    th_indiv, rd_indiv = [], []
    for s in tqdm(range(n_sub), desc="HSV per subject"):
        _, _, th, rd = v_hsv_model_rgba(rgba[s, :, :])
        th_indiv.append(th); rd_indiv.append(rd)
    th_indiv = np.asarray(th_indiv)
    rd_indiv = np.asarray(rd_indiv)

    ex_group = rgba[:, 3, :].mean(axis=0)
    ex_group = ss.rankdata(ex_group)
    denom = (ex_group.max() - ex_group.min())
    rd_group = (ex_group - ex_group.min()) / (denom if denom != 0 else 1.0)

    th_group = pg.circ_mean(th_indiv, axis=0)
    th_group[th_group < 0] += 2 * np.pi

    hsv = np.zeros((n_ver, 3), dtype=float)
    hsv[:, 0] = th_group / (2 * np.pi)
    hsv[:, 1] = rd_group
    hsv[:, 2] = 0.86
    rgb_group = color.hsv_to_rgb(hsv)
    return th_indiv, rd_indiv, th_group, rd_group, rgb_group

# -------------------- USAGE OVERVIEW --------------------
# 1) Build masks once (fsaverage5 .annot files):
# lh_annot = "$SUBJECTS_DIR/fsaverage5/label/lh.HCP-MMP1.annot"
# rh_annot = "$SUBJECTS_DIR/fsaverage5/label/rh.HCP-MMP1.annot"
# masks = build_primary_masks_fs5(lh_annot, rh_annot)
#
# 2) For each subject time series ts_fs5 [V x T] (V=20484, LH->RH):
# rgba = v_ts_nnls_vertex_fs5(ts_fs5, masks)
# hsv, rgb, theta, rd = v_hsv_model_rgba(rgba)
#
# 3) Group level (stack subjects):
# rgba_all = np.stack([rgba_subj1, rgba_subj2, ...], axis=0)  # [S x 4 x V]
# th_ind, rd_ind, th_grp, rd_grp, rgb_grp = v_hsv_model_rgba_indiv(rgba_all)


