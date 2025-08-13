"""
This script turns your per-vertex non-negative regression outputs (`rgba`: betas + R²) into analysis-ready angles and 
visualizations on fsaverage5, then exports scalar maps to GIFTI. It first converts the three betas (R,G,B) into an HSV hue (degrees)
with a proper wrap to [0,360), then builds an HSV map per vertex where hue encodes sensory angle, saturation is ranked R²
(magnitude, with a zero-division guard), and value is fixed for consistent display; it also returns θ in radians (0..2π) for circular
stats and the ranked-R² vector. For groups, it takes a stack `[S×4×V]`, yields individual θ/R², computes the circular mean angle across subjects
and a group magnitude by ranking mean R², and produces group RGB colors. A vector-composition helper compares two angle maps across conditions
using the signed circular difference and 1 − resultant vector length for effect size. Finally, a robust saver writes per-vertex 
LH/RH GIFTI files with a dynamic hemisphere split (fsaverage5: 10,242 verts per hemi by default), avoiding hardcoded fs\_LR sizes.

"""
# hsv_and_gifti_fs5.py
# HSV + circular stats + GIFTI export (dynamic split)


# Imports
import os
import numpy as np
import nibabel as nib
from nibabel import gifti
import matplotlib.colors as color
import scipy.stats as ss
import pycircstat as circ  # pip install pycircstat

# ---------- RGB betas -> Hue (degrees) ----------
def v_rgba2hue(rgba):
    """
    rgba: [4 x V], rows 0..2 are non-negative betas (R,G,B).
    Returns hue (degrees) in [0, 360).
    """
    rgbv = np.asarray(rgba[:3, :], dtype=float)
    V = rgbv.shape[1]
    H = np.zeros(V, dtype=float)

    delta = np.max(rgbv, axis=0) - np.min(rgbv, axis=0)
    del_0 = (delta == 0)
    H[del_0] = 0.0

    if np.any(~del_0):
        rgbv_pos = rgbv[:, ~del_0]
        del_pos  = delta[~del_0]
        ind_max  = np.argmax(rgbv_pos, axis=0)
        Hp = np.zeros(rgbv_pos.shape[1], dtype=float)

        sel = (ind_max == 0)  # R max
        if np.any(sel):
            Hp[sel] = 60.0 * ((rgbv_pos[1, sel] - rgbv_pos[2, sel]) / del_pos[sel])
        sel = (ind_max == 1)  # G max
        if np.any(sel):
            Hp[sel] = 60.0 * ((rgbv_pos[2, sel] - rgbv_pos[0, sel]) / del_pos[sel] + 2.0)
        sel = (ind_max == 2)  # B max
        if np.any(sel):
            Hp[sel] = 60.0 * ((rgbv_pos[0, sel] - rgbv_pos[1, sel]) / del_pos[sel] + 4.0)

        # Wrap to [0, 360)
        H[~del_0] = Hp % 360.0

    return H

# ---------- Per-subject HSV + angles ----------
def v_hsv_model_rgba(rgba):
    """
    rgba: [4 x V] (rows 0..2 = betas; row 3 = R^2)
    Returns:
      hsv:   [V x 3] with hue in [0,1], sat=ranked R^2, val=0.86
      rgb:   [V x 3] (float in [0,1])
      theta: [V] radians in [0, 2π)
      rd:    [V] ranked R^2 in [0,1]
    """
    rgba = np.asarray(rgba, dtype=float)
    hue_deg = v_rgba2hue(rgba)  # [V], degrees

    # Rank-based magnitude (guard against all-equal)
    rd = ss.rankdata(rgba[3, :])
    den = rd.max() - rd.min()
    rd = (rd - rd.min()) / (den if den != 0 else 1.0)

    hsv = np.zeros((rgba.shape[1], 3), dtype=float)
    hsv[:, 0] = hue_deg / 360.0
    hsv[:, 1] = rd
    hsv[:, 2] = 0.86

    rgb   = color.hsv_to_rgb(hsv)
    theta = 2.0 * np.pi * hsv[:, 0]  # radians
    return hsv, rgb, theta, rd

# ---------- Group aggregation ----------
def v_hsv_model_rgba_indiv(rgba):
    """
    rgba: [S x 4 x V]
    Returns:
      th_indiv: [S x V] individual angles (radians)
      rd_indiv: [S x V] individual ranked R^2
      th_group: [V]     circular mean angle (radians in [0, 2π))
      rd_group: [V]     rank of group-mean R^2 in [0,1]
      rgb_group:[V x 3] HSV mapped from (th_group, rd_group, 0.86)
    """
    rgba = np.asarray(rgba, dtype=float)
    S, _, V = rgba.shape

    th_indiv, rd_indiv = [], []
    for s in range(S):
        _, _, th, rd = v_hsv_model_rgba(rgba[s, :, :])
        th_indiv.append(th); rd_indiv.append(rd)
    th_indiv = np.asarray(th_indiv)
    rd_indiv = np.asarray(rd_indiv)

    # Group magnitude: rank of mean R^2
    ex_group = rgba[:, 3, :].mean(axis=0)
    ex_rank  = ss.rankdata(ex_group)
    den = ex_rank.max() - ex_rank.min()
    rd_group = (ex_rank - ex_rank.min()) / (den if den != 0 else 1.0)

    # Circular mean of angles (pycircstat expects radians)
    th_group = circ.descriptive.mean(th_indiv, axis=0)
    th_group[th_group < 0] += 2.0 * np.pi  # wrap to [0, 2π)

    hsv = np.zeros((V, 3), dtype=float)
    hsv[:, 0] = th_group / (2.0 * np.pi)  # hue 0..1
    hsv[:, 1] = rd_group
    hsv[:, 2] = 0.86
    rgb_group = color.hsv_to_rgb(hsv)

    return th_indiv, rd_indiv, th_group, rd_group, rgb_group

# ---------- Vector composition (condition difference) ----------
def v_vector_comp(mat1, mat2):
    """
    mat1, mat2: [S x V] angles (radians).
    Returns theta_com: [S x V] signed dispersion difference:
      sign from circ.cdiff, magnitude 1 - resultant_vector_length over the pair.
    """
    mat1 = np.asarray(mat1, dtype=float)
    mat2 = np.asarray(mat2, dtype=float)

    theta_dif = circ.cdiff(mat1, mat2)    # signed angle diff in [-π, π)
    theta_sign = np.zeros_like(theta_dif)
    theta_sign[theta_dif > 0] =  1.0
    theta_sign[theta_dif < 0] = -1.0

    theta_com = []
    for s in range(mat1.shape[0]):
        theta_pack = np.vstack((mat1[s, :], mat2[s, :]))  # [2 x V]
        # 1 - R, where R is resultant_vector_length across the two angles
        theta_diff = 1.0 - circ.descriptive.resultant_vector_length(theta_pack, axis=0)
        theta_com.append(theta_diff)
    theta_com = np.asarray(theta_com)
    theta_com = theta_com * theta_sign
    return theta_com

# ---------- Export to GIFTI ----------
def v_save_gii(data, savepath, savename, half=False, n_lh=None):
    """
    Save per-vertex scalar(s) to GIFTI.
    - data: 1D [V] array (LH then RH). If 'half' is True, 'data' is assumed LH only.
    - half: if True, write a single LH file only (lh.{savename}.func.gii).
    - n_lh: number of LH vertices (fsaverage5=10242, fs_LR32k=32492). If None, infer V//2.
    Writes:
      lh.{savename}.func.gii  (and rh.* if half==False)
    """
    os.makedirs(savepath, exist_ok=True)

    if half:
        arr = np.asarray(data, dtype=np.float32).ravel()
        img = gifti.GiftiImage()
        img.add_gifti_data_array(gifti.GiftiDataArray(arr))
        nib.save(img, os.path.join(savepath, f"lh.{savename}.func.gii"))
        return

    arr = np.asarray(data, dtype=np.float32).ravel()
    V = arr.size
    if n_lh is None:
        if V % 2 != 0:
            raise ValueError("Cannot infer n_lh from odd-length data.")
        n_lh = V // 2
    n_rh = V - n_lh
    if n_lh <= 0 or n_rh <= 0:
        raise ValueError(f"Bad split n_lh={n_lh}, n_rh={n_rh}, V={V}")

    lh_img = gifti.GiftiImage()
    rh_img = gifti.GiftiImage()
    lh_img.add_gifti_data_array(gifti.GiftiDataArray(arr[:n_lh]))
    rh_img.add_gifti_data_array(gifti.GiftiDataArray(arr[n_lh:]))

    nib.save(lh_img, os.path.join(savepath, f"lh.{savename}.func.gii"))
    nib.save(rh_img, os.path.join(savepath, f"rh.{savename}.func.gii"))
