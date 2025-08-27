# 03_v_model_hsv.py
# Full pipeline:
# - Load [S x 4 x V] RGBA stacks (βV, βS, βA, R²) produced in step 01
# - Compute θ (angle), r (ranked R² magnitude), and group RGB (HSV→RGB)
# - Save arrays + LH/RH GIFTIs
# - Optional between-group comparison (paired or unpaired):
#       * Paired (same S): signed circular dispersion + sign-flip perms
#       * Unpaired: 0.5 * |mean vector difference| + label-shuffle perms
# - Quick visuals (polar θ–r, histograms)
#
# NEW:
#   --lh-surf/--rh-surf to explicitly provide fsaverage5 surfaces (recommended).
#   Falls back to SUBJECTS_DIR or FREESURFER_HOME if not provided.
#   Mesh is loaded once and reused during permutations.
#
# Usage examples:
#   python3 03_v_model_hsv.py \
#       --in-dir ./data/out_fs5_autism --group-name autism \
#       --lh-surf $SUBJECTS_DIR/fsaverage5/surf/lh.white \
#       --rh-surf $SUBJECTS_DIR/fsaverage5/surf/rh.white
#
#   python3 03_v_model_hsv.py \
#       --in-dir ./data/out_fs5_autism   --group-name autism \
#       --in-dir2 ./data/out_fs5_control --group-name2 control \
#       --n-perm 5000 --vert-thresh 0.15 \
#       --lh-surf $SUBJECTS_DIR/fsaverage5/surf/lh.white \
#       --rh-surf $SUBJECTS_DIR/fsaverage5/surf/rh.white
#
# Requirements: numpy, scipy, nibabel, matplotlib, pycircstat, networkx

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolor
from matplotlib.ticker import MaxNLocator
import nibabel as nib
from nibabel.freesurfer.io import read_geometry
import pycircstat as circ
from scipy import stats as ss
import networkx as nx

V_HEMI = 10242  # fsaverage5 vertices per hemisphere

# --------------------------- utilities ---------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def split_lr(vec):
    """Split a V-long vector into LH/RH (fsaverage5)."""
    return vec[:V_HEMI], vec[V_HEMI:]

def load_rgba_stack(path):
    """Load [S x 4 x V] RGBA stack (βV, βS, βA, R²)."""
    arr = np.load(path)
    if not (arr.ndim == 3 and arr.shape[1] == 4):
        raise ValueError(f"Expected [S x 4 x V], got {arr.shape} from {path}")
    return arr

def rgba_to_hue_deg(rgba_4xV):
    """
    Hue transform (degrees in [0, 360)):
    rows 0..2 are βV (visual), βS (somato), βA (auditory).
    """
    rgbv = rgba_4xV[:3, :]
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

def r2_to_rank01(r2_vec):
    """Rank R² to [0,1] (paper defines magnitude as ranked & rescaled)."""
    rd = ss.rankdata(r2_vec)
    denom = (rd.max() - rd.min())
    return (rd - rd.min()) / (denom if denom != 0 else 1.0)

def rgba_stack_to_group(rgba_Sx4xV):
    """Return (theta_indiv, r_indiv, theta_group, r_group, RGB_group[V x 3])."""
    S, _, V = rgba_Sx4xV.shape
    theta_indiv = np.zeros((S, V), dtype=float)
    r_indiv     = np.zeros((S, V), dtype=float)
    for s in range(S):
        H = rgba_to_hue_deg(rgba_Sx4xV[s, :, :])
        theta_indiv[s, :] = np.deg2rad(H)
        r_indiv[s, :]     = r2_to_rank01(rgba_Sx4xV[s, 3, :])

    # group r: rank(mean R²) as in the paper
    r2_mean = rgba_Sx4xV[:, 3, :].mean(axis=0)
    r_group = r2_to_rank01(r2_mean)

    # group θ: circular mean (0..2π)
    theta_grp = circ.descriptive.mean(theta_indiv, axis=0)
    theta_grp[theta_grp < 0] += 2*np.pi

    # RGB from HSV (hue=θ/2π, sat=r, val=0.86)
    hsv = np.zeros((V, 3), dtype=float)
    hsv[:, 0] = theta_grp / (2*np.pi)
    hsv[:, 1] = r_group
    hsv[:, 2] = 0.86
    RGB = mcolor.hsv_to_rgb(hsv)
    return theta_indiv, r_indiv, theta_grp, r_group, RGB

def save_scalar_gifti(vec_V, out_lh, out_rh):
    """Save a scalar vector as LH/RH .func.gii (fsaverage5)."""
    from nibabel.gifti import GiftiImage, GiftiDataArray
    lh, rh = split_lr(vec_V)
    gi_l = GiftiImage(); gi_r = GiftiImage()
    gi_l.add_gifti_data_array(GiftiDataArray(np.asarray(lh, dtype=np.float32)))
    gi_r.add_gifti_data_array(GiftiDataArray(np.asarray(rh, dtype=np.float32)))
    nib.save(gi_l, out_lh); nib.save(gi_r, out_rh)

def save_rgb_gifti(rgb_Vx3, out_lh, out_rh):
    """Save RGB as 3 channels in LH/RH .func.gii."""
    from nibabel.gifti import GiftiImage, GiftiDataArray
    lh, rh = split_lr(rgb_Vx3)
    gi_l = GiftiImage(); gi_r = GiftiImage()
    for k in range(3):
        gi_l.add_gifti_data_array(GiftiDataArray(np.asarray(lh[:, k], dtype=np.float32)))
        gi_r.add_gifti_data_array(GiftiDataArray(np.asarray(rh[:, k], dtype=np.float32)))
    nib.save(gi_l, out_lh); nib.save(gi_r, out_rh)

# --------------------------- mesh handling ---------------------------

class FS5Mesh:
    def __init__(self, lh_surf=None, rh_surf=None):
        self.lh_v, self.lh_f, self.rh_v, self.rh_f = self._load_mesh(lh_surf, rh_surf)

    @staticmethod
    def _auto_fs5_surfaces():
        """Try SUBJECTS_DIR and FREESURFER_HOME to find fsaverage5 surfaces."""
        candidates = []
        sd = os.environ.get("SUBJECTS_DIR")
        fh = os.environ.get("FREESURFER_HOME")
        if sd:
            candidates.append((os.path.join(sd, "fsaverage5/surf/lh.white"),
                               os.path.join(sd, "fsaverage5/surf/rh.white")))
        if fh:
            candidates.append((os.path.join(fh, "subjects/fsaverage5/surf/lh.white"),
                               os.path.join(fh, "subjects/fsaverage5/surf/rh.white")))
        # common module installs
        candidates += [
            ("/usr/local/freesurfer/subjects/fsaverage5/surf/lh.white",
             "/usr/local/freesurfer/subjects/fsaverage5/surf/rh.white"),
            ("/opt/freesurfer/subjects/fsaverage5/surf/lh.white",
             "/opt/freesurfer/subjects/fsaverage5/surf/rh.white"),
            ("/mnt/lmod/software/freesurfer/freesurfer7/subjects/fsaverage5/surf/lh.white",
             "/mnt/lmod/software/freesurfer/freesurfer7/subjects/fsaverage5/surf/rh.white"),
        ]
        for lh, rh in candidates:
            if os.path.isfile(lh) and os.path.isfile(rh):
                return lh, rh
        return None, None

    def _load_mesh(self, lh_surf, rh_surf):
        # Respect explicit paths first
        if lh_surf and rh_surf and os.path.isfile(lh_surf) and os.path.isfile(rh_surf):
            lh_path, rh_path = lh_surf, rh_surf
        else:
            lh_path, rh_path = self._auto_fs5_surfaces()
            if not (lh_path and rh_path):
                raise FileNotFoundError(
                    "fsaverage5 surfaces not found.\n"
                    "Provide --lh-surf and --rh-surf, e.g.:\n"
                    "  --lh-surf $SUBJECTS_DIR/fsaverage5/surf/lh.white "
                    "--rh-surf $SUBJECTS_DIR/fsaverage5/surf/rh.white\n"
                    "Or set SUBJECTS_DIR/FREESURFER_HOME."
                )
        lh_v, lh_f = read_geometry(lh_path)
        rh_v, rh_f = read_geometry(rh_path)
        return lh_v, lh_f.astype(int), rh_v, rh_f.astype(int)

def cluster_labeling_on_fs5(mesh: FS5Mesh, scalar_data, thresh, negate=False):
    """
    Threshold scalar_data (length V) on fsaverage5 and return connected-component
    clusters and sizes for LH and RH based on face adjacencies.
    """
    lh_v, lh_f, rh_v, rh_f = mesh.lh_v, mesh.lh_f, mesh.rh_v, mesh.rh_f
    V = lh_v.shape[0] + rh_v.shape[0]
    if scalar_data.shape[0] != V:
        raise ValueError("Vector length != fsaverage5 vertex count.")

    data = scalar_data.copy()
    if negate:
        data[data > thresh] = 0
    else:
        data[data < thresh] = 0
    mask = (data != 0).astype(int)
    lh_mask = mask[:V_HEMI]; rh_mask = mask[V_HEMI:]

    # Left hemisphere
    lh_idx = np.where(lh_mask == 1)[0]
    G = nx.Graph()
    G.add_nodes_from(lh_idx)
    for f0, f1, f2 in lh_f:
        if lh_mask[f0] and lh_mask[f1]:
            G.add_edge(f0, f1)
        if lh_mask[f1] and lh_mask[f2]:
            G.add_edge(f1, f2)
        if lh_mask[f2] and lh_mask[f0]:
            G.add_edge(f2, f0)
    lh_cc = [list(cc) for cc in nx.connected_components(G) if len(cc) > 0]
    lh_sizes = np.array([len(c) for c in lh_cc]) if lh_cc else np.array([])

    # Right hemisphere
    rh_idx = np.where(rh_mask == 1)[0]
    G = nx.Graph()
    # offset faces by V_HEMI for a shared index space
    G.add_nodes_from((i + V_HEMI for i in rh_idx))
    for f0, f1, f2 in rh_f:
        f0 += V_HEMI; f1 += V_HEMI; f2 += V_HEMI
        if rh_mask[f0 - V_HEMI] and rh_mask[f1 - V_HEMI]:
            G.add_edge(f0, f1)
        if rh_mask[f1 - V_HEMI] and rh_mask[f2 - V_HEMI]:
            G.add_edge(f1, f2)
        if rh_mask[f2 - V_HEMI] and rh_mask[f0 - V_HEMI]:
            G.add_edge(f2, f0)
    rh_cc = [list(cc) for cc in nx.connected_components(G) if len(cc) > 0]
    rh_sizes = np.array([len(c) for c in rh_cc]) if rh_cc else np.array([])

    return lh_cc, lh_sizes, rh_cc, rh_sizes

# ------------------ paired design (same S) helpers ------------------

def signed_circ_disp(theta_A_SxV, theta_B_SxV):
    """
    Paired signed circular dispersion:
      sign = direction of mean circular difference, magnitude = 1 - resultant length
    Returns S x V array.
    """
    if theta_A_SxV.shape != theta_B_SxV.shape:
        raise ValueError("Paired comparison requires same shape arrays.")
    S, V = theta_A_SxV.shape
    dif = circ.cdiff(theta_A_SxV, theta_B_SxV)  # [-π, π]
    sign = np.zeros_like(dif); sign[dif > 0] = 1; sign[dif < 0] = -1
    out = []
    for s in range(S):
        pack = np.vstack((theta_A_SxV[s, :], theta_B_SxV[s, :]))
        disp = 1.0 - circ.descriptive.resultant_vector_length(pack, axis=0)  # V
        out.append(disp)
    out = np.asarray(out)  # S x V
    return out * sign

def perm_maxclus_paired(mesh: FS5Mesh, theta_com_SxV, vert_thresh, n_perm=5000, seed=0):
    """
    Sign-flip permutations for paired design:
      - Randomly multiply each subject by ±1
      - Stat = mean across subjects
      - Record max cluster size at each permutation
    """
    rng = np.random.default_rng(seed)
    S, V = theta_com_SxV.shape
    maxClus = np.zeros(n_perm, dtype=int)
    for n in range(n_perm):
        signs = rng.choice([1.0, -1.0], size=S)[:, None]
        stat = (theta_com_SxV * signs).mean(axis=0)  # V
        _, lsz, _, rsz = cluster_labeling_on_fs5(mesh, stat, vert_thresh, negate=False)
        all_sizes = np.concatenate((lsz, rsz)) if (lsz.size + rsz.size) else np.array([0])
        maxClus[n] = int(all_sizes.max())
    return maxClus

# ------------------ unpaired design helpers ------------------

def group_mean_vector(theta_SxV):
    """Complex mean vector per vertex (shape [V], complex)."""
    z = np.exp(1j * theta_SxV)  # S x V
    return z.mean(axis=0)       # V complex

def angle_stat_unpaired(thetaA_SxV, thetaB_SxV):
    """
    Unpaired group difference statistic in [0,1]:
      stat = 0.5 * | mA - mB |, where mA, mB are complex mean vectors.
    """
    mA = group_mean_vector(thetaA_SxV)
    mB = group_mean_vector(thetaB_SxV)
    return 0.5 * np.abs(mA - mB)  # V

def perm_maxclus_unpaired(mesh: FS5Mesh, thetaA_SxV, thetaB_SxV, vert_thresh, n_perm=5000, seed=0):
    """
    Label-shuffle permutation for unpaired design:
      - Pool subjects
      - Shuffle labels to sizes (SA, SB)
      - Recompute stat = 0.5 * |mA - mB|
      - Record max cluster size
    """
    rng = np.random.default_rng(seed)
    SA, V = thetaA_SxV.shape
    SB, V2 = thetaB_SxV.shape
    if V != V2:
        raise ValueError("Vertex count mismatch between groups.")
    all_theta = np.vstack([thetaA_SxV, thetaB_SxV])   # (SA+SB) x V
    N = SA + SB
    idx = np.arange(N)
    maxClus = np.zeros(n_perm, dtype=int)
    for n in range(n_perm):
        rng.shuffle(idx)
        A_idx = idx[:SA]; B_idx = idx[SA:]
        stat = angle_stat_unpaired(all_theta[A_idx, :], all_theta[B_idx, :])  # V
        _, lsz, _, rsz = cluster_labeling_on_fs5(mesh, stat, vert_thresh, negate=False)
        all_sizes = np.concatenate((lsz, rsz)) if (lsz.size + rsz.size) else np.array([0])
        maxClus[n] = int(all_sizes.max())
    return maxClus

# --------------------------- plotting ---------------------------

def plot_polar_theta_r(theta_group, r_group, out_png):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(theta_group, r_group, s=1, alpha=0.7)
    ax.set_theta_zero_location('E')  # 0° at right (visual)
    ax.set_theta_direction(-1)       # clockwise; 120°=somato, 240°=auditory
    ax.set_title("Group sensory space (θ vs r)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_hist(vec, out_png, title, bins=60):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(vec, bins=bins)
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

# --------------------------- CLI ---------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="HSV model postprocessing + visuals + comparison (fsaverage5).")
    ap.add_argument("--in-dir", required=True, help="Group dir containing <group>_rgba_all.npy")
    ap.add_argument("--group-name", required=True, help="Group label (e.g., autism or control)")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: <in-dir>)")
    # optional second group for comparison
    ap.add_argument("--in-dir2", help="Second group's dir for comparison")
    ap.add_argument("--group-name2", help="Second group label")
    ap.add_argument("--n-perm", type=int, default=5000, help="Permutations for cluster-size")
    ap.add_argument("--vert-thresh", type=float, default=0.20, help="Vertex threshold on stat (in [0,1] for unpaired)")
    # explicit mesh paths (recommended if autodetect fails)
    ap.add_argument("--lh-surf", help="Path to fsaverage5 lh.white (or lh.midthickness)")
    ap.add_argument("--rh-surf", help="Path to fsaverage5 rh.white (or rh.midthickness)")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir or args.in_dir)

    # ----- Load & compute group 1 -----
    g = args.group_name
    rgba_path = os.path.join(args.in_dir, f"{g}_rgba_all.npy")
    rgba = load_rgba_stack(rgba_path)

    th_ind, r_ind, th_grp, r_grp, rgb_grp = rgba_stack_to_group(rgba)

    # Save arrays
    np.save(os.path.join(out_dir, f"{g}_theta_individual.npy"), th_ind)
    np.save(os.path.join(out_dir, f"{g}_r2rank_individual.npy"), r_ind)
    np.save(os.path.join(out_dir, f"{g}_theta_group.npy"), th_grp)
    np.save(os.path.join(out_dir, f"{g}_r2rank_group.npy"), r_grp)
    np.save(os.path.join(out_dir, f"{g}_rgb_group.npy"), rgb_grp)

    # Save GIFTI maps
    save_scalar_gifti(th_grp, os.path.join(out_dir, f"lh.{g}_theta.func.gii"),
                              os.path.join(out_dir, f"rh.{g}_theta.func.gii"))
    save_scalar_gifti(r_grp,  os.path.join(out_dir, f"lh.{g}_r.func.gii"),
                              os.path.join(out_dir, f"rh.{g}_r.func.gii"))
    save_rgb_gifti(rgb_grp,   os.path.join(out_dir, f"lh.{g}_rgb.func.gii"),
                              os.path.join(out_dir, f"rh.{g}_rgb.func.gii"))

    # Quick visuals
    plot_polar_theta_r(th_grp, r_grp, os.path.join(out_dir, f"{g}_polar_theta_r.png"))
    plot_hist(th_grp, os.path.join(out_dir, f"{g}_theta_hist.png"), f"{g} θ (rad)")
    plot_hist(r_grp,  os.path.join(out_dir, f"{g}_r_hist.png"),     f"{g} r (ranked R²)")

    # ----- Optional comparison -----
    if args.in_dir2 and args.group_name2:
        # Load group 2 and make mesh once (explicit paths preferred)
        g2 = args.group_name2
        rgba2_path = os.path.join(args.in_dir2, f"{g2}_rgba_all.npy")
        rgba2 = load_rgba_stack(rgba2_path)
        th_ind2, _, th_grp2, _, _ = rgba_stack_to_group(rgba2)

        SA, V = th_ind.shape
        SB, V2 = th_ind2.shape
        if V != V2:
            raise ValueError("Vertex count mismatch between groups.")
        mesh = FS5Mesh(args.lh_surf, args.rh_surf)

        if SA == SB:
            # ----- paired design (sign-flip) -----
            theta_com = signed_circ_disp(th_ind, th_ind2)  # S x V
            np.save(os.path.join(out_dir, f"{g}_vs_{g2}_theta_com.npy"), theta_com)
            stat = theta_com.mean(axis=0)
            np.save(os.path.join(out_dir, f"{g}_vs_{g2}_theta_com_mean.npy"), stat)
            print(f"Running {args.n_perm} permutations (paired, threshold={args.vert_thresh})...")
            maxClus = perm_maxclus_paired(mesh, theta_com, args.vert_thresh, n_perm=args.n_perm, seed=0)
            stat_name = "theta_com_mean"
        else:
            # ----- unpaired design (label-shuffle) -----
            stat = angle_stat_unpaired(th_ind, th_ind2)  # V in [0,1]
            np.save(os.path.join(out_dir, f"{g}_vs_{g2}_stat_unpaired.npy"), stat)
            print(f"Running {args.n_perm} permutations (unpaired label-shuffle, threshold={args.vert_thresh})...")
            maxClus = perm_maxclus_unpaired(mesh, th_ind, th_ind2, args.vert_thresh, n_perm=args.n_perm, seed=0)
            stat_name = "stat_unpaired"

        np.save(os.path.join(out_dir, f"{g}_vs_{g2}_maxClus.npy"), maxClus)

        # Clusterize observed map and report if any cluster ≥ 95th percentile
        p95 = float(np.percentile(maxClus, 95))
        _, lsz, _, rsz = cluster_labeling_on_fs5(mesh, stat, args.vert_thresh, negate=False)
        obs_max = int(max(np.concatenate([lsz, rsz])) if (lsz.size + rsz.size) else 0)

        # Save thresholded stat as GIFTI for viewing
        thr = stat.copy()
        thr[thr < args.vert_thresh] = 0.0
        save_scalar_gifti(thr, os.path.join(out_dir, f"lh.{g}_vs_{g2}_{stat_name}_thr.func.gii"),
                               os.path.join(out_dir, f"rh.{g}_vs_{g2}_{stat_name}_thr.func.gii"))

        with open(os.path.join(out_dir, f"{g}_vs_{g2}_summary.json"), "w") as f:
            json.dump({
                "design": "paired" if SA == SB else "unpaired",
                "n_perm": int(args.n_perm),
                "vertex_threshold": float(args.vert_thresh),
                "null_p95_maxClusterSize": p95,
                "observed_maxClusterSize": obs_max,
                "observed_any_cluster_ge_p95": bool(obs_max >= p95),
                "groupA": {"name": g,  "n": int(SA)},
                "groupB": {"name": g2, "n": int(SB)}
            }, f, indent=2)

        print(f"Done. 95th percentile of null max cluster size: {p95:.1f} | observed max: {obs_max}")

if __name__ == "__main__":
    main()
