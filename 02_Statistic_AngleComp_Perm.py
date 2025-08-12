# -------- COPY-PASTE READY: same structure, no neuromaps, works on fsaverage5 --------
import os, sys
import numpy as np
import nibabel as nib
import pingouin as pg
import pycircstat as circ
import hcp_utils as hcp
import networkx as nx
from nibabel.freesurfer.io import read_geometry

# --- locate fsaverage5 surfaces (adjust or add paths if needed) ---
def _first_existing(paths):
    for p in paths:
        p2 = os.path.expandvars(p)
        if os.path.isfile(p2):
            return p2
    return None

lh_path = _first_existing([
    "$SUBJECTS_DIR/fsaverage5/surf/lh.white",
    "$FREESURFER_HOME/subjects/fsaverage5/surf/lh.white",
    "/mnt/lmod/software/freesurfer/freesurfer7/subjects/fsaverage5/surf/lh.white",
])
rh_path = _first_existing([
    "$SUBJECTS_DIR/fsaverage5/surf/rh.white",
    "$FREESURFER_HOME/subjects/fsaverage5/surf/rh.white",
    "/mnt/lmod/software/freesurfer/freesurfer7/subjects/fsaverage5/surf/rh.white",
])
if lh_path is None or rh_path is None:
    raise FileNotFoundError("fsaverage5 surfaces not found. Set SUBJECTS_DIR or FREESURFER_HOME.")

# --- load geometry (coords, faces) and set dynamic sizes (no 32492 hardcode) ---
lh_vert, lh_face = read_geometry(lh_path)  # lh_vert: (N_L, 3), lh_face: (F_L, 3)
rh_vert, rh_face = read_geometry(rh_path)  # rh_vert: (N_R, 3), rh_face: (F_R, 3)
NL, NR = lh_vert.shape[0], rh_vert.shape[0]

def v_cluster_labeling(data, mask, thres, neg = False):

    if mask is not None:
        data_lr = np.zeros(NL + NR, dtype=float)
        data_lr[np.asarray(mask).astype(bool)] = data
    else:
        data_lr = np.asarray(data, dtype=float).copy()

    if neg:
        data_lr[data_lr > thres] = 0
    else:
        data_lr[data_lr < thres] = 0
    
    lr_mask = np.array(data_lr!=0, dtype=int)
    lh_mask = lr_mask[:NL]
    rh_mask = lr_mask[NL:]

    # left hemis 
    lh_mask_indices = np.where(lh_mask==1)[0]
    lh_mask_vertice = lh_vert[lh_mask_indices,:]

    G = nx.Graph()
    for i, vertex in enumerate(lh_vert):
        if i in lh_mask_indices:
            G.add_node(i, coords=vertex)
    for face in lh_face.astype(int):
        if all(v in lh_mask_indices for v in face):
            G.add_edge(face[0], face[1])
            G.add_edge(face[1], face[2])
            G.add_edge(face[2], face[0])

    ccsL  = nx.connected_components(G) 
    lh_clus = []
    lh_clusize = []
    for ccl in ccsL:
        lh_clus.append(list(ccl))
        lh_clusize.append(len(ccl))
    lh_clusize = np.asarray(lh_clusize)

    # right hemis 
    rh_mask_indices = np.where(rh_mask==1)[0]
    rh_mask_vertice = rh_vert[rh_mask_indices,:]

    G = nx.Graph()
    for i, vertex in enumerate(rh_vert):
        if i in rh_mask_indices:
            G.add_node(i, coords=vertex)
    for face in rh_face.astype(int):
        if all(v in rh_mask_indices for v in face):
            G.add_edge(face[0], face[1])
            G.add_edge(face[1], face[2])
            G.add_edge(face[2], face[0])

    ccsR  = nx.connected_components(G) 
    rh_clus = []
    rh_clusize = []
    for ccr in ccsR:
        rh_clus.append(list(ccr))
        rh_clusize.append(len(ccr))
    rh_clusize = np.asarray(rh_clusize)   

    return lh_clus, lh_clusize, rh_clus, rh_clusize


def v_vector_comp(mat1, mat2):
    #   calculating the variance of two angles and assign a sign to the variance
    ##  mat1 (and mat2): subjects x vertices
    ### input is the angle transformed from sensory betas
    
    theta_dif = circ.cdiff(mat1, mat2)
    theta_dif_sign = np.zeros((np.shape(theta_dif)))
    theta_dif_sign[theta_dif>0] = 1   
    theta_dif_sign[theta_dif<0] = -1

    theta_com = []
    for s in range(mat1.shape[0]):
        theta_pack = np.vstack((mat1[s,:], mat2[s,:]))
        theta_diff = 1 - pg.circ_r(theta_pack, axis=0)
        theta_com.append(theta_diff)
    theta_com = np.asarray(theta_com)
    theta_com = theta_com * theta_dif_sign

    return theta_com

def v_perm_clusize(theta_com, vertP_thres, n_perm):
    # sign-flip across subjects, cluster on the reduced (mean) map each perm
    n_sub = theta_com.shape[0]
    n_ver = theta_com.shape[1]

    maxClus = np.zeros(n_perm)
    for n in range(n_perm):
        rands = np.random.randint(2, size=n_sub)
        signs = np.where(rands==1, 1.0, -1.0)[:, None]
        stat = (theta_com * signs).mean(axis=0)  # (n_ver,)
        lh_clus, lh_clusize, rh_clus, rh_clusize = v_cluster_labeling(stat, None, vertP_thres, neg = False) 
        all_sizes = np.concatenate((lh_clusize, rh_clusize)) if (lh_clusize.size + rh_clusize.size) else np.array([0])
        maxClus[n] = np.max(all_sizes)

    return maxClus
# ================== DRIVER / MAIN ================== #
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="pixdim[1,2,3] should be non-zero")

    # If you already created theta_com above, we’ll use it.
    # Otherwise we make a tiny smoke test so the script produces output.
    try:
        theta_com  # is it defined earlier?
    except NameError:
        # ---- smoke test (safe) ----
        # Make a small synthetic signal on the mesh so clustering/perms run
        n_sub = 12
        n_vert = NL + NR
        rng = np.random.default_rng(0)
        theta_com = rng.normal(0, 0.02, size=(n_sub, n_vert))
        # plant a contiguous LH cluster above threshold
        seed = NL // 2
        theta_com += 0.30 * (np.arange(n_vert) >= seed) * (np.arange(n_vert) < seed + 200)

    # Permutation settings
    vertP_thres = 0.20   # adjust to your stat scale
    n_perm = 1000

    maxClus = v_perm_clusize(theta_com, vertP_thres, n_perm)
    np.save("maxClus.npy", maxClus)

    print(f"[OK] NL={NL} NR={NR}  n_perm={n_perm}  thres={vertP_thres}")
    print(f"Max cluster size (observed over perms): {int(maxClus.max())}")
    print(f"First 10: {maxClus[:10].astype(int)}")
    print("Saved: maxClus.npy")
