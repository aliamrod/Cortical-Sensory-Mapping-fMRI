# Last Update: 16 March, 2026
"""
Objective: Perform core feature extraction required:
[1] Load vertex-wise surface time series.
[2] Compute V1/S1/A1 seed time series from the HCP-MMP atlas.
[3] Run positive linear regression at each vertex.
[4] Save rgba = [beta_v1, beta_s1, beta_a1, explained_variance].
[5] Optionally convert to HSV/theta/rd for downstream analysis.
"""

#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import nibabel as nib
import pandas as pd
import scipy.stats as ss
import matplotlib.colors as color

from sklearn.linear_model import LinearRegression
from nibabel.freesurfer.io import read_annot


# ---------------------------------------------------------------------
# Constants: fsaverage5
# ---------------------------------------------------------------------
N_LH = 10242
N_RH = 10242
N_VERT = 20484


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def sanitize_id(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def row_is_ok(row: pd.Series) -> bool:
    prep = str(row.get("preprocessing_failed_fmriprep_stable", "")).strip()
    surf = str(row.get("surface_status", "")).strip()
    return (prep in ("", "OK")) and (surf in ("", "OK"))


def safe_rank01(x: np.ndarray) -> np.ndarray:
    r = ss.rankdata(x)
    denom = r.max() - r.min()
    if denom == 0:
        return np.zeros_like(r, dtype=np.float32)
    return ((r - r.min()) / denom).astype(np.float32)


def zscore_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score each row across time.
    x shape: (n_rows, T)
    """
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Load GIFTI timeseries
# ---------------------------------------------------------------------
def load_gifti_timeseries(path: str, expected_vertices: int = N_LH) -> np.ndarray:
    """
    Load .func.gii and return (V, T).
    Robust to (T, V) or (V, T).
    """
    img = nib.load(path)
    data = np.asarray(img.agg_data())

    if data.ndim != 2:
        raise ValueError(f"{path}: expected 2D data, got {data.shape}")

    if data.shape[0] == expected_vertices and data.shape[1] != expected_vertices:
        vt = data
    elif data.shape[1] == expected_vertices and data.shape[0] != expected_vertices:
        vt = data.T
    else:
        if data.shape[0] == data.shape[1]:
            raise ValueError(f"{path}: ambiguous square data {data.shape}")
        vt = data if data.shape[0] > data.shape[1] else data.T

    if vt.shape[0] != expected_vertices:
        raise ValueError(
            f"{path}: expected {expected_vertices} vertices, got {vt.shape}"
        )

    vt = vt.astype(np.float32, copy=False)
    vt[~np.isfinite(vt)] = 0.0
    return vt


# ---------------------------------------------------------------------
# Atlas loading (fsaverage5 HCP-MMP1 annot)
# ---------------------------------------------------------------------
def load_parc_fsavg5(templates_dir: str) -> np.ndarray:
    """
    Load combined fsaverage5 parcel vector of length 20484.

    LH parcels remain as-is.
    RH parcels are offset by +180 for non-zero labels:
      LH: 1..180
      RH: 181..360
    """
    lh_annot = os.path.join(templates_dir, "lh.HCP-MMP1.fsaverage5.annot")
    rh_annot = os.path.join(templates_dir, "rh.HCP-MMP1.fsaverage5.annot")

    if not os.path.exists(lh_annot):
        raise FileNotFoundError(lh_annot)
    if not os.path.exists(rh_annot):
        raise FileNotFoundError(rh_annot)

    lh_lab, _, _ = read_annot(lh_annot)
    rh_lab, _, _ = read_annot(rh_annot)

    if lh_lab.shape[0] != N_LH or rh_lab.shape[0] != N_RH:
        raise RuntimeError(
            f"Annot size mismatch: lh={lh_lab.shape[0]}, rh={rh_lab.shape[0]}, expected {N_LH}/{N_RH}"
        )

    rh_off = rh_lab.copy()
    rh_off[rh_off != 0] += 180

    parc = np.concatenate([lh_lab, rh_off]).astype(np.int64, copy=False)

    if parc.shape[0] != N_VERT:
        raise RuntimeError(f"Expected combined parcel len {N_VERT}, got {parc.shape[0]}")

    return parc


# ---------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------
def v_ts_nnls_vertex(ts: np.ndarray, parc: np.ndarray) -> np.ndarray:
    """
    ts: (20484, T)
    parc: (20484,)

    Returns rgba: (4, 20484)
      0 = beta_V1
      1 = beta_S1
      2 = beta_A1
      3 = explained variance ratio
    """
    if ts.ndim != 2:
        raise ValueError(f"ts must be 2D, got {ts.shape}")
    if ts.shape[0] != N_VERT:
        raise ValueError(f"Expected {N_VERT} vertices, got {ts.shape[0]}")
    if parc.shape != (N_VERT,):
        raise ValueError(f"Expected parc shape {(N_VERT,)}, got {parc.shape}")
    if ts.shape[1] < 5:
        raise ValueError(f"Too few timepoints: T={ts.shape[1]}")

    ts = np.asarray(ts, dtype=np.float32)
    ts[~np.isfinite(ts)] = 0.0

    # standardize per vertex across time
    ts_z = zscore_rows(ts)

    # Reference-paper ROI definitions
    v1_mask = (parc == 1) | (parc == 181)
    s1_mask = (
        (parc == 9) | (parc == 51) | (parc == 52) | (parc == 53) |
        (parc == 189) | (parc == 231) | (parc == 232) | (parc == 233)
    )
    a1_mask = (parc == 24) | (parc == 204)

    if v1_mask.sum() == 0:
        raise RuntimeError("V1 mask empty")
    if s1_mask.sum() == 0:
        raise RuntimeError("S1 mask empty")
    if a1_mask.sum() == 0:
        raise RuntimeError("A1 mask empty")

    ts_v1 = ts_z[v1_mask, :].mean(axis=0)
    ts_s1 = ts_z[s1_mask, :].mean(axis=0)
    ts_a1 = ts_z[a1_mask, :].mean(axis=0)

    # predictors T x 3
    X = np.vstack((ts_v1, ts_s1, ts_a1)).T.astype(np.float32, copy=False)
    # standardize predictors too
    X = zscore_rows(X.T).T

    Y = ts_z.T  # T x V

    reg = LinearRegression(positive=True, fit_intercept=False)
    res = reg.fit(X, Y)

    rgba = np.zeros((4, ts.shape[0]), dtype=np.float32)
    rgba[:3, :] = res.coef_.T.astype(np.float32, copy=False)

    y_bar = Y.mean(axis=0)
    y_hat = res.predict(X)

    ss_total = np.sum((Y - y_bar) ** 2, axis=0)
    ss_exp = np.sum((y_hat - y_bar) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ve = ss_exp / ss_total
        ve[~np.isfinite(ve)] = 0.0

    rgba[3, :] = ve.astype(np.float32, copy=False)
    return rgba


def v_rgba2hue(rgba: np.ndarray) -> np.ndarray:
    rgbv = rgba[:3, :]
    H = np.zeros(rgbv.shape[1], dtype=np.float32)

    delta = np.max(rgbv, axis=0) - np.min(rgbv, axis=0)
    del_0 = delta == 0
    H[del_0] = 0.0

    rgbv_pos = rgbv[:, ~del_0]
    del_pos = delta[~del_0]
    ind_max = np.argmax(rgbv_pos, axis=0)
    Hp = np.zeros(rgbv_pos.shape[1], dtype=np.float32)

    m0 = ind_max == 0
    m1 = ind_max == 1
    m2 = ind_max == 2

    Hp[m0] = 60 * ((rgbv_pos[1, m0] - rgbv_pos[2, m0]) / del_pos[m0])
    Hp[m1] = 60 * (((rgbv_pos[2, m1] - rgbv_pos[0, m1]) / del_pos[m1]) + 2)
    Hp[m2] = 60 * (((rgbv_pos[0, m2] - rgbv_pos[1, m2]) / del_pos[m2]) + 4)

    H[~del_0] = Hp
    H = np.mod(H, 360.0)
    return H.astype(np.float32)


def v_hsv_model_rgba(rgba: np.ndarray):
    hue = v_rgba2hue(rgba)

    rd = rgba[3, :]
    rd = safe_rank01(rd)

    hsv = np.zeros((rgba.shape[1], 3), dtype=np.float32)
    hsv[:, 0] = hue / 360.0
    hsv[:, 1] = rd
    hsv[:, 2] = 0.86

    rgb = color.hsv_to_rgb(hsv).astype(np.float32)
    theta = (2.0 * np.pi * hue / 360.0).astype(np.float32)

    return hsv, rgb, theta, rd


def summarize_rgba(rgba: np.ndarray) -> Dict[str, float]:
    ve = rgba[3, :]
    dom = np.argmax(rgba[:3, :], axis=0)

    return {
        "mean_ve": float(np.mean(ve)),
        "median_ve": float(np.median(ve)),
        "p90_ve": float(np.percentile(ve, 90)),
        "frac_ve_gt_0": float(np.mean(ve > 0)),
        "frac_dom_v1": float(np.mean(dom == 0)),
        "frac_dom_s1": float(np.mean(dom == 1)),
        "frac_dom_a1": float(np.mean(dom == 2)),
    }


# ---------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------
@dataclass
class RowResult:
    row_idx: int
    rid: str
    ok: bool
    err: Optional[str] = None
    rgba_path: Optional[str] = None
    hsv_path: Optional[str] = None
    rgb_path: Optional[str] = None
    theta_path: Optional[str] = None
    rd_path: Optional[str] = None
    n_timepoints: Optional[int] = None
    mean_ve: Optional[float] = None
    median_ve: Optional[float] = None
    p90_ve: Optional[float] = None
    frac_ve_gt_0: Optional[float] = None
    frac_dom_v1: Optional[float] = None
    frac_dom_s1: Optional[float] = None
    frac_dom_a1: Optional[float] = None


def process_row(
    row: pd.Series,
    row_idx: int,
    parc: np.ndarray,
    outdir: str,
    require_ok: bool,
    write_hsv: bool,
    skip_existing: bool,
) -> RowResult:
    rid = sanitize_id(str(row["uid2"]))

    rgba_path = os.path.join(outdir, f"{rid}_rgba.npy")
    hsv_path = os.path.join(outdir, f"{rid}_hsv.npy")
    rgb_path = os.path.join(outdir, f"{rid}_rgb.npy")
    theta_path = os.path.join(outdir, f"{rid}_theta.npy")
    rd_path = os.path.join(outdir, f"{rid}_rd.npy")

    try:
        if require_ok and not row_is_ok(row):
            raise RuntimeError("QC failed")

        if skip_existing and os.path.exists(rgba_path):
            return RowResult(
                row_idx=row_idx,
                rid=rid,
                ok=True,
                rgba_path=rgba_path,
                hsv_path=hsv_path if os.path.exists(hsv_path) else None,
                rgb_path=rgb_path if os.path.exists(rgb_path) else None,
                theta_path=theta_path if os.path.exists(theta_path) else None,
                rd_path=rd_path if os.path.exists(rd_path) else None,
            )

        path_L = str(row["path_bold_L"])
        path_R = str(row["path_bold_R"])

        if not os.path.exists(path_L):
            raise FileNotFoundError(f"Missing left GIFTI: {path_L}")
        if not os.path.exists(path_R):
            raise FileNotFoundError(f"Missing right GIFTI: {path_R}")

        ts_L = load_gifti_timeseries(path_L, expected_vertices=N_LH)
        ts_R = load_gifti_timeseries(path_R, expected_vertices=N_RH)

        if ts_L.shape[1] != ts_R.shape[1]:
            raise ValueError(f"L/R timepoint mismatch: {ts_L.shape} vs {ts_R.shape}")

        ts = np.vstack((ts_L, ts_R))  # (20484, T)
        rgba = v_ts_nnls_vertex(ts, parc)
        stats = summarize_rgba(rgba)

        os.makedirs(outdir, exist_ok=True)
        np.save(rgba_path, rgba)

        if write_hsv:
            hsv, rgb, theta, rd = v_hsv_model_rgba(rgba)
            np.save(hsv_path, hsv)
            np.save(rgb_path, rgb)
            np.save(theta_path, theta)
            np.save(rd_path, rd)

        return RowResult(
            row_idx=row_idx,
            rid=rid,
            ok=True,
            rgba_path=rgba_path,
            hsv_path=hsv_path if write_hsv else None,
            rgb_path=rgb_path if write_hsv else None,
            theta_path=theta_path if write_hsv else None,
            rd_path=rd_path if write_hsv else None,
            n_timepoints=int(ts.shape[1]),
            mean_ve=stats["mean_ve"],
            median_ve=stats["median_ve"],
            p90_ve=stats["p90_ve"],
            frac_ve_gt_0=stats["frac_ve_gt_0"],
            frac_dom_v1=stats["frac_dom_v1"],
            frac_dom_s1=stats["frac_dom_s1"],
            frac_dom_a1=stats["frac_dom_a1"],
        )

    except Exception as e:
        return RowResult(
            row_idx=row_idx,
            rid=rid,
            ok=False,
            err=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------
def compute_row_range(n: int, args) -> Tuple[int, int]:
    if args.start_row is not None or args.stop_row is not None:
        start = 0 if args.start_row is None else int(args.start_row)
        stop = n if args.stop_row is None else int(args.stop_row)
    else:
        chunk_idx = args.chunk_idx
        if chunk_idx is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env is None:
                raise RuntimeError("No --chunk-idx and SLURM_ARRAY_TASK_ID not set")
            chunk_idx = int(env)

        if args.chunk_size <= 0:
            raise ValueError("--chunk-size must be > 0")

        start = chunk_idx * args.chunk_size
        stop = min(n, start + args.chunk_size)

    if start < 0 or start >= n:
        raise IndexError(f"start={start} out of range for n={n}")
    if stop <= start or stop > n:
        raise IndexError(f"stop={stop} invalid for n={n}")

    return start, stop


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute fsaverage5 sensory features for normative modeling"
    )
    ap.add_argument("--master-csv", required=True)
    ap.add_argument("--templates-dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-idx", type=int, default=None)
    ap.add_argument("--start-row", type=int, default=None)
    ap.add_argument("--stop-row", type=int, default=None)

    ap.add_argument("--require-ok", action="store_true")
    ap.add_argument("--write-hsv", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.master_csv)
    ensure_columns(
    df,
    ["uid2", "subject_id", "session_id", "run", "path_bold_L", "path_bold_R"]
)

    if len(df) == 0:
        raise ValueError("Empty manifest")

    start, stop = compute_row_range(len(df), args)
    os.makedirs(args.outdir, exist_ok=True)

    log(f"Loading fsaverage5 annot from {args.templates_dir}")
    parc = load_parc_fsavg5(args.templates_dir)

    chunk_tag = f"rows_{start:06d}-{stop-1:06d}"
    results_tsv = os.path.join(args.outdir, f"{chunk_tag}.results.tsv")
    summary_json = os.path.join(args.outdir, f"{chunk_tag}.summary.json")

    log(f"Processing rows [{start}, {stop}) / {len(df)}")
    results: List[RowResult] = []
    n_ok = 0
    n_fail = 0

    for row_idx in range(start, stop):
        row = df.iloc[row_idx]
        res = process_row(
            row=row,
            row_idx=row_idx,
            parc=parc,
            outdir=args.outdir,
            require_ok=args.require_ok,
            write_hsv=args.write_hsv,
            skip_existing=args.skip_existing,
        )
        results.append(res)

        if res.ok:
            n_ok += 1
            log(f"[OK] row={row_idx} rid={res.rid}")
        else:
            n_fail += 1
            log(f"[FAIL] row={row_idx} rid={res.rid} err={res.err}")
            if args.fail_fast:
                break

    pd.DataFrame([asdict(r) for r in results]).to_csv(results_tsv, sep="\t", index=False)

    summary = {
        "master_csv": args.master_csv,
        "templates_dir": args.templates_dir,
        "outdir": args.outdir,
        "start_row": start,
        "stop_row": stop,
        "n_total_in_chunk": len(results),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "require_ok": bool(args.require_ok),
        "write_hsv": bool(args.write_hsv),
        "skip_existing": bool(args.skip_existing),
        "results_tsv": results_tsv,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Done {chunk_tag}: ok={n_ok}, fail={n_fail}")
    sys.exit(2 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()


"""
Final dataset contains per-run sensory maps. 

So, subject-level features containing
[1] mean sensory angle
[2] variance of sensory angle
[3] distribution of V1/S1/A1 dominance
[4] mean variance explained

Next, vertex-wise normative modeling is used to predict:

theta (vertex) ~ age + sex + site

Which is then used to compute:
z = (raw observed scoring - predicted) / model_variance
"""

