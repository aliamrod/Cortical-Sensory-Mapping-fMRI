#!/usr/bin/env python3
"""
CSV → BIDS (fMRIPrep-ready, safer version)

- Strictly uses CSV columns; no path guessing.
- Creates sub-*/[ses-*/]func/ only as needed (no fake sessions).
- Symlinks or copies *_bold.nii.gz if source exists.
- Sidecars:
    * If a source JSON sits next to the NIfTI, copy it.
    * Else create minimal JSON with TaskName + RepetitionTime (read from header).
- Optional mirror of events.tsv if present next to the source.
- Writes participants.tsv (one row per unique participant).
- Emits logs + subject lists + manifest + per-subject file counts.

CSV expected columns (rename as needed with flags):
  subject_id, session_id, run, site, diagnosis, age, sex, path_fmri
"""

import argparse, csv, json, os, re, sys
from pathlib import Path
import nibabel as nib
import pandas as pd

# ---------- helpers ----------
def digits_only(s: str) -> str:
    s = re.sub(r"[^0-9]", "", str(s or "").strip())
    return s

def norm_entity(prefix: str, val: str) -> str | None:
    v = str(val or "").strip()
    if not v: 
        return None
    v = re.sub(fr"(?i)^{prefix}-", "", v)   # strip existing prefix if present
    d = digits_only(v)
    return f"{prefix}-{d}" if d else None

def infer_task_from_path(path: str, default="rest") -> str:
    m = re.search(r"task-([A-Za-z0-9_]+)", os.path.basename(path or ""))
    return m.group(1) if m else default

def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try: dst.unlink()
        except FileNotFoundError: pass
    if use_symlink:
        os.symlink(src, dst)
    else:
        import shutil; shutil.copy2(src, dst)

def read_tr_from_header(nii_path: Path) -> float | None:
    try:
        img = nib.load(str(nii_path))
        tr = float(img.header.get_zooms()[-1])
        return float(tr) if tr and tr > 0 else None
    except Exception:
        return None

def write_minimal_json(dst_json: Path, task: str, tr: float | None):
    payload = {"TaskName": str(task)}
    if tr is not None:
        payload["RepetitionTime"] = round(float(tr), 6)
    dst_json.write_text(json.dumps(payload, indent=2) + "\n")

def maybe_copy_sidecar(src_nii: Path, dst_json: Path) -> bool:
    cand = src_nii.with_suffix("").with_suffix(".json")
    if cand.exists():
        # Copy verbatim, don’t validate content here
        import shutil; shutil.copy2(cand, dst_json)
        return True
    return False

def maybe_copy_events(src_nii: Path, dst_func_dir: Path, stem: str):
    src_events = src_nii.with_suffix("").with_suffix(".tsv").with_name(
        src_nii.with_suffix("").with_suffix(".tsv").name.replace("_bold.tsv", "_events.tsv")
    )
    if src_events.exists():
        import shutil; shutil.copy2(src_events, dst_func_dir / f"{stem}_events.tsv")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="CSV → BIDS (fMRIPrep-ready)")
    ap.add_argument("--csv", required=True, help="Master CSV with paths and metadata")
    ap.add_argument("--bids-dir", required=True, help="Output BIDS root")
    ap.add_argument("--logs-dir", default=None, help="Logs directory (default: <BIDS>/logs)")
    ap.add_argument("--use-symlinks", action="store_true", help="Use symlinks (default: copy)")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; do not write files")
    # Column names (override if your CSV headers differ)
    ap.add_argument("--col-subject", default="subject_id")
    ap.add_argument("--col-session", default="session_id")
    ap.add_argument("--col-run",     default="run")
    ap.add_argument("--col-site",    default="site")
    ap.add_argument("--col-dx",      default="diagnosis")
    ap.add_argument("--col-age",     default="age")
    ap.add_argument("--col-sex",     default="sex")
    ap.add_argument("--col-path",    default="path_fmri")
    ap.add_argument("--col-fail",    default="preprocessing_failed_fmriprep_stable")
    ap.add_argument("--dataset-name", default="Sensory Mapping BIDS")
    ap.add_argument("--bids-version", default="1.8.0")
    args = ap.parse_args()

    BIDS = Path(args.bids_dir); BIDS.mkdir(parents=True, exist_ok=True)
    LOGS = Path(args.logs_dir) if args.logs_dir else (BIDS / "logs"); LOGS.mkdir(parents=True, exist_ok=True)
    missing_log = LOGS / "missing_files.txt"
    if not args.dry_run:
        missing_log.write_text("")  # truncate

    # dataset_description.json (minimal but valid)
    dd = BIDS / "dataset_description.json"
    if not args.dry_run:
        dd.write_text(json.dumps(
            {"Name": args.dataset_name, "BIDSVersion": args.bids_version, "DatasetType": "raw"},
            indent=2
        ) + "\n")

    # participants.tsv header
    ptsv = BIDS / "participants.tsv"
    if not args.dry_run and not ptsv.exists():
        ptsv.write_text("participant_id\tsite\tdiagnosis\tage\tsex\n")

    # Counters
    rows_total=rows_skip=rows_keep=rows_with_path=rows_exist=files_linked=0
    subj_seen=set(); subj_dirs=set(); subj_with_bold=set(); missing_paths=[]

    # For manifest building
    manifest_rows=[]

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1

            # Skip explicit failures
            fail = str(row.get(args.col_fail, "") or "").strip().lower()
            if fail in {"failed", "true", "1", "yes", "fail"}:
                rows_skip += 1
                continue
            rows_keep += 1

            subj_raw = (row.get(args.col-subject) or row.get(args.col_subject) or "").strip()  # robustness
            if not subj_raw:
                continue

            participant_id = subj_raw if subj_raw.startswith("sub-") else f"sub-{subj_raw}"
            sess = norm_entity("ses", row.get(args.col_session, ""))
            run  = norm_entity("run", row.get(args.col_run, "")) or "run-1"
            site = (row.get(args.col_site, "") or "").strip() or "n/a"
            dx   = (row.get(args.col_dx, "") or "").strip() or "n/a"
            age  = (row.get(args.col_age, "") or "").strip() or "n/a"
            sex  = (row.get(args.col_sex, "") or "").strip() or "n/a"
            path = (row.get(args.col_path, "") or "").strip()

            # subject root (no fake ses if missing)
            subj_root = BIDS / participant_id
            func_dir  = (subj_root / sess / "func") if sess else (subj_root / "func")

            if participant_id not in subj_dirs and not args.dry_run:
                func_dir.mkdir(parents=True, exist_ok=True)
                subj_dirs.add(participant_id)

            if participant_id not in subj_seen and not args.dry_run:
                with open(ptsv, "a") as pf:
                    pf.write(f"{participant_id}\t{site}\t{dx}\t{age}\t{sex}\n")
                subj_seen.add(participant_id)

            if not path:
                continue
            rows_with_path += 1

            src = Path(path)
            task = infer_task_from_path(path)
            stem = f"{participant_id}_{(sess+'_') if sess else ''}task-{task}_{run}"
            dst_nii  = func_dir / f"{stem}_bold.nii.gz"
            dst_json = func_dir / f"{stem}_bold.json"

            if src.exists():
                rows_exist += 1
                manifest_rows.append({
                    "participant_id": participant_id,
                    "session": sess or "",
                    "task": task,
                    "run": run,
                    "src_path": str(src),
                    "dst_path": str(dst_nii),
                })
                if not args.dry_run:
                    link_or_copy(src, dst_nii, use_symlink=args.use_symlinks)
                    # Sidecar: copy if exists; else write minimal with TR
                    copied = maybe_copy_sidecar(src, dst_json)
                    if not copied:
                        tr = read_tr_from_header(src)
                        write_minimal_json(dst_json, task, tr)
                    # Optional: mirror events.tsv if present
                    maybe_copy_events(src, func_dir, stem)
                    files_linked += 1
                    subj_with_bold.add(participant_id)
            else:
                missing_paths.append(path)

    if not args.dry_run and missing_paths:
        with open(missing_log, "a") as mf:
            mf.write("\n".join(missing_paths) + "\n")

    # Subject lists for fMRIPrep
    swb = sorted(subj_with_bold)
    if not args.dry_run:
        (LOGS / "subjects_with_bold.txt").write_text("\n".join(swb) + ("\n" if swb else ""))
        (LOGS / "subjects_with_bold_noprefix.txt").write_text(
            "\n".join(s.replace("sub-","",1) for s in swb) + ("\n" if swb else "")
        )

    # Manifests & per-subject counts
    if not args.dry_run:
        man = pd.DataFrame(manifest_rows)
        if not man.empty:
            man.to_csv(BIDS / "manifest.csv", index=False)
        # per-subject counts (files actually present in tree)
        counts = []
        for sd in sorted([p for p in (BIDS).glob("sub-*") if p.is_dir()]):
            n = sum(1 for _ in sd.rglob("*") if _.is_file())
            counts.append({"participant_id": sd.name, "n_files": n})
        pd.DataFrame(counts).to_csv(BIDS / "per_subject_file_counts.csv", index=False)

    summary = {
        "rows_total": rows_total,
        "rows_skipped_failed": rows_skip,
        "rows_kept": rows_keep,
        "rows_with_path": rows_with_path,
        "rows_where_path_exists": rows_exist,
        "files_linked_or_copied": files_linked,
        "unique_subjects_seen": len(subj_seen),
        "unique_subjects_with_bold": len(subj_with_bold),
        "bids_dir": str(BIDS.resolve()),
        "logs_dir": str(LOGS.resolve()),
        "missing_log": str(missing_log.resolve()),
        "manifest_csv": str((BIDS / "manifest.csv").resolve()),
        "per_subject_counts_csv": str((BIDS / "per_subject_file_counts.csv").resolve()),
        "dry_run": args.dry_run,
        "mode": "symlink" if args.use_symlinks else "copy",
    }
    print(json.dumps(summary, indent=2))
    print("PROCESS COMPLETED.")

if __name__ == "__main__":
    sys.exit(main())
