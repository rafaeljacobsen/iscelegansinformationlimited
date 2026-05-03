"""Build data/ artifacts from external sources.

This script depends on data files that are NOT shipped in this folder:
  - /mnt/a/WBIbackup/{worm}_preprocessed.h5  (raw GCaMP/mNeptune fluorescence)
  - ../neuro1401final/randomwalkanalysis/wt_cache.h5  (already-built worm tracks)
  - ../neuro1401final/afd_proxy_dataset.pkl
  - ../neuro1401final/ramot_fig4_data.csv, ramot_figS2_data.csv

You only need to run this once (or after upstream data changes). Every
fig*.py + table1.py reads only from data/ and does not need this script.

Outputs into data/:
  - afd_dataset.pkl    : 21 signal + 5 control AFD traces, F0-normalized
  - wt_cache.h5        : copy of the Roberts et al. wt_cache
  - ramot_fig4.csv     : copy
  - ramot_figS2.csv    : copy
  - curvature_psd.npz  : per-worm Welch PSDs of head curvature
  - figS1_static.png   : copy of the existing Fig S1 PNG (not regenerable)
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

ORIG = HERE.parent / "neuro1401final"
sys.path.insert(0, str(ORIG))                      # so we can import build_afd_dataset, etc.
sys.path.insert(0, str(ORIG / "randomwalkanalysis"))


def build_afd_dataset():
    """Signal worms come straight from afd_proxy_dataset.pkl. Control worms
    are recomputed using the median-signal F0 (paper's noise definition)."""
    from f0_normalized import f0_normalize_controls

    df = pd.read_pickle(ORIG / "afd_proxy_dataset.pkl")
    df, _F0_med = f0_normalize_controls(df)

    out = pd.DataFrame({
        "name": df["name"],
        "kind": np.where(df["name"].str.contains("control", case=False),
                         "control", "signal"),
        "time": df["time"],
        "temperature": df["temperature"],
        "afd": df["afd"],
    })
    # Keep only the worms used in the paper: 21 signal (15wormtempvar*) + 5 controls.
    keep = (out["name"].str.startswith("15wormtempvar")
            | out["name"].str.contains("control", case=False))
    out = out[keep].reset_index(drop=True)

    out.to_pickle(DATA / "afd_dataset.pkl")
    n_sig = out[out["kind"] == "signal"]["name"].nunique()
    n_ctrl = out[out["kind"] == "control"]["name"].nunique()
    print(f"  afd_dataset.pkl  signal={n_sig}  control={n_ctrl}  rows={len(out)}")


def copy_static_files():
    pairs = [
        (ORIG / "randomwalkanalysis" / "wt_cache.h5", DATA / "wt_cache.h5"),
        (ORIG / "ramot_fig4_data.csv", DATA / "ramot_fig4.csv"),
        (ORIG / "ramot_figS2_data.csv", DATA / "ramot_figS2.csv"),
    ]
    for src, dst in pairs:
        shutil.copy2(src, dst)
        print(f"  {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")


def build_curvature_psd():
    """Welch PSDs of head curvature for every 15wormtempvar* worm."""
    import os
    import h5py
    from scipy.signal import welch
    from plot_curvature import pick_helper_nn, compute_curvature

    DT = 0.20
    FS = 1.0 / DT
    NPERSEG = 1024
    BASE = "/mnt/a/WBIbackup"

    df = pd.read_pickle(ORIG / "afd_proxy_dataset.pkl")
    worms = sorted(df[df["name"].str.startswith("15wormtempvar")]["name"].unique())

    freqs = None
    psds = []
    names = []
    for w in worms:
        prep = os.path.join(BASE, w, f"{w}_preprocessed.h5")
        if not os.path.exists(prep):
            continue
        with h5py.File(prep, "r") as h:
            key = pick_helper_nn(h)
            if key is None:
                continue
            nn = np.asarray(h[key])
        out = compute_curvature(nn)
        if out is None:
            continue
        curv = pd.Series(out[0]).interpolate(limit_direction="both").to_numpy()
        curv = curv - np.mean(curv)
        nps = min(NPERSEG, len(curv))
        f, P = welch(curv, fs=FS, nperseg=nps, detrend="linear")
        if freqs is None:
            freqs = f
            n_freqs = len(f)
        # different videos may have slightly different frequency grids when
        # they have fewer than NPERSEG samples; interpolate onto the first one.
        if len(f) != n_freqs:
            P = np.interp(freqs, f, P, left=np.nan, right=np.nan)
        psds.append(P)
        names.append(w)

    np.savez(DATA / "curvature_psd.npz",
             freqs=freqs, psds=np.array(psds), worm_names=np.array(names))
    print(f"  curvature_psd.npz  worms={len(names)}  freqs={len(freqs)}")


def copy_figS1():
    src = ORIG / "afd_complete.png"           # closest existing figure
    # Fig S1 in the paper is built from the WBI raw data with a frame +
    # heatmap + temp + velocity strip; we don't have a script for it. We
    # ship a static placeholder copied from the current project.
    if (ORIG / "one_worm_stack.png").exists():
        # If a more S1-like image exists, prefer that one.
        src = ORIG / "one_worm_stack.png"
    shutil.copy2(src, DATA / "figS1_static.png")
    print(f"  figS1_static.png  (copied from {src.name})")


def main():
    print("Building data/ from external sources...")
    build_afd_dataset()
    copy_static_files()
    build_curvature_psd()
    copy_figS1()
    total_mb = sum(p.stat().st_size for p in DATA.iterdir()) / 1e6
    print(f"\ndata/ total: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
