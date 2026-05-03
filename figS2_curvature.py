"""Figure S2 — Welch PSD of head-bend curvature for the 21 signal worms.

Reads the per-worm PSDs that build_data.py precomputed from the WBI raw
data.  Plots each worm as a thin line (colored by index) plus the median
across worms in bold black.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data" / "curvature_psd.npz"
OUT = Path(__file__).resolve().parent / "figS2_curvature.png"


def main():
    z = np.load(DATA, allow_pickle=True)
    freqs = z["freqs"]
    psds = z["psds"]
    names = z["worm_names"]
    n = len(names)
    print(f"loaded {n} per-worm PSDs ({len(freqs)} freq bins)")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    cmap = plt.cm.viridis
    for i, (w, P) in enumerate(zip(names, psds)):
        ax.loglog(freqs[1:], P[1:], color=cmap(i / max(n - 1, 1)),
                  lw=0.7, alpha=0.4)
    median = np.nanmedian(psds, axis=0)
    ax.loglog(freqs[1:], median[1:], color="black", lw=2.0,
              label="median across worms")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel(r"PSD (curv$^2$ / Hz)")
    ax.set_title("Welch PSD of head curvature")
    ax.grid(True, which="both", alpha=0.3, lw=0.4)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
