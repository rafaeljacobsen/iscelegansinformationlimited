"""Figure 1 — AFD response kernel K(τ).

Top row: per-worm K(τ) for the 21 worms in the 15wormtempvar* cohort.
  (a) nonparametric ridge fits + cross-worm mean (black).
  (b) 3-parameter Tsukada fits + cross-worm mean.
Bottom: convolution of each kernel with the temperature for one sample worm
        (15wormtempvar14), overlaid on the recorded AFD trace.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from kernel import fit_kernel, bin_to_1s, design_matrix

DT_RECORD = 0.20      # 5 fps recording
DT_BIN = 1.0          # bin to 1 second before fitting
WINDOW_S = 100.0
RIDGE = 1.0
TEMP_SMOOTH_S = 4.0   # Gaussian σ on temperature (preserves stimulus periods ≥ 60 s)
SAMPLE_WORM = "15wormtempvar14"

DATA = Path(__file__).resolve().parent / "data" / "afd_dataset.pkl"
OUT = Path(__file__).resolve().parent / "fig1_kernel.png"


def main():
    df = pd.read_pickle(DATA)
    df = df[df["kind"] == "signal"].copy()

    bin_k = int(round(DT_BIN / DT_RECORD))
    K_np_all, K_pa_all, params, binned = {}, {}, {}, {}
    for w, sub in df.groupby("name", sort=False):
        temp_raw = gaussian_filter1d(sub["temperature"].to_numpy(),
                                     sigma=TEMP_SMOOTH_S / DT_RECORD)
        temp = bin_to_1s(temp_raw, bin_k)
        afd = bin_to_1s(sub["afd"].to_numpy(), bin_k)
        temp_d = temp - temp.mean()
        afd_d = afd - afd.mean()
        tau, K_np, K_pa, p = fit_kernel(temp_d, afd_d,
                                        dt=DT_BIN, window_s=WINDOW_S, ridge=RIDGE)
        K_np_all[w] = K_np
        K_pa_all[w] = K_pa
        params[w] = p
        binned[w] = (temp_d, afd_d)

    K_np_mat = np.array(list(K_np_all.values()))
    K_pa_mat = np.array(list(K_pa_all.values()))
    P = np.array(list(params.values()))
    print(f"fitted {len(K_np_all)} worms")
    print(f"  λ  median {np.median(P[:, 0]):.4f}   IQR {np.percentile(P[:, 0], 75)-np.percentile(P[:, 0], 25):.4f}")
    print(f"  α₀ median {np.median(P[:, 1]):.4f}   IQR {np.percentile(P[:, 1], 75)-np.percentile(P[:, 1], 25):.4f}")
    print(f"  α₁ median {np.median(P[:, 2]):.4f}   IQR {np.percentile(P[:, 2], 75)-np.percentile(P[:, 2], 25):.4f}")

    # ---- assemble the figure: 2 top panels + 1 wide bottom panel ----
    fig = plt.figure(figsize=(13, 8), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8], hspace=0.35)

    for ax, mat, label in [
        (fig.add_subplot(gs[0, 0]), K_np_mat, "(a) nonparametric ridge"),
        (fig.add_subplot(gs[0, 1]), K_pa_mat, "(b) 3-parameter fit"),
    ]:
        for K in mat:
            ax.plot(tau, K, color="gray", lw=0.8, alpha=0.5)
        ax.plot(tau, mat.mean(axis=0), color="black", lw=2.4, label="mean")
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.set_xlabel("lag τ (s)")
        ax.set_ylabel("K(τ)")
        ax.set_title(label)
        ax.legend(fontsize=9, loc="upper right")

    # bottom panel: forward-prediction overlay for SAMPLE_WORM
    ax = fig.add_subplot(gs[1, :])
    temp_d, afd_d = binned[SAMPLE_WORM]
    n_lags = len(tau)
    Xfull = design_matrix(temp_d, n_lags)
    K_np = K_np_all[SAMPLE_WORM]
    K_pa = K_pa_all[SAMPLE_WORM]
    t_axis = np.arange(len(afd_d)) * DT_BIN
    ax.plot(t_axis, afd_d, color="green", lw=1.0, label="AFD trace")
    ax.plot(t_axis, Xfull @ K_pa, color="black", lw=1.2, alpha=0.9,
            label="parametric K · T")
    ax.plot(t_axis, Xfull @ K_np, color="tab:blue", lw=1.2, alpha=0.9,
            label="nonparametric K · T")
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔF/F (demeaned)")
    ax.set_title(f"Forward prediction on {SAMPLE_WORM}")
    ax.legend(fontsize=9, loc="upper right")

    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
