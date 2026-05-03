"""Figure S3 — noise measurement from 5 control AFD recordings.

(a) Mean autocorrelation across the 5 controls + single-exponential fit.
    Yields τ_n and σ_n² = D_n · τ_n (Table 1).
(b) Stack of the 5 control ΔF/F traces themselves.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parent / "data" / "afd_dataset.pkl"
OUT = Path(__file__).resolve().parent / "figS3_noise.png"

DT = 0.20
MAX_LAG_S = 60.0


def linear_detrend(x):
    t = np.arange(len(x), dtype=float)
    m, b = np.polyfit(t, x, 1)
    return x - (m * t + b)


def biased_acf(x, max_lag_samples):
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")
    return full[len(x) - 1: len(x) - 1 + max_lag_samples + 1] / len(x)


def main():
    df = pd.read_pickle(DATA)
    controls = sorted(df[df["kind"] == "control"]["name"].unique())
    n = len(controls)
    max_lag = int(round(MAX_LAG_S / DT))
    tau = np.arange(max_lag + 1) * DT

    # --- panel a: mean ACF + exp fit ---
    acfs = [biased_acf(linear_detrend(df.loc[df["name"] == w, "afd"].to_numpy()),
                       max_lag) for w in controls]
    c_mean = np.mean(acfs, axis=0)

    # Fit exp on positive values (truncate at first zero crossing).
    zc = np.where(c_mean <= 0)[0]
    t_max = tau[zc[0] - 1] if len(zc) > 0 and zc[0] > 2 else tau[-1]
    mask = (tau <= t_max) & (c_mean > 0)
    popt, _ = curve_fit(lambda tau_, C0, tc: C0 * np.exp(-tau_ / tc),
                        tau[mask], c_mean[mask],
                        p0=[c_mean[0], 20.0], bounds=([0, 1e-3], [np.inf, 1e4]),
                        maxfev=4000)
    sigma2_n, tau_n = popt
    Dn = sigma2_n / tau_n
    yhat = sigma2_n * np.exp(-tau[mask] / tau_n)
    r2 = 1 - np.var(c_mean[mask] - yhat) / np.var(c_mean[mask])
    print(f"OU fit on mean ACF over {n} controls:")
    print(f"  σ_n² = {sigma2_n:.5f}   τ_n = {tau_n:.2f} s   "
          f"D_n = {Dn:.5e}   R² = {r2:.3f}")

    # --- assemble figure (1 + n rows, panel a on top) ---
    fig = plt.figure(figsize=(11, 3 + 1.6 * n), dpi=110)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[3.0] + [1.0] * n)

    ax = fig.add_subplot(gs[0])
    ax.plot(tau, c_mean, color="darkgreen", lw=1.6, label="mean ACF")
    ax.plot(tau[mask], sigma2_n * np.exp(-tau[mask] / tau_n),
            "k--", lw=1.6, label=fr"exp fit  $R^2={r2:.3f}$")
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    ax.set_xlabel(r"lag $\tau$ (s)")
    ax.set_ylabel(r"$\langle\mathrm{AFD}(t)\,\mathrm{AFD}(0)\rangle$")
    ax.set_xlim(0, MAX_LAG_S)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_title("(a) noise autocorrelation + OU fit")

    # --- panel b: 5 raw traces stacked ---
    for i, w in enumerate(controls):
        ax = fig.add_subplot(gs[i + 1])
        sub = df[df["name"] == w]
        t = sub["time"].to_numpy()
        a = sub["afd"].to_numpy()
        ax.plot(t, a, color="darkgreen", lw=0.8)
        ax.set_ylabel("ΔF/F")
        ax.set_title(w, fontsize=9)
        ax.grid(True, alpha=0.25)
    ax.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
