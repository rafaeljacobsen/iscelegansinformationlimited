"""Figure 2 — C. elegans locomotion.

(a) sample worm trajectory colored by forward (blue) / reverse (red) state.
    Picks the longest cached track in wt_cache.h5.

(b) cross-worm velocity autocorrelation V(t) = <v_x(0) v_x(t)>, fit with
    V(t) = A0·exp(-t/t0)
         + A1·exp(-t/t1)·cos(2π f1 t)
         + A2·exp(-t/t2)·cos(2π f2 t)
    The amplitudes are reparameterized so V(0) = <v_x²> by construction.
"""
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parent / "data" / "wt_cache.h5"
OUT = Path(__file__).resolve().parent / "fig2_locomotion.png"

F, R = 2, 1                          # cached state codes
SEED = 0
MAX_LAG_S = 30.0
FIT_LO, FIT_HI = 0.3, 25.0           # skip filter-dominated initial drop


def autocorr_biased(x, max_lag):
    """Biased ACF via FFT, returned for lags 0..max_lag."""
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F_ = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(np.abs(F_) ** 2, n=nfft)[:n]
    return (ac / (n - np.arange(n)))[:max_lag + 1]


def model(t, A_long, t_long, A1, t1, f1, A2, t2, f2):
    return (A_long * np.exp(-t / t_long)
            + A1 * np.exp(-t / t1) * np.cos(2 * np.pi * f1 * t)
            + A2 * np.exp(-t / t2) * np.cos(2 * np.pi * f2 * t))


def model_c0(C0):
    """Same model, with amplitudes reparameterized to sum to C0 at t=0."""
    def f(t, p_long, p1, t_long, t1, f1, t2, f2):
        p2 = 1.0 - p_long - p1
        return model(t,
                     p_long * C0, t_long,
                     p1 * C0, t1, f1,
                     p2 * C0, t2, f2)
    return f


def panel_a(ax):
    """Longest cached track, colored by F/R state."""
    with h5py.File(DATA, "r") as f:
        best = max(f.keys(), key=lambda k: int(f[k].attrs["n_frames"]))
        grp = f[best]
        x = grp["x"][:].astype(float)
        y = grp["y"][:].astype(float)
        state = grp["state"][:]
        cohort = grp.attrs["cohort"]
        wid = int(grp.attrs["wid"])
        n = len(x)
        dt = float(grp.attrs["dt"])

    x = x - x.mean()
    y = y - y.mean()
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    color_map = {F: "#1f77b4", R: "#d62728"}
    colors = [color_map[int(v)] for v in state[:-1]]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=0.8))
    ax.plot(x[0], y[0], "ko", ms=6, label="start")
    ax.plot(x[-1], y[-1], "kx", ms=8, mew=2, label="end")
    pad = 0.5
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    handles = [plt.Line2D([], [], color="#1f77b4", lw=2.5, label="forward"),
               plt.Line2D([], [], color="#d62728", lw=2.5, label="reverse"),
               plt.Line2D([], [], marker="o", color="k", lw=0, label="start"),
               plt.Line2D([], [], marker="x", color="k", lw=0, mew=2, label="end")]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax.set_title(f"(a) sample track  (cohort {cohort}, wid {wid}, "
                 f"{n*dt/60:.1f} min)")


def panel_b(ax):
    """Velocity ACF fit with exp + 2 damped cosines."""
    rng = np.random.default_rng(SEED)
    with h5py.File(DATA, "r") as f:
        keys = sorted(f.keys())
        dt = float(np.median([f[k].attrs["dt"] for k in keys]))
        max_lag = int(round(MAX_LAG_S / dt))
        ac_sum = np.zeros(max_lag + 1)
        ac_n = 0
        for k in keys:
            grp = f[k]
            v_signed = grp["v"][:].astype(np.float64)
            theta = grp["theta"][:].astype(np.float64)
            if len(v_signed) < max_lag + 1:
                continue
            vx = v_signed * np.cos(theta)
            vy = v_signed * np.sin(theta)
            alpha = float(rng.uniform(0, 2 * np.pi))
            v_proj = vx * np.cos(alpha) + vy * np.sin(alpha)
            ac_sum += autocorr_biased(v_proj, max_lag)
            ac_n += 1

    ac = ac_sum / ac_n
    lags = np.arange(max_lag + 1) * dt
    C0 = float(ac[0])
    print(f"V(0) = ⟨v_x²⟩ = {C0:.5f}  (mm/s)²  averaged over {ac_n} worms")

    fit_mask = (lags >= FIT_LO) & (lags <= FIT_HI)
    p, _ = curve_fit(model_c0(C0), lags[fit_mask], ac[fit_mask],
                     p0=[0.3, 0.4, 8.0, 1.5, 0.5, 4.0, 0.2],
                     bounds=([0.0, 0.0, 1.0, 0.1, 0.25, 0.1, 0.05],
                             [1.0, 1.0, 100.0, 10.0, 2.0, 30.0, 0.49]),
                     maxfev=80000)
    p_long, p1, t_long, t1, f1, t2, f2 = p
    p2 = 1.0 - p_long - p1
    A_long, A1, A2 = p_long * C0, p1 * C0, p2 * C0
    print(f"  exp + 2 damped osc:")
    print(f"    A_long={A_long:.5f}  t_long={t_long:.2f} s")
    print(f"    A1={A1:.5f}  t1={t1:.2f} s  f1={f1:.3f} Hz")
    print(f"    A2={A2:.5f}  t2={t2:.2f} s  f2={f2:.3f} Hz")

    ax.plot(lags, ac, color="0.55", lw=1.2, label="data")
    ax.scatter(lags[::15], ac[::15], s=12, color="0.25")
    ax.plot(lags, model(lags, A_long, t_long, A1, t1, f1, A2, t2, f2),
            color="#d62728", lw=2.4, label="fit")
    ax.axvspan(0, FIT_LO, color="0.92", zorder=0)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("lag t (s)")
    ax.set_ylabel(r"$V(t)$  (mm/s)²")
    ax.set_xlim(0, MAX_LAG_S)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("(b) velocity autocorrelation")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
