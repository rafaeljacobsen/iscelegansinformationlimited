"""Figure 4 — behavioral statistics of C. elegans locomotion.

(a) D_r from velocity autocorrelation of forward runs > 20 s, fit with a
    single exponential on lags 5–20 s.
(b) Sample reversal showing PCA arrows fit to the 1 mm before / 1 mm after
    the reversal.   Pinned to a fixed (cohort, wid, rev_idx) for byte-level
    reproducibility.
(c) Bar charts: mean forward / reverse interval duration and integrated path.
(d) Histogram of signed reorientation angle γ across all reversals.
"""
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

DATA = Path(__file__).resolve().parent / "data" / "wt_cache.h5"
OUT = Path(__file__).resolve().parent / "fig4_behavior.png"

F, R = 2, 1                              # cached state codes

# Panel (b) — pinned reversal
PIN_KEY = "C15_0105"
PIN_REV_IDX = 1

# Panel (a) — D_r fit window
MIN_RUN_S = 20.0
MAX_LAG_S = 30.0
SMOOTH_S = 3.0
FIT_LO, FIT_HI = 5.0, 20.0
SEED = 42


def acf_unbiased(v, max_lag):
    n = len(v)
    nfft = 1 << (2 * n - 1).bit_length()
    f_ = np.fft.rfft(v, n=nfft)
    full = np.fft.irfft(f_ * np.conj(f_), n=nfft)[:n]
    return (full / (n - np.arange(n)))[:max_lag + 1]


def intervals(state, target):
    in_s = (state == target).astype(int)
    diff = np.diff(in_s, prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def signed_angle(b, a):
    cross = b[0] * a[1] - b[1] * a[0]
    dot = b[0] * a[0] + b[1] * a[1]
    return float(np.degrees(np.arctan2(cross, dot)))


def panel_a(ax):
    rng = np.random.default_rng(SEED)
    with h5py.File(DATA, "r") as f:
        keys = sorted(f.keys())
        dt = float(np.median([f[k].attrs["dt"] for k in keys]))
        max_lag = int(round(MAX_LAG_S / dt))
        smooth_n = int(round(SMOOTH_S / dt))
        if smooth_n % 2 == 0:
            smooth_n += 1
        min_run_n = int(round(MIN_RUN_S / dt))
        ac_sum = np.zeros(max_lag + 1)
        n_used = 0
        for k in keys:
            grp = f[k]
            state = grp["state"][:]
            x = grp["x"][:].astype(float)
            y = grp["y"][:].astype(float)
            for s, e in intervals(state, F):
                if e - s < min_run_n + smooth_n + 10:
                    continue
                xs = savgol_filter(x[s:e], smooth_n, 3)
                ys = savgol_filter(y[s:e], smooth_n, 3)
                vx = np.diff(xs) / dt
                vy = np.diff(ys) / dt
                alpha = float(rng.uniform(0, 2 * np.pi))
                v_proj = vx * np.cos(alpha) + vy * np.sin(alpha)
                if len(v_proj) < max_lag + 1:
                    continue
                ac_sum += acf_unbiased(v_proj, max_lag)
                n_used += 1
    ac = ac_sum / n_used
    t = np.arange(max_lag + 1) * dt
    mask = (t >= FIT_LO) & (t <= FIT_HI) & (ac > 0)
    popt, _ = curve_fit(lambda tt, A, lam: A * np.exp(-lam * tt),
                        t[mask], ac[mask], p0=(ac[mask][0], 0.05),
                        maxfev=20000)
    A_fit, lam_fit = popt
    print(f"(a) D_r fit on {n_used} runs > {MIN_RUN_S} s")
    print(f"    λ_tot = {lam_fit:.4f} s⁻¹  →  D_r (2D) = {lam_fit:.4f} rad²/s")

    ax.plot(t, ac, color="#1f77b4", lw=1.4, label="data (forward runs > 20 s)")
    ax.plot(t[mask], A_fit * np.exp(-lam_fit * t[mask]), "k--", lw=1.4,
            label=fr"fit  $\lambda_{{\rm tot}}={lam_fit:.3f}$ s$^{{-1}}$")
    ax.axvspan(FIT_LO, FIT_HI, color="0.92", zorder=0, label="fit window")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"lag $\tau$ (s)")
    ax.set_ylabel(r"$\langle v_x(0)\,v_x(\tau)\rangle$  (mm/s)²")
    ax.set_xlim(0, MAX_LAG_S)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) D_r from velocity autocorrelation")


def panel_b(ax):
    with h5py.File(DATA, "r") as f:
        grp = f[PIN_KEY]
        x = grp["x"][:].astype(float)
        y = grp["y"][:].astype(float)
        state = grp["state"][:]
        rs = int(grp["rev_start"][PIN_REV_IDX])
        re = int(grp["rev_end"][PIN_REV_IDX])
        b = grp["rev_before"][PIN_REV_IDX]
        a = grp["rev_after"][PIN_REV_IDX]
        cohort = grp.attrs["cohort"]
        wid = int(grp.attrs["wid"])

    pad = 250
    s0, e0 = max(0, rs - pad), min(len(x), re + pad)
    fwd = state[s0:e0] == F
    rev = state[s0:e0] == R
    xs, ys = x[s0:e0], y[s0:e0]
    for mask, color in [(fwd, "#1f77b4"), (rev, "#d62728")]:
        seg_x, seg_y = [], []
        for k, m in enumerate(mask):
            if m:
                seg_x.append(xs[k]); seg_y.append(ys[k])
            else:
                if len(seg_x) > 1:
                    ax.plot(seg_x, seg_y, color=color, lw=1.4, alpha=0.85)
                seg_x, seg_y = [], []
        if seg_x:
            ax.plot(seg_x, seg_y, color=color, lw=1.4, alpha=0.85)

    ax.plot(x[rs], y[rs], "ko", ms=7)
    ax.plot(x[re], y[re], "ks", ms=7)
    ax.annotate("", xy=(b[2], b[3]), xytext=(b[0], b[1]),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.5,
                                mutation_scale=22))
    ax.annotate("", xy=(a[2], a[3]), xytext=(a[0], a[1]),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.5,
                                mutation_scale=22))
    ax.text(b[0], b[1], "  pre-tumble", color="#1f77b4", fontsize=10)
    ax.text(a[2], a[3], "  post-tumble", color="#2ca02c", fontsize=10)

    bvec = (b[2] - b[0], b[3] - b[1])
    avec = (a[2] - a[0], a[3] - a[1])
    dtheta = signed_angle(bvec, avec)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"(b) sample reversal  cohort {cohort} wid {wid}  "
                 fr"$\Delta\theta = {dtheta:+.1f}°$")


def panel_cd(ax_c, ax_d):
    fdur, fdist, rdur, rdist, reor = [], [], [], [], []
    with h5py.File(DATA, "r") as f:
        for k in sorted(f.keys()):
            grp = f[k]
            x = grp["x"][:]; y = grp["y"][:]
            state = grp["state"][:]
            dt = float(grp.attrs["dt"])
            rb = grp["rev_before"][:]
            ra = grp["rev_after"][:]
            step = np.hypot(np.diff(x), np.diff(y))
            for s, e in intervals(state, F):
                if e - s < 2:
                    continue
                fdur.append((e - s) * dt)
                fdist.append(float(step[s:e - 1].sum()))
            for s, e in intervals(state, R):
                if e - s < 2:
                    continue
                rdur.append((e - s) * dt)
                rdist.append(float(step[s:e - 1].sum()))
            for i in range(len(rb)):
                bvec = (rb[i, 2] - rb[i, 0], rb[i, 3] - rb[i, 1])
                avec = (ra[i, 2] - ra[i, 0], ra[i, 3] - ra[i, 1])
                reor.append(signed_angle(bvec, avec))
    fdur, fdist = np.array(fdur), np.array(fdist)
    rdur, rdist = np.array(rdur), np.array(rdist)
    reor = np.array(reor)
    alpha = float(np.cos(np.radians(reor)).mean())
    print(f"(c,d) F={len(fdur)}  R={len(rdur)}  reor={len(reor)}")
    print(f"      F dur {fdur.mean():.2f} s   R dur {rdur.mean():.2f} s")
    print(f"      F path {fdist.mean():.3f} mm   R path {rdist.mean():.3f} mm")
    print(f"      α = ⟨cos Δθ⟩ = {alpha:+.3f}")

    # (c) duration + path bars in one panel (twin axes)
    ax2 = ax_c.twinx()
    width = 0.35
    pos = np.arange(2)
    ax_c.bar(pos - width/2, [fdur.mean(), rdur.mean()],
             yerr=[fdur.std()/np.sqrt(len(fdur)), rdur.std()/np.sqrt(len(rdur))],
             width=width, color=["#1f77b4", "#d62728"], alpha=0.85,
             label="duration", capsize=5)
    ax2.bar(pos + width/2, [fdist.mean(), rdist.mean()],
            yerr=[fdist.std()/np.sqrt(len(fdist)), rdist.std()/np.sqrt(len(rdist))],
            width=width, color=["#aec7e8", "#ff9896"], alpha=0.85,
            label="path length", capsize=5)
    ax_c.set_xticks(pos)
    ax_c.set_xticklabels(["forward", "reverse"])
    ax_c.set_ylabel("duration (s)")
    ax2.set_ylabel("integrated path (mm)")
    ax_c.set_title("(c) F vs R duration + path")

    # (d) γ histogram
    bins = np.linspace(-np.pi, np.pi, 49)
    ax_d.hist(np.radians(reor), bins=bins, color="#444",
              edgecolor="white", alpha=0.85)
    ax_d.axvline(0, color="k", lw=0.7, ls="--")
    ax_d.set_xlim(-np.pi, np.pi)
    ax_d.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_d.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax_d.set_xlabel(r"reorientation $\gamma$")
    ax_d.set_ylabel("count")
    ax_d.set_title(fr"(d) reorientation Δθ   $\alpha = {alpha:+.3f}$")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=120)
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_cd(axes[1, 0], axes[1, 1])
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
