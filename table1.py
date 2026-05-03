"""Table 1 — print every fitted parameter from the paper.

Runs the four fits end-to-end:
  1. AFD kernel: (λ, α₀, α₁) — per-worm median, 21 worms
  2. Noise OU:   (D_n, τ_n, σ_n²) — fit on mean ACF of 5 controls
  3. Velocity:   (A0, t0, A1, t1, f1, A2, t2, f2) — exp + 2 damped cosines
  4. Behavior:   (D_r, α, λ_R0, λ_T) from forward / reverse statistics
"""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from kernel import fit_kernel, bin_to_1s

HERE = Path(__file__).resolve().parent
AFD = HERE / "data" / "afd_dataset.pkl"
WT = HERE / "data" / "wt_cache.h5"

DT_RECORD = 0.20
DT_BIN = 1.0
WINDOW_S = 100.0
RIDGE = 1.0
TEMP_SMOOTH_S = 4.0

F = 2

# velocity-ACF settings (must match fig2_locomotion.py)
SEED = 0
MAX_LAG_S_VEL = 30.0
FIT_LO_VEL, FIT_HI_VEL = 0.3, 25.0

# D_r settings (must match fig4_behavior.py)
MIN_RUN_S = 20.0
MAX_LAG_S_DR = 30.0
SMOOTH_S_DR = 3.0
FIT_LO_DR, FIT_HI_DR = 5.0, 20.0
SEED_DR = 42


# ---------- helpers ----------
def autocorr_biased(x, max_lag):
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F_ = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(np.abs(F_) ** 2, n=nfft)[:n]
    return (ac / (n - np.arange(n)))[:max_lag + 1]


def acf_unbiased(v, max_lag):
    n = len(v)
    nfft = 1 << (2 * n - 1).bit_length()
    f_ = np.fft.rfft(v, n=nfft)
    full = np.fft.irfft(f_ * np.conj(f_), n=nfft)[:n]
    return (full / (n - np.arange(n)))[:max_lag + 1]


def linear_detrend(x):
    t = np.arange(len(x), dtype=float)
    m, b = np.polyfit(t, x, 1)
    return x - (m * t + b)


# ---------- 1. AFD kernel parameters ----------
def kernel_params(df_signal):
    bin_k = int(round(DT_BIN / DT_RECORD))
    P = []
    for _w, sub in df_signal.groupby("name", sort=False):
        temp = bin_to_1s(gaussian_filter1d(sub["temperature"].to_numpy(),
                                           TEMP_SMOOTH_S / DT_RECORD), bin_k)
        afd = bin_to_1s(sub["afd"].to_numpy(), bin_k)
        _tau, _Knp, _Kpa, p = fit_kernel(temp - temp.mean(), afd - afd.mean(),
                                          dt=DT_BIN, window_s=WINDOW_S, ridge=RIDGE)
        P.append(p)
    P = np.array(P)
    return tuple(np.median(P, axis=0))     # (lam, a0, a1) median over 21 worms


# ---------- 2. Noise (OU) parameters ----------
def noise_params(df_control):
    DT = DT_RECORD
    max_lag = int(round(60.0 / DT))
    tau = np.arange(max_lag + 1) * DT
    controls = sorted(df_control["name"].unique())
    acfs = []
    for w in controls:
        a = linear_detrend(df_control.loc[df_control["name"] == w, "afd"].to_numpy())
        a = a - a.mean()
        full = np.correlate(a, a, mode="full")
        acfs.append(full[len(a) - 1: len(a) - 1 + max_lag + 1] / len(a))
    c_mean = np.mean(acfs, axis=0)
    zc = np.where(c_mean <= 0)[0]
    t_max = tau[zc[0] - 1] if len(zc) > 0 and zc[0] > 2 else tau[-1]
    mask = (tau <= t_max) & (c_mean > 0)
    popt, _ = curve_fit(lambda tau_, C0, tc: C0 * np.exp(-tau_ / tc),
                        tau[mask], c_mean[mask],
                        p0=[c_mean[0], 20.0], bounds=([0, 1e-3], [np.inf, 1e4]))
    sigma2_n, tau_n = popt
    Dn = sigma2_n / tau_n
    return Dn, tau_n, sigma2_n


# ---------- 3. Velocity-ACF fit ----------
def velocity_params():
    rng = np.random.default_rng(SEED)
    with h5py.File(WT, "r") as f:
        keys = sorted(f.keys())
        dt = float(np.median([f[k].attrs["dt"] for k in keys]))
        max_lag = int(round(MAX_LAG_S_VEL / dt))
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
    fit_mask = (lags >= FIT_LO_VEL) & (lags <= FIT_HI_VEL)
    def model(t, p_long, p1, t_long, t1, f1, t2, f2):
        p2 = 1.0 - p_long - p1
        return (p_long * C0 * np.exp(-t / t_long)
                + p1 * C0 * np.exp(-t / t1) * np.cos(2 * np.pi * f1 * t)
                + p2 * C0 * np.exp(-t / t2) * np.cos(2 * np.pi * f2 * t))
    popt, _ = curve_fit(model, lags[fit_mask], ac[fit_mask],
                        p0=[0.3, 0.4, 8.0, 1.5, 0.5, 4.0, 0.2],
                        bounds=([0.0, 0.0, 1.0, 0.1, 0.25, 0.1, 0.05],
                                [1.0, 1.0, 100.0, 10.0, 2.0, 30.0, 0.49]),
                        maxfev=80000)
    p_long, p1, t_long, t1, f1, t2, f2 = popt
    p2 = 1.0 - p_long - p1
    return (p_long * C0, t_long,
            p1 * C0, t1, f1,
            p2 * C0, t2, f2)


# ---------- 4. Behavioral parameters ----------
def behavior_params():
    rng = np.random.default_rng(SEED_DR)
    with h5py.File(WT, "r") as f:
        keys = sorted(f.keys())
        dt = float(np.median([f[k].attrs["dt"] for k in keys]))
        max_lag = int(round(MAX_LAG_S_DR / dt))
        smooth_n = int(round(SMOOTH_S_DR / dt))
        if smooth_n % 2 == 0:
            smooth_n += 1
        min_run_n = int(round(MIN_RUN_S / dt))
        ac_sum = np.zeros(max_lag + 1); n_used = 0

        F_dur = []; R_dur = []; reor = []
        for k in keys:
            grp = f[k]
            x = grp["x"][:].astype(float)
            y = grp["y"][:].astype(float)
            state = grp["state"][:]
            d = float(grp.attrs["dt"])
            rb = grp["rev_before"][:]; ra = grp["rev_after"][:]

            # D_r: vel-ACF of long forward runs
            in_F = (state == F).astype(int)
            diff = np.diff(np.concatenate([[0], in_F, [0]]))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            for s, e in zip(starts, ends):
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

            # F/R durations
            for s, e in zip(starts, ends):
                if e - s >= 2:
                    F_dur.append((e - s) * d)
            in_R = (state == 1).astype(int)
            diff_r = np.diff(np.concatenate([[0], in_R, [0]]))
            r_starts = np.flatnonzero(diff_r == 1)
            r_ends = np.flatnonzero(diff_r == -1)
            for s, e in zip(r_starts, r_ends):
                if e - s >= 2:
                    R_dur.append((e - s) * d)
            for i in range(len(rb)):
                bvec = (rb[i, 2] - rb[i, 0], rb[i, 3] - rb[i, 1])
                avec = (ra[i, 2] - ra[i, 0], ra[i, 3] - ra[i, 1])
                cross = bvec[0] * avec[1] - bvec[1] * avec[0]
                dot = bvec[0] * avec[0] + bvec[1] * avec[1]
                reor.append(np.degrees(np.arctan2(cross, dot)))

    ac = ac_sum / n_used
    t = np.arange(max_lag + 1) * dt
    mask = (t >= FIT_LO_DR) & (t <= FIT_HI_DR) & (ac > 0)
    popt, _ = curve_fit(lambda tt, A, lam: A * np.exp(-lam * tt),
                        t[mask], ac[mask], p0=(ac[mask][0], 0.05),
                        maxfev=20000)
    D_r = popt[1]                                # 2D: λ_tot = D_r
    F_dur = np.mean(F_dur)
    R_dur = np.mean(R_dur)
    alpha = float(np.cos(np.radians(np.array(reor))).mean())
    lam_R0 = 1.0 / F_dur
    lam_T = 1.0 / R_dur
    return D_r, alpha, lam_R0, lam_T


# ---------- assemble + print ----------
def main():
    df = pd.read_pickle(AFD)
    df_sig = df[df["kind"] == "signal"]
    df_ctrl = df[df["kind"] == "control"]

    lam, a0, a1 = kernel_params(df_sig)
    Dn, taun, s2n = noise_params(df_ctrl)
    A0_, t0, A1_, t1, f1, A2_, t2, f2 = velocity_params()
    Dr, alpha, lam_R0, lam_T = behavior_params()

    print("\n=== Table 1 — fitted parameters ===\n")
    print(f"  α₀                 {a0:>11.3f}")
    print(f"  α₁                 {a1:>11.3f}")
    print(f"  λ                  {lam:>11.3f}  s⁻¹")
    print()
    print(f"  D_n                {Dn:>11.3e}  (ΔF/F)²·s")
    print(f"  τ_n                {taun:>11.2f}  s")
    print(f"  σ_n² = D_n·τ_n     {s2n:>11.5f}")
    print()
    print(f"  A0                 {A0_:>11.3e}  (mm/s)²")
    print(f"  t0                 {t0:>11.2f}  s")
    print(f"  A1                 {A1_:>11.3e}  (mm/s)²")
    print(f"  t1                 {t1:>11.2f}  s")
    print(f"  f1                 {f1:>11.3f}  Hz")
    print(f"  A2                 {A2_:>11.3e}  (mm/s)²")
    print(f"  t2                 {t2:>11.2f}  s")
    print(f"  f2                 {f2:>11.3f}  Hz")
    print()
    print(f"  D_r                {Dr:>11.3f}  rad²/s")
    print(f"  α                  {alpha:>+11.3f}")
    print(f"  λ_R0               {lam_R0:>11.3f}  Hz")
    print(f"  λ_T                {lam_T:>11.3f}  Hz")


if __name__ == "__main__":
    main()
