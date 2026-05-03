"""Figure 5 — drift speed comparison.

Theory line: the Mattingly information-performance bound (2D version)
    v_d / v_0  ≤  [(1-α)λ_R0 / ((1-α)λ_R0 + 2D_r)] · P_run
                 · √( (1/2) · İ(ℓ) / (λ_R0 · P_run) )
evaluated at the WT random-walk parameters (Roberts + this work).

Data points: V_drift inferred from Ramot et al. 2008 thermotaxis indices via
    TI(V) = (W − C) / (W + C),    W = 1 − Φ(+s_z),  C = Φ(−s_z),
inverted by brentq for each measured TI.  Only worms cultivated at T_c=15 °C
and tested near T_start = 21 °C are shown (the closest match to our 20 °C
recording temperature).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

from spectra import I_dot_bits, A0, A1, A2

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig5_drift.png"
RAMOT4 = HERE / "data" / "ramot_fig4.csv"
RAMOT_S2 = HERE / "data" / "ramot_figS2.csv"

# ---- Mattingly bound parameters (from fig2_locomotion.py + fig4_behavior.py) ----
F_DUR, R_DUR = 12.83, 2.54           # mean forward / reverse durations (s)
D_R = 0.0507                         # rad²/s, used in place of "2 D_r" in 2D
ALPHA = -0.064                       # ⟨cos Δθ⟩ across all reversals
LAM_R0 = 1.0 / F_DUR                 # 0.078 Hz
P_RUN = F_DUR / (F_DUR + R_DUR)      # forward duty cycle
V0_MM = np.sqrt(2 * (A0 + A1 + A2) / P_RUN)   # mm/s — 2D run-tumble baseline
V0_UM = V0_MM * 1000

PREFACTOR = ((1 - ALPHA) * LAM_R0 / ((1 - ALPHA) * LAM_R0 + D_R)) * P_RUN


def vd_over_v0(I_bits):
    return PREFACTOR * np.sqrt(0.5 * I_bits * np.log(2) / (LAM_R0 * P_RUN))


# ---- Ramot inverse forward model: V_drift from TI ---------------------
TAU, TAU_P = 13.4, 6.7               # paper Table 1 timing
ALPHA_RAMOT = TAU / (TAU + TAU_P)
T_TOT = 600.0                        # 10 min assay
S_Z = 0.25                           # half starting-zone (cm)


def v_run_quadratic(s2_df):
    """v(T) quadratic fit through the constant-T points of Fig S2."""
    sub = s2_df[s2_df["condition"] == "constant_T"]
    return np.polyfit(sub["T_degC"].to_numpy(),
                      sub["v_mm_per_s"].to_numpy(), 2)


def TI_of_V(V_drift_cm, T_start, vfit):
    """Forward model: predicted TI given drift speed."""
    sigma = np.polyval(vfit, T_start) / 10.0 * np.sqrt(ALPHA_RAMOT * TAU * T_TOT)
    Fp = norm.cdf(S_Z, loc=V_drift_cm * T_TOT, scale=sigma)
    Fm = norm.cdf(-S_Z, loc=V_drift_cm * T_TOT, scale=sigma)
    return (1 - Fp - Fm) / (1 - Fp + Fm)


def V_from_TI(TI, T_start, vfit):
    lo, hi = -1e-2, 1e-2
    for _ in range(8):
        try:
            return brentq(lambda V: TI_of_V(V, T_start, vfit) - TI,
                          lo, hi, xtol=1e-12)
        except ValueError:
            lo *= 2; hi *= 2
    return np.nan


def main():
    print(f"v_0 = √(2·V(0)/P_run) = {V0_UM:.1f} μm/s   "
          f"(P_run = {P_RUN:.3f})")

    # theory curve
    ell_grid = np.linspace(0.01, 0.20, 80)
    I_grid = np.array([I_dot_bits(e) for e in ell_grid])
    vd_norm = np.array([vd_over_v0(I) for I in I_grid])

    # Ramot points at T_c ≈ 15 °C, T_start ≈ 21 °C
    df4 = pd.read_csv(RAMOT4)
    s2 = pd.read_csv(RAMOT_S2)
    vfit = v_run_quadratic(s2)
    df4["ell"] = df4["gradient_degC_per_cm"] / 10.0
    df4["T_start"] = df4["T_c_degC"] + df4["T_start_minus_T_c_degC"]
    sub = df4[(abs(df4["T_c_degC"] - 15) < 0.6)
              & (abs(df4["T_start"] - 21) < 0.6)].copy()
    sub["V_um_per_s"] = [V_from_TI(ti, ts, vfit) * 1e4
                         for ti, ts in zip(sub["TI"], sub["T_start"])]
    sub["v_norm"] = sub["V_um_per_s"].abs() / V0_UM

    # comparison printout
    print(f"{'ℓ (°C/mm)':>10}  {'V_drift (μm/s)':>14}  {'v_d/v_0':>9}  "
          f"{'bound':>9}  {'efficiency':>10}")
    for _, r in sub.sort_values("ell").iterrows():
        bound = vd_over_v0(I_dot_bits(r["ell"]))
        eff = r["v_norm"] / bound
        print(f"  {r['ell']:8.3f}  {abs(r['V_um_per_s']):14.2f}  "
              f"{r['v_norm']:9.4f}  {bound:9.4f}  {eff:10.3f}")

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    ax.plot(ell_grid, vd_norm, color="tab:green", lw=2,
            label="information theoretic bound")
    ax.scatter(sub["ell"], sub["v_norm"], color="black", s=46, zorder=5,
               label="Ramot 2008")
    ax.set_xlabel(r"$dT/dx$  $\ell$  (°C/mm)")
    ax.set_ylabel(r"$v_d / v_0$")
    ax.set_xlim(0.01, 0.20)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
