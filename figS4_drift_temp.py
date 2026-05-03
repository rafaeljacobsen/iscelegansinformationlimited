"""Figure S4 — V_drift vs absolute temperature, T_c = 15 °C cohort.

Inverts the Ramot et al. 2008 thermotactic-index forward model to recover
V_drift at every measured (gradient, T_c, T_start) point, and plots one
connected line per gradient at T_c = 15 °C.  The dashed |TI|=0.9 envelope
marks the regime where the assay saturates and V_drift becomes unreliable.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
RAMOT4 = HERE / "data" / "ramot_fig4.csv"
RAMOT_S2 = HERE / "data" / "ramot_figS2.csv"
OUT = HERE / "figS4_drift_temp.png"

# Ramot Table 1 + assay constants
TAU = 13.4                # mean run duration (s)
TAU_P = 6.7               # mean pirouette duration (s)
ALPHA = TAU / (TAU + TAU_P)
T_TOT = 600.0             # 10-minute assay (s)
S_Z = 0.25                # half starting-zone width (cm)


def main():
    df = pd.read_csv(RAMOT4)
    s2 = pd.read_csv(RAMOT_S2)

    const_T = s2[s2["condition"] == "constant_T"]
    quad = np.polyfit(const_T["T_degC"], const_T["v_mm_per_s"], 2)

    def TI_of_V(V_cm, T_start):
        sigma = np.polyval(quad, T_start) / 10.0 * np.sqrt(ALPHA * TAU * T_TOT)
        Fp = norm.cdf(S_Z, loc=V_cm * T_TOT, scale=sigma)
        Fm = norm.cdf(-S_Z, loc=V_cm * T_TOT, scale=sigma)
        return (1 - Fp - Fm) / (1 - Fp + Fm)

    def V_from_TI(TI, T_start):
        lo, hi = -1e-2, 1e-2
        for _ in range(8):
            try:
                return brentq(lambda V: TI_of_V(V, T_start) - TI,
                              lo, hi, xtol=1e-12)
            except ValueError:
                lo *= 2; hi *= 2
        return np.nan

    df["T_start"] = df["T_c_degC"] + df["T_start_minus_T_c_degC"]
    df["V_um_per_s"] = [V_from_TI(ti, ts) * 1e4
                        for ti, ts in zip(df["TI"], df["T_start"])]

    sub = df[abs(df["T_c_degC"] - 15) < 0.6].copy()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    for grad, g in sub.groupby("gradient_degC_per_cm"):
        g = g.sort_values("T_start")
        ax.plot(g["T_start"], g["V_um_per_s"], "o-", lw=1.6, ms=8,
                label=fr"$\partial T/\partial x$ = {grad/10:.2f} °C/mm")

    # |TI| = 0.9 saturation envelope
    T_grid = np.linspace(sub["T_start"].min(), sub["T_start"].max(), 100)
    V_sat = np.array([V_from_TI(-0.9, t) * 1e4 for t in T_grid])
    ax.plot(T_grid, V_sat, "k--", lw=1.2, label=r"$|TI| = 0.9$")
    ax.fill_between(T_grid, V_sat, ax.get_ylim()[0], color="0.85", alpha=0.5)

    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("absolute temperature (°C)")
    ax.set_ylabel(r"$V_{\rm drift}$ (μm/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
