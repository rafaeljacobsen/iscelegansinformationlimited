"""Parametric spectral models and Shannon information rate.

Hard-coded parameters reproduce Table 1 of the paper. Three power spectra:
  K2(ω)   = |K̃(ω)|²,  AFD response kernel
  V(ω)    = velocity power spectrum (Roberts et al., 122 worms)
  N(ω)    = AFD spontaneous-noise power spectrum (5 controls, OU fit)

Numerical integration of equation (8):
  İ_{s→AFD}(ℓ) = (1/4π ln 2) · ∫_{-∞}^{∞} log(1 + ℓ² V(ω) K2(ω) / N(ω)) dω

The integrand has narrow Lorentzian peaks at ω = ±2π·f1, ±2π·f2 from V(ω);
quad is split around them so it doesn't miss the peaks.
"""
import numpy as np
from scipy.integrate import quad

# ---- AFD kernel K̃(ω) (Tsukada 3-param fit, per-worm median over 21 worms) ----
A0_K, A1_K, LAM = 0.227, 0.235, 0.131           # α₀, α₁, λ

# ---- Velocity V(ω) (exp + 2 damped-cosine fit, Roberts 122 worms) ----
A0, T0      = 0.00620, 7.47                     # exponential persistence
A1, T1, F1  = 0.00699, 1.49, 0.346              # body-bend oscillation
A2, T2, F2  = 0.00413, 1.87, 0.143              # slow head-swing oscillation

# ---- Noise N(ω) (OU fit to 5 control AFD traces, F₀-normalized) ----
DN, TAUN    = 0.000635, 17.06                   # D_n, τ_n


def K2(w):
    """|K̃(ω)|² — equation (4) form, with K̃(ω) = (α₁-α₀)/[λ(λ+iω)] + α₁/(λ+iω)."""
    return ((2*A1_K - A0_K)**2 + w**2 * (A1_K - A0_K)**2 / LAM**2) \
           / (LAM**2 + w**2)**2


def V(w):
    """Velocity power spectrum — Lorentzian + 4 shifted Lorentzians."""
    return ((2*A0/T0) / ((1/T0)**2 + w**2)
          + (A1/T1) / ((1/T1)**2 + (w - 2*np.pi*F1)**2)
          + (A1/T1) / ((1/T1)**2 + (w + 2*np.pi*F1)**2)
          + (A2/T2) / ((1/T2)**2 + (w - 2*np.pi*F2)**2)
          + (A2/T2) / ((1/T2)**2 + (w + 2*np.pi*F2)**2))


def N(w):
    """OU noise power: 2D_n / (1/τ_n)² + ω²)."""
    return 2*DN / ((1/TAUN)**2 + w**2)


def integrand_log(w, ell):
    """Log integrand of equation (8), in nats."""
    return np.log(1.0 + ell**2 * V(w) * K2(w) / N(w))


def I_dot_bits(ell):
    """Information rate İ_{s→AFD}(ℓ) in bits/s."""
    peaks = sorted([-2*np.pi*F1, -2*np.pi*F2, 0.0, 2*np.pi*F2, 2*np.pi*F1])
    edges = [-np.inf, *peaks, np.inf]
    nats = sum(quad(lambda w, e=ell: integrand_log(w, e), a, b, limit=400)[0]
               for a, b in zip(edges[:-1], edges[1:]))
    return nats / (4 * np.pi * np.log(2))


if __name__ == "__main__":
    print(f"{'ℓ (°C/mm)':>10}  {'İ (bits/s)':>14}")
    for ell in [0.05, 0.1, 0.3, 0.5, 1.0]:
        print(f"  {ell:8.2f}      {I_dot_bits(ell):8.5f}")
