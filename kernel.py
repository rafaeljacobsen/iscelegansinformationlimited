"""AFD response kernel K(τ) — ridge regression and 3-parameter Tsukada fit.

Forward model (equation 9 in the paper):
  AFD(t) = Σ_τ K(τ) · T(t − τ),   K(τ) = exp(−λτ) · (α₀ − λα₁τ)

For a single worm, K is fit by:
  1. Ridge regression on the lagged design matrix → nonparametric K(τ).
  2. Closed-form (α₀, α₁) per λ + Brent search over λ → parametric K(τ).
     (Appendix A of the paper.)

Inputs are 1-second-binned, demeaned signals.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def parametric_kernel(tau, lam, a0, a1):
    return np.exp(-lam * tau) * (a0 - lam * a1 * tau)


def design_matrix(u, n_lags):
    """Lower-triangular Toeplitz: row t, column k holds u[t − k]."""
    n = len(u)
    X = np.zeros((n, n_lags))
    for k in range(n_lags):
        X[k:, k] = u[: n - k]
    return X


def fit_kernel(temp, afd, dt=1.0, window_s=100.0, ridge=1.0):
    """Fit K(τ) for one worm. Returns (tau, K_nonparam, K_param, (lam, a0, a1)).

    `temp` and `afd` should be 1-second-binned and demeaned by the caller.
    """
    n_lags = int(round(window_s / dt))
    tau = np.arange(n_lags) * dt
    X_full = design_matrix(temp, n_lags)
    # First n_lags rows are edge-effected (incomplete convolution); drop them.
    X = X_full[n_lags:]
    y = afd[n_lags:]

    # Nonparametric ridge: K_np = (XᵀX + ρI)⁻¹ Xᵀy
    K_np = np.linalg.solve(X.T @ X + ridge * np.eye(n_lags), X.T @ y)

    # Parametric K(τ) = α₀·exp(−λτ) − α₁·λτ·exp(−λτ).
    # Linear in (α₀, α₁) for fixed λ → solve 2×2 normal equations per λ.
    yy = float(y @ y)
    def loss_alpha(lam):
        if lam <= 0:
            return np.inf, 0.0, 0.0
        b0 = np.exp(-lam * tau)
        b1 = -lam * tau * np.exp(-lam * tau)
        c0, c1 = X @ b0, X @ b1
        M = np.array([[c0 @ c0, c0 @ c1],
                      [c0 @ c1, c1 @ c1]])
        rhs = np.array([c0 @ y, c1 @ y])
        try:
            a0, a1 = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return np.inf, 0.0, 0.0
        loss = (yy - 2 * (a0 * (c0 @ y) + a1 * (c1 @ y))
                + a0 * a0 * (c0 @ c0) + 2 * a0 * a1 * (c0 @ c1)
                + a1 * a1 * (c1 @ c1))
        return float(loss), float(a0), float(a1)

    # Coarse geometric scan, then Brent refinement around the minimum.
    lams = np.geomspace(1e-4, 50.0, 600)
    losses = np.array([loss_alpha(l)[0] for l in lams])
    j = int(np.argmin(losses))
    lo, hi = lams[max(0, j - 1)], lams[min(len(lams) - 1, j + 1)]
    res = minimize_scalar(lambda l: loss_alpha(l)[0],
                          bracket=(lo, lams[j], hi),
                          method="Brent", options={"xtol": 1e-9})
    lam = float(res.x) if np.isfinite(res.fun) else lams[j]
    _, a0, a1 = loss_alpha(lam)
    K_param = parametric_kernel(tau, lam, a0, a1)
    return tau, K_np, K_param, (lam, a0, a1)


def bin_to_1s(x, n_per_s):
    """Block-average a 5 fps trace down to 1 fps."""
    n = (len(x) // n_per_s) * n_per_s
    return x[:n].reshape(-1, n_per_s).mean(axis=1)
