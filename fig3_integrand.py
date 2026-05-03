"""Figure 3 — Information rate integrand vs ω for several gradients ℓ."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from spectra import K2, V, N

OUT = Path(__file__).resolve().parent / "fig3_integrand.png"

ELL = [0.05, 0.1, 0.3, 0.5, 1.0]      # °C/mm

# Dense, log-spaced grid on each side of zero so the narrow Lorentzian peaks
# don't get aliased away.
w = np.unique(np.concatenate([
    -np.geomspace(0.001, 100, 4000)[::-1],
    [0.0],
    np.geomspace(0.001, 100, 4000),
]))


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    for ell in ELL:
        snr = ell**2 * V(w) * K2(w) / N(w)
        integrand = np.log(1.0 + snr) / (4 * np.pi * np.log(2))
        ax.plot(w, integrand, lw=1.4, label=fr"$\ell = {ell}$ °C/mm")
    ax.set_xlabel(r"$\omega$ (rad/s)")
    ax.set_ylabel(r"$\dot{I}_{s\to\mathrm{AFD}}$")
    ax.set_xlim(-3, 3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
