# Is C. elegans Information Limited? — reproducibility folder

Self-contained code + data to reproduce every figure and Table 1 from the
project report. Total disk: ~40 MB.

## Reproduce

```
pip install -r requirements.txt
for f in fig*.py; do python $f; done
python table1.py
```

Each `fig*.py` writes its PNG next to itself. `table1.py` prints to stdout.

## What's where

| File | Produces | Reads |
|---|---|---|
| `fig1_kernel.py` | AFD response kernel K(τ) — per-worm + cross-worm mean, plus forward prediction on `15wormtempvar14` | `data/afd_dataset.pkl` |
| `fig2_locomotion.py` | sample worm trajectory + cross-worm velocity autocorrelation fit | `data/wt_cache.h5` |
| `fig3_integrand.py` | information-rate integrand vs ω for 5 gradients | (`spectra.py`) |
| `fig4_behavior.py` | D_r fit, sample reversal, F/R duration + path bars, γ histogram | `data/wt_cache.h5` |
| `fig5_drift.py` | drift speed comparison: theory bound vs Ramot 2008 data | `data/ramot_fig4.csv`, `data/ramot_figS2.csv` |
| `figS2_curvature.py` | head curvature Welch PSDs across 21 worms | `data/curvature_psd.npz` |
| `figS3_noise.py` | noise autocorrelation OU fit + 5 control traces | `data/afd_dataset.pkl` |
| `figS4_drift_temp.py` | drift speed vs absolute T at 3 gradients | `data/ramot_fig4.csv`, `data/ramot_figS2.csv` |
| `table1.py` | prints all 18 fitted parameters | `data/afd_dataset.pkl`, `data/wt_cache.h5` |

Shared modules:
- `spectra.py` — parametric K̃(ω), V(ω), N(ω) + numerical Shannon integral
- `kernel.py` — AFD kernel ridge fit + closed-form 3-parameter Tsukada fit

## Data

`data/` contains the precomputed inputs every script reads. They were built
once by `build_data.py` from the upstream sources (whole-brain imaging
dataset, Roberts et al. 2016 worm tracker archive, Ramot et al. 2008 figure
extracts). Most users do **not** need to rerun `build_data.py` — the
artifacts are checked in.

| File | Size | Source |
|---|---|---|
| `data/afd_dataset.pkl` | ~5 MB | 21 signal worms (15wormtempvar*) + 5 controls. Controls' ΔF/F is renormalized to the median signal F₀ (paper's noise definition). |
| `data/wt_cache.h5` | ~35 MB | Per-worm tracks (x, y, θ, v, state, reversal arrows) for 122 wild-type worms from Roberts et al. 2016. |
| `data/ramot_fig4.csv` | 4 KB | TI values from Ramot et al. 2008 Fig 4. |
| `data/ramot_figS2.csv` | 1 KB | Run-speed-at-constant-T points from Ramot et al. 2008 Fig S2. |
| `data/curvature_psd.npz` | tiny | Welch PSD of head curvature, per-worm, 21 worms. |
| `data/figS1_static.png` | tiny | Static copy of Fig S1 — see note below. |

### Fig S1

Fig S1 (raw whole-brain imaging frame + per-neuron heatmap + temperature +
velocity) is **not** regenerable in this folder. The image is shipped as
`data/figS1_static.png` for completeness. Reproducing it requires the
original WBI .h5 files which are not redistributable.

## Rebuilding `data/`

`build_data.py` regenerates `data/` from the upstream sources. It reads
from:
- `/mnt/a/WBIbackup/{worm}_preprocessed.h5` — raw GCaMP/mNeptune fluorescence
- `../neuro1401final/randomwalkanalysis/wt_cache.h5` — already-built worm tracks
- `../neuro1401final/afd_proxy_dataset.pkl` — pre-extracted AFD traces
- `../neuro1401final/ramot_*.csv` — Ramot extractions

It is the **only** script with external data dependencies; every figure
script reads only `data/`.
# iscelegansinformationlimited
