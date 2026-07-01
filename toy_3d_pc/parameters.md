# How to Tune the Parameters

## Jun 26 — Initial Sweep (Groups A–F)

**Setup**

- CryoET w/ 32 tilts (observations) + 5deg per tilt.
- Clean distribution $\pi = \delta_x$ where $x$ is a solid torus.
  - To approximate the torus with a point cloud, uniformly sample the volume of the object with $N$ points (via rejection sampling).
- Pretrain model using pseudoinverse (Algorithm 1 warm-start).
- All sweeps A–F ran under the **old defaults**: linear interpolant, $\gamma = 0.995$.

**What was swept**


| Group | Parameter                         | Values tested                                               |
| ----- | --------------------------------- | ----------------------------------------------------------- |
| A     | coupling ($\alpha_z$, $\alpha_y$) | (0,0), (0,0.5), (0,1), (0.05,0.05), (0.5,0.5), full obs     |
| B     | EMA decay $\gamma$                | 0.9, 0.99, 0.995, 0.999                                     |
| C     | ODE sample steps                  | 8, 16, 32, (64 baseline)                                    |
| D     | pretrain steps                    | 0, 500, 2000, 5000                                          |
| E     | EM vs inner-step ratio            | (200 em × 100 tr), (50 em × 400 tr) vs baseline (100 × 200) |
| F     | interpolant style                 | GVP vs linear                                               |


**Findings**

- GVP interpolant helps a lot. See [here](https://wandb.ai/clarkmiyamoto-new-york-university/toy3d-pc-scsi/runs/e63ab6w1/overview?nw=nwuserclarkmiyamoto). Joan had the intuition this would matter, but wasn't sure why. GVP maintains unit variance throughout the ODE ($\alpha^2 + \beta^2 = 1$ at all $t$), which likely gives smoother velocity fields for point cloud targets.
- Slower EMA $\gamma = 0.999$ appears to help.
- **Convergence is slow**: needs ~100 EM steps × 200 inner steps, taking ~8 hours.
- **New defaults set**: `--interpolant-style gvp`, `--ema-decay 0.999`.

---

## Jun 26 — Next Sweep (Groups G–I)

**Key gap**: every A–F sweep ran under the old linear+0.995 defaults. The interaction effects under the new GVP+0.999 regime are unknown, and the learning rate was never swept.

**Group G — Learning rate** (G0 new baseline, G1 lr=5e-4, G2 lr=1e-3)

Hypothesis: GVP's cosine schedule preserves unit variance at all $t$, so velocity-field gradients are smoother and do not blow up at intermediate $t$ the way linear OT can. A 2.5–5× higher LR should remain stable under GVP and may let each inner training step take a larger useful step, cutting the EM iterations needed to converge.

- G0: explicit GVP + $\gamma=0.999$ reference (all other values at default) — needed because A0 used linear + 0.995.
- G1: `lr=5e-4` (2.5×)
- G2: `lr=1e-3` (5×)

**Group H — Iso-compute cheap ODE → more EM iters** (H1: 16 steps × 400 EM, H2: 8 steps × 800 EM)

Hypothesis: each inner training step runs a full 64-step ODE — the dominant cost. Under GVP (smooth cosine interpolant, no sharp kink at $t=0$), fewer Euler steps should still produce accurate $\hat{x}$ targets. Reinvesting that saved compute into 4×/8× more outer prior updates directly addresses slow EM convergence. Total ODE evaluations are identical to the baseline ($100 \times 200 \times 64 = 1{,}280{,}000$) in both H1 and H2.

- H1: `--sample-steps 16 --em-steps 400`
- H2: `--sample-steps 8 --em-steps 800` (+2 hr wall time for per-step logging overhead)

**Group I — Tomo-quantile / F† quality** (I1: 0.05, I2: 0.3, I3: 0.5)

Hypothesis: the pseudo-inverse quality sets the warm-start quality, which determines how many EM steps are needed from the start. The default `tomo-quantile=0.15` was never ablated. A tighter quantile discards more borderline voxels → cleaner but sparser initial cloud. A looser one → denser but noisier. For a torus with $K=32$ tilts, the ring only appears in roughly half the tilt views, so the median quantile (0.5) likely over-carves — useful as a sanity-check lower bound.

- I1: `--tomo-quantile 0.05` (very loose)
- I2: `--tomo-quantile 0.3` (moderately tight)
- I3: `--tomo-quantile 0.5` (median — expected to over-carve)

