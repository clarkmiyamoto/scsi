# Parameter Tuning Notes

**Current focus: `torus` and `l_shape`.** Neither is fully solved yet — everything below is
either a hard constraint (confirmed across many sweeps) or an open question specific to these
two shapes. Other shapes (`dumbbell`, `trefoil`, `t_shape`) are parked; don't spend more sweep
budget there until torus/l_shape converge.

## Known constraints — do not violate

These are the things we're confident *break* training. What actually makes it work well is
still unclear (see Open questions) — but violating any of these reliably makes it worse.

1. **`--sample-steps` ≥ 16** (Euler). 8 or fewer visibly degrades transported targets
   (Group K). Untested whether Heun tolerates fewer (Q14–16/R14–16 pending).
2. **`--ema-decay` ≥ 0.995, prefer 0.999** (slower/closer to 1). Faster EMA (0.9–0.99) hurts;
   0.999 is current default (Group B).
3. **Warm-start / pretrain is required.** `--pretrain-steps 0` gives clearly bad performance —
   confirmed enough that we dropped the no-pretrain arm (Q6/R6) from later sweeps rather than
   keep re-running it. Exact minimum steps is unknown; the default (2000) works, untested
   whether less is fine.
4. **`--lr` < 1e-3.** 1e-3 or higher is too high; default `2e-4` is the known-good point
   (Group G, Q12/Q13/R12/R13 pending confirmation on l_shape/trefoil).
5. **Coupling (`--alpha-z`/`--alpha-y`) needs to be small but nonzero: ~0.05–0.1.** Around 0.5
   kills training. (Two-object mixture sweep J saw no clear signal across 0–0.2, but that was a
   less sensitive setup — treat 0.05/0.1 as the safe range until l_shape/torus-specific Q/R
   results land.)

## Open questions

- What does "working" actually look like for torus/l_shape — no sweep so far has produced a
  clearly-converged reference sample to compare against.
- GVP interpolant > linear (Group F/A), but by how much on l_shape/trefoil specifically is
  still running (Q11/R11).
- EM step budget: 100×200 wasn't enough to converge on l_shape/trefoil — bumped to 500 EM steps
  Jul 1 for the Q/R re-ablation (see below).
- Splat radius, `n_points`, and tomo-quantile were tuned on torus only; per-shape sensitivity
  (Groups M/N/O) is partially run, results not yet synthesized.

## Sweep log (condensed)

| Date | Groups | Setup | Key finding |
|---|---|---|---|
| Jun 26 | A–F | single torus, linear+0.995 baseline | GVP interpolant helps a lot; slower EMA (0.999) helps; convergence is slow (~100×200 steps, ~8h) |
| Jun 26 | G–I | torus, GVP+0.999 baseline | lr sweep (G), iso-compute cheap-ODE-more-EM (H), tomo-quantile (I) — see raw results in W&B, not re-derived here |
| Jun 30 | J, K | two-shape mixture (torus+cylinder) | K: 16 sample-steps is the Euler floor (4/8 degrade, 16≈32). J: no clear ranking from coupling 0–0.2 or iid-vs-template — inconclusive, needs a sharper metric |
| Jun 30 | L | 5 shapes (torus, dumbbell, trefoil, l_shape, t_shape), single-shape baselines | established per-shape baseline recipe below; l_shape and trefoil didn't finish (see incident) |
| Jun 30 | M–P | planned follow-ups: splat radius (M), n_points (N), tomo-quantile (O), 5-shape mixture + integrator (P) | **Jul 1 incident**: all M/O/P jobs + N2/N4 were simultaneously SIGNAL-Terminated ~1-2h in, across nodes — looks like an external kill, not per-job timeout. Only N1/N3 completed. Rerun a couple jobs first before mass-resubmitting. |
| Jul 1 | Q, R | full A–K re-ablation, but only on `l_shape` (Q) and `trefoil` (R) — every prior parameter was only ever validated on torus | bumped em-steps 100→500 (not converged at 100); dropped no-pretrain arm (see constraint #3); found + fixed a latent bug: per-EM checkpoints weren't namespaced per run, so concurrent jobs could clobber each other's checkpoints — now under `toy_3d_pc_checkpoints/<out-stem>/` |
| Jul 1 | S | new `--canonicalize` flag ablation: `l_shape` solo, `torus` solo, `l_shape`+`torus` mixture | see plan below; not yet run |

**Baseline recipe** (Group L, still current default): `--n-objects 400 --interpolant-style gvp
--sample-steps 16 --alpha-z 0.05 --alpha-y 0.05 --ema-decay 0.999 --em-steps 100
--training-steps 200 --pretrain-steps 2000 --tomo-quantile 0.15 --radius 0.08 --n-points 512
--n-tilts 32 --tilt-step 5`.

## Active runs (Q/R re-ablation on l_shape/trefoil, 500 EM steps)

Each arm overrides exactly one axis from the baseline recipe above. `--time=08:00:00` per job
is shorter than the ~40h needed for 500 EM steps, so each job needs manual `--resume`
resubmission roughly every 8h (pointing at
`toy_3d_pc_checkpoints/<out-stem>/model_em{k:04d}.pt`) until `em_step` reaches 500.

| Axis | l_shape (Q) | trefoil (R) |
|---|---|---|
| coupling off / mid / full | Q1 / Q2 / Q3 | R1 / R2 / R3 |
| EMA fast (0.9) / mid (0.995) | Q4 / Q5 | R4 / R5 |
| pretrain light (500) / heavy (5000) | Q7 / Q8 | R7 / R8 |
| EM ratio wide (200×100) / deep (50×400) | Q9 / Q10 | R9 / R10 |
| linear interpolant | Q11 | R11 |
| lr high (5e-4) / v.high (1e-3) | Q12 / Q13 | R12 / R13 |
| ODE 8 / 32 steps | Q14 / Q15 | R14 / R15 |
| Heun @16 | Q16 | R16 |
| tomo-quantile (gaps only) | Q17 (0.5) | R17/R18/R19 (0.05/0.3/0.5) |
| n_points low/high | Q18 / Q19 | *(covered by N1/N2)* |
| radius low/high | *(covered by M1/M2)* | R20 / R21 |

Also pending rerun from the Jul 1 incident (never completed): L3 (trefoil), L4 (l_shape), M1,
M2, N2, O1, O2. Submit in waves and check `squeue` between waves rather than all at once.

## Group S — `--canonicalize` ablation (planned Jul 1)

Goal: does canonicalizing the interpolant target (`x̂_C = C(x̂)`, see `canonicalize.py` /
`CLAUDE.md`) improve training, and does it relax any of the "known constraints" above (which
were all tuned *without* canonicalize)? Per `CLAUDE.md`, `torus` has no identifiable in-plane
angle under `pca_canonicalize` (continuous symmetry) — it's included per explicit request as a
direct empirical check, but the expected outcome there is no-op-or-worse, not an improvement.
`l_shape` (no continuous symmetry, current focus shape) is the primary testbed.

Shapes: `l_shape` solo, `torus` solo, `l_shape`+`torus` mixture (uniform). `--n-objects` bumped
400→2048 (matches the CLI default) and `--em-steps` 500 (matches the Q/R standard, needs manual
`--resume` resubmission every ~8h like Q/R) — both changed relative to the old Group L baseline,
so canonicalize-on/off controls are run at the *same* new settings rather than reusing L1/L4.

| # | Shape(s) | canonicalize | Deviation from baseline recipe | Compares against |
|---|---|---|---|---|
| S1 | l_shape | on | none (n-objects 2048, em-steps 500) | S1b |
| S1b | l_shape | off | none | S1 |
| S2 | torus | on | none | S2b |
| S2b | torus | off | none | S2 |
| S3 | l_shape+torus | on | none | S3b |
| S3b | l_shape+torus | off | none | S3 |
| S4 | l_shape | on | `--alpha-z 0 --alpha-y 0` (coupling off) | S1 |
| S5 | l_shape | on | `--ema-decay 0.995` (faster EMA) | S1 |
| S6 | l_shape | on | `--sample-steps 8` (cheap ODE) | S1 |
| S7 | l_shape | on | `--lr 5e-4` (higher lr) | S1 |
| S8 | l_shape+torus | on | `--alpha-z 0 --alpha-y 0` | S3 |
| S9 | l_shape+torus | on | `--ema-decay 0.995` | S3 |
| S10 | l_shape+torus | on | `--sample-steps 8` | S3 |
| S11 | l_shape+torus | on | `--lr 5e-4` | S3 |

Interaction arms (S4–S11) test whether canonicalize — by removing the rotation ambiguity from
the transported target — relaxes constraints #1 (sample-steps floor), #2 (slow EMA), #4 (lr
ceiling), #5 (small-nonzero coupling). Each changes exactly one axis off the canonicalize-on
baseline (S1 or S3). Submit S1/S1b/S2/S2b/S3/S3b first; only submit S4–S11 once the baseline
A/B shows canonicalize is worth the interaction budget on `l_shape`.
