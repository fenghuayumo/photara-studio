# EMC densification strategy (2026-09-18)

New `DensificationStrategy::emc` (`--splat-strategy emc`), an error-map MCMC densification scheme so
error-map-driven placement can be evaluated as a first-class strategy instead of a bounded re-order:

- Score: SSIM-CS error map (power 4) -> rasterizer-backward
  contribution-weighted per-splat average -> window mean -> opacity-gated ->
  score power (0.4). No gradient gate; error is the primary signal.
- Relocation: dead rows (opacity < 1/255, log-scale <= -40, non-finite) are
  recycled in place as the +delta child of an error-sampled parent
  (long-axis split). Count-preserving, so error-primary selection cannot
  bloat the model.
- Growth: fixed multiplier of the live count per refine (preset 1.05,
  `--splat-growth-factor` override), error-sampled, with 15% of the budget
  reserved for accumulated on-screen oversize rows
  (weight = (oversize + 1e-3) * score^blend).
- Long-axis split (arXiv:2508.12313): largest axis x0.5, others x0.85,
  offset 0.5 x major scale along the major axis, opacity mapped by a
  scheduled factor k = 0.5 -> 0.6 over 15k iterations. Parent takes the
  -delta side; the child (or the recycled row) takes +delta.
- No refine-time opacity decay (recycling replaces pruning;
  the other ADC strategies keep the Brush decay).
- Shares the Brush-parity training config (LRs, initial opacity 0.5, mean
  noise, background noise 0.1, progressive resolution, refine every 200
  from iteration 0).

## First results (ori, seed 42, 10k iters, cap 1M, 10-view external eval)

| strategy | PSNR | SSIM | Gaussians | time |
|---|---:|---:|---:|---:|
| adc_plus (sd=0) | 22.3741 | 0.9111 | 671,246 | 32.3 s |
| adc_igs (sd=0)  | 25.8113 | 0.9532 | 614,468 | 36.3 s |
| emc 1.05        | 22.2844 | 0.8657 | 526,272 | 29.3 s |
| emc 1.10        | 22.9774 | 0.8931 | 1,000,000 (cap) | 47.5 s |

Mechanism verified: +5%/refine compounding growth, late-window dead-row
recycling (58-92 rows per refine), active oversize channel, exact
split/opacity math (unit-tested).

Interpretation: on the ori turntable protocol the error-primary placement
does not beat the gradient-gated ADC-IGS machinery - IGS reaches +2.8 dB
with 39% fewer Gaussians than EMC at the 1M cap. This is consistent with
the 2026-09-17 finding that SSIM-error-primary scores drift toward
hard-to-fit background rays on this data. EMC's promise is walkthrough-style scenes; further datasets/settings (office,
per-step densification, adaptive noise) remain to be evaluated.

Artifacts: artifacts/emc_eval_20260918 (commands, logs, eval, PLYs).

## Office dataset addendum (2026-09-18, same day)

On the 900-view office walkthrough (seed 42, 30k iters, cap 3M, 113
held-out views, same binary both arms): EMC leads at the 5k milestone
(23.675 vs 23.231, +0.44 dB) exactly while its +5%/refine budget is open,
then saturates the cap at iteration ~5,200 and finishes at 25.064 vs
ADC-IGS 25.678 (-0.62 dB; EMC wins only 13/113 views). Error-driven
placement is the more sample-efficient scheme with an open budget, but
dead-row-only recycling is a weaker lever than IGS's evidence-driven
replacement once the cap binds. Full data: artifacts/office_emc_20260918.

## EMC v2: MCMC cadence + revised noise (2026-09-18)

Aligned EMC to the reference MCMC cadence (start 500, every 100) and
added the revised per-step noise (per-step, opacity-gated, sqrt(scale)-axis
scaled, 80 -> 0.8 decay; Brush mean noise no longer runs for EMC). Office
rerun, same binary both arms: EMC improves 25.064 -> 25.280 dB; the gap to
ADC-IGS narrows from -0.62 to -0.35 dB, the 10k milestone flips to
+0.17 dB, and per-view wins double (13 -> 23 of 113). The remaining
15k-30k tail tracks the still-unported pieces: per-splat regularizers,
oversize hard clip + penalty, parent-moment preservation. Full data:
artifacts/office_emc_r2_20260918.
