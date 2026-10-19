# Reconstruction quality improvements — 2026-10-19

The quality pass optimizes existing camera poses and landmarks using native
correspondence tracks. It is separate from frozen-camera coverage extension:
freezing all old poses can preserve precision but cannot improve their drift.

`photara_sfm_refine` is a native SfM tool built with the regular tools, independent
of the experimental probe and test executables:

```
photara_sfm_refine input-full.asfm new-output.asfm
```

It performs joint bundle adjustment with fixed calibration, Huber loss at 1 px,
and up to 100 iterations using the existing CUDA/Vulkan/CPU backend selection.
The registration set is unchanged. Camera poses and landmarks can move; the
first pose and first point fix the gauge. The tool rejects unusable or
cost-increasing solves and refuses to overwrite outputs. Reference models or
reference observations are not solver inputs. Lower native cost alone is not
treated as evidence of improved reconstruction quality.

Artifacts, executable/input/source hashes, complete logs and pixel comparisons
are in `artifacts/sfm-continued-20261019/`. Both final solves and both native
reference-evaluation processes exit zero. Saved-model proofs verify exact
calibration-record equality and unchanged registration flags. Five BA/SfM
regression tests pass (12.66 s).

## London: existing pose errors improve

Final model: `artifacts/sfm-continued-20261019/london-refined/model.asfm`.
Input is the retained 1859-camera model with fresh native correspondence tracks
from the completed balanced frontend. This pass keeps all 1859 cameras;
registration counts are not the precision claim.

The comparison uses exactly the same 2868 measured reference pixels in both
models and covers every registered camera. Every test camera is excluded from
its freshly fitted depth, and pixels are not filtered by target residual.
Reference observations are used only for this post-reconstruction evaluation.

| Error / px | Before joint refinement | After joint refinement |
| --- | ---: | ---: |
| Mean | 0.976775 | 0.952902 |
| Median | 0.678343 | 0.666128 |
| RMS | 1.902695 | 1.544184 |
| P95 | 2.687593 | 2.593031 |
| Maximum | 69.906072 | 39.032613 |

RMS decreases by 18.8%, and the maximum on this fixed cohort by 44.2%; mean
decreases by 2.4%. The separate reference model has mean 0.900437 px and RMS
1.289402 px on these pixels. The remaining 39 px tail is a real limitation.

The reference-aligned center-error median/P95 changes from 0.011414/0.026149
to 0.008406/0.021525 median reference-camera steps. Rotation median/P95 changes
from 0.040794/0.116178 to 0.030869/0.106035 degrees. This alignment is refitted
per model; the pixel comparison above provides the same-observation check.
Native refinement takes 21.63 s wall time, including loading and saving.

A temporary probe first established the improvement. Its code was archived
with that run and restored out of the experimental sources when the standalone
tool was added. The final tool output receives its own complete pixel evaluation.
CUDA runs are numerically close but not byte-identical: camera centers differ
from the prototype by at most about 5.2e-6 scene units. No prototype equality is
claimed for the final poses.

## Alameda: aggregate error and large tails improve

Final model: `artifacts/sfm-continued-20261019/alameda-refined/model.asfm`.
It refines the completed balanced global candidate while retaining 1720
registrations and the calibrated lens model. The original comparison parent is
the retained 1710-camera control, not the rejected 1709-camera vocabulary run.

The fixed shared cohort is exactly the same as the earlier global-candidate
comparison: 1708 cameras and 2681 measured pixels. Lost-camera pixels are not
included in either side, so removing bad cameras does not create the improvement
in this table.

| Error / px | Retained 1710-camera parent | Balanced global result | Final joint refinement |
| --- | ---: | ---: | ---: |
| Mean | 1.913158 | 1.599400 | 1.547204 |
| Median | 0.943493 | 0.932546 | 0.923403 |
| RMS | 8.245253 | 4.750148 | 4.541831 |
| P95 | 4.330761 | 4.172044 | 4.095102 |
| Maximum | 219.990168 | 128.656554 | 125.473919 |

Compared with the retained parent, final mean decreases by **19.1%**, RMS by
**44.9%**, and P95 by **5.4%**. The new joint refinement also improves all five
statistics over the balanced global result, rather than merely inheriting the
retrieval-stage gain. Reference mean/RMS is 1.145033/1.592762 px on these pixels.
The remaining 125 px maximum shows that the weakest regions still need work.

The final result has twelve gained cameras and two lost cameras relative to
the original parent; their identities and the earlier conservative frozen
alternative are documented in the coverage report. Independent reference
evaluation and native old-anchor histograms remain available for auditing the
weak recovered branch. This is not a guarantee of uniformly correct depth or
camera geometry. Native final refinement takes 32.12 s wall time, including
loading and saving.

For coverage-only results, exact feature-index merging and the earlier complete
Alameda frontend continuation, see
[`SFM_FROZEN_TRACK_EXTENSION.md`](SFM_FROZEN_TRACK_EXTENSION.md).

These saved-model refinements do not establish a complete new cold default
pipeline result, nor prove that every local region is accurate. The optional
balanced retrieval policy remains separate from the quality tool.

Further offline improvements from refreshing native track associations are documented in [SFM_QUALITY_TRACK_REFRESH.md](SFM_QUALITY_TRACK_REFRESH.md).
