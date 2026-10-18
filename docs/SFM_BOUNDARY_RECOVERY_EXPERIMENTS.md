# Bounded boundary recovery experiments — 2026-10-18

This follows [the branch constraint experiments](SFM_BRANCH_CONSTRAINT_EXPERIMENTS.md).
There is **no validated registration gain** in this follow-up. The frozen parent
models remain Alameda **1711/1734** and London **1858/1874**. These are the previous
control's final graphs, not fresh complete alignments. The one-camera losses in
the earlier traced reruns remain unresolved; this experiment does not establish
end-to-end reproducibility or rendering quality.

## Positioning quota

The existing opt-in quota reserves at most one quarter of the positioning budget
for eligible tracks connecting at least two strong-core cameras to weaker views.
It preserves the four-view qualification and is disabled by default.

| Fixed missing cohort: selected cross-cohort incidences | Quota 0 | Quota 20 |
| --- | ---: | ---: |
| Alameda | 255 | 255 |
| London | 4 | 4 |

The quota selected 350/454 tracks globally, respectively, but did not increase
the measured missing-cohort boundary incidences. The runner therefore skipped
complete reconstruction trials. This result supports neither enabling the quota
nor a claim that its effect elsewhere in the scene is neutral. Evidence:
`artifacts/sfm-boundary-quota-20261018/manifest.json`.

## Joint missing-run recovery

All variants reconstruct missing views independently with native fixed intrinsics,
using the existing bounded pair initializer. Original registered camera poses,
intrinsics and original landmark/observation contents are checked for exact
preservation. References are never supplied to reconstruction. The unchanged
validator requires at least 30 independent shared points, 80% fit consensus,
held-out 3D/pixel agreement, and final per-camera support.

Variants:

1. **Control:** the previous 16-anchor search.
2. **Covisible:** up to 32 anchors. Additional anchors are nominated by original
   map tracks containing preserved boundary features and at least three registered
   observers; bounded temporal proposals fill remaining slots. Those tracks only
   nominate images. Local depth is freshly triangulated.
3. **Frozen geometry:** the same bounded nomination, with original raw descriptor
   proposals between stable anchors filtered using their fixed native poses.
   Both epipolar directions must pass the 4-pixel angular gate. Target pairs retain
   ordinary estimated geometry. This avoids discarding anchor-only proposals solely
   because a fresh two-view essential-matrix estimate fails. Depth triangulation
   and submap acceptance thresholds are unchanged.

The frozen variant requires its own unambiguous frontend match checkpoint and
validates feature indices. It never obtains missing-view poses from the parent.
Neither this variant nor covisible nomination runs in the default mapper.

| Missing run | Control shared points | Covisible | Frozen geometry | Accepted additions |
| --- | ---: | ---: | ---: | ---: |
| Alameda 07333–07342, ten inputs | 14 | 29 | 31 | 0 |
| Alameda 08180–08184, five inputs | — | — | — | 0 |
| London 04895–04920, five inputs | — | — | — | 0 |
| London 04985–05010, six inputs | 0 | 0 | 0 | 0 |

A dash means independent initialization remained incomplete, before shared-point
validation. The initializer's best trial reached three cameras for these runs;
it deliberately does not commit that partial scene. London's six-input run
reconstructs only a five-camera subset; those local cameras are not registrations
in the parent map.

For Alameda's ten-camera run, frozen geometry finds 691 candidate tracks, 32 with
at least two stable observations, 31 independently triangulated shared points,
and 303 verified boundary matches. Of the 24 fit points after the deterministic
every-fifth-point holdout split, only **6** enter the similarity consensus.
The candidate is rejected before held-out validation. Crossing the 30-point
minimum therefore does not solve the inconsistency; simply lowering it to 29
would not be a justified recovery strategy.

For London's five-camera subset, there are 498 candidate tracks but zero with
two stable observations and zero verified boundary matches. Additional internal
local geometry and anchor-only matching still do not create a map connection.
These counts do not determine whether retrieval, descriptor matching or boundary
verification first lost that connection.

## Evidence and verification

`scripts/sfm_group_boundary_trial.py NEW_DIRECTORY [control covisible frozen]`
runs the variants sequentially and records input/source/binary hashes, process
results, resource samples, models, CSV rejection reports and complete logs.
It refuses an existing output directory. No reference paths are passed to the
recovery process. The frontend is freshly recomputed per variant; byte-identical
proposals across variants are not claimed.

- `artifacts/sfm-group-boundary-20261018/`: control/covisible comparisons.
- `artifacts/sfm-group-frozen-20261018/`: raw-match frozen-geometry trials, including
  frontend caches and the extended stage-count CSVs.
- `artifacts/sfm-group-frozen-20261018/ctest.log`: mapping, submap recovery,
  checkpoint and Studio alignment regression results.

All six real-data trials finish successfully and report exact preservation of
the original cameras, calibration and landmarks. Four relevant CTests pass.
No new view passed acceptance, so there is no new-camera external-quality result
and no default algorithm change to promote. The next useful work is consistent
cross-boundary target correspondences and independently checked local shape;
larger anchor counts alone have now been measured and are insufficient here.
