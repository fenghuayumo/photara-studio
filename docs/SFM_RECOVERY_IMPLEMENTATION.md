# Local recovery implementation and validation — 2026-10-19

The proposal repair, generalized pose-and-scale prototype and frozen-boundary
segment refinement from [the plan](SFM_RECOVERY_PLAN.md) are implemented. **No new
camera passes real-data acceptance yet**, so these variants remain explicit
developer experiments. Default reconstruction sources and options are unchanged.
There is no evidence to promise that enabling the new recovery on every dataset
would preserve quality; the default path is protected by not promoting it yet.

## Implemented behavior

`experiments/sfm_local_proposals.hpp` replays the exact saved feature/raw-match
snapshots and verifies all proposals with native per-image calibration. It
validates pair/feature indices and preserves camera poses, calibration and
registration state. Stable-anchor pairs use their frozen geometry. Other pairs
use the intended 15-inlier local proposal policy, including pairs absent from
the original 30-inlier frontend output. The pair initializer still requires
30 matches and triangulated points; default global frontend policy is unchanged.

The `native-cycle` variant retains a weak boundary match only when an exact
feature triangle through another missing view corroborates it, and rejects weak
pairs with fewer than 15 surviving matches. `native-raw` deliberately omits this
screen as a diagnostic control. The triangle uses native-verified matches and
does not use inherited missing poses.

`experiments/sfm_ray_group_alignment.hpp` independently reconstructs local cameras
and landmarks, then builds unambiguous local-point-to-stable-image associations.
It uses PoseLib's existing `gp4ps` solver to estimate a seven-parameter similarity
from local points and world-frame rays in multiple frozen original cameras.
This is a different validator from the old shared-3D-point alignment, not a claim
to pass its minimum-30-independent-depth requirement.

The ray validator:

- Splits whole local landmarks before any candidate residual is computed. All
  anchor observations of a held-out track remain outside similarity estimation
  and subsequent BA. Repeated observations cannot inflate independent track counts.
- Requires at least 30 fit tracks, 12 held-out tracks, three anchors in each split,
  and spatial spread in three anchors. Ambiguous anchor-feature/landmark assignments
  are excluded.
- Rejects practically planar local geometry, negligible anchor baselines,
  degenerate minimal samples and ill-conditioned seven-parameter fit Hessians.
- Runs bounded deterministic RANSAC and robust normalized ray refinement, then
  checks native-camera projections at 2 pixels. Fit and held-out consensus must
  reach 80%, with distinct-track checks in addition to observation counts.
- Validates held-out tracks across multiple target cameras, then optionally
  refines the full segment with original anchor poses and calibration fixed.
  Held-out boundary pixels are never added to the BA problem.
- Revalidates held-out geometry after BA, rejects regression, filters final inlier
  observations and requires at least 30 retained observations per recovered camera.
  Original anchor poses are checked for exact preservation before committing.
- Stages everything in a disposable scene. A rejected proposal cannot mutate
  the parent reconstruction.

The internal BA comparison uses a per-track mean boundary error, with individual
non-projectable/extreme errors capped at 1000 pixels, and permits at most 0.05 px
mean regression. This bounded internal statistic is not an external quality score.
All 2-pixel inlier/holdout gates are applied separately.

`ray` and `ray-ba` use the unscreened 15-inlier proposals as tentative links: local
points already require independent multi-view reconstruction, and the multi-anchor
track/held-out pixel checks provide the later corroboration. These variants cannot
silently register a camera merely because a weak image pair passed verification.

## Real-data measurements

All group trials start with the same frozen final parents: Alameda 1711/1734,
London 1858/1874. Native-raw, native-cycle and star/pair comparisons replay the
same serialized feature and descriptor proposals. Binary/source/input hashes
are recorded, so separate builds are not represented as identical binaries.
Reference geometry never enters reconstruction.

| Missing run / variant | Shared 3D points | Fit consensus | Outcome |
| --- | ---: | ---: | --- |
| Alameda 07333–07342 / replay control | 31 | 6 | Rejected |
| Same / native raw | 49 | 7 | Rejected |
| Same / native cycle + pair initializer | 32 | 7 | Rejected |
| Same / native cycle + star initializer | 35 | 7 | Rejected |
| London 04985–05010 / replay control | 0 | — | Rejected |
| Same / native raw | 5 | — | Rejected |
| Same / native cycle | 0 | — | Rejected |

The Alameda terminal five-image run and London's first five-image run still
fail independent initialization. Pair growth reaches only three cameras;
the helper deliberately does not commit that partial scene. London's six-image
run reconstructs only a five-camera subset; none is counted as a registered gain.

The generalized alignment uses a larger set than the shared-depth validator:
Alameda's ten-image group yields **120 unambiguous local tracks across nine
anchors** (96 fit / 24 held out). The best bounded RANSAC hypothesis has **67/149
fit ray observations**, below 80%, so no placement or segment BA is accepted.
London's five-camera subset yields only **seven local tracks** (five fit / two
held out), even though four anchors appear; it fails minimum support before
RANSAC. A count of anchors by itself is insufficient.

This further narrows the failure: removing the shared-stable-depth requirement
alone does not solve Alameda's inconsistent boundary constraints, and London's
weak image-pair hypotheses rarely attach to usable independent local landmarks.
These measurements still do not distinguish every incorrect association from
local shape error. Do not lower the consensus gate to obtain a registration gain.

Frozen parents' original poses, intrinsics and original landmark/observation
contents pass exact preservation checks in all completed trials. No accepted
new segment exists for external quality promotion. Fresh full-frontend candidate
reconstruction and 3DGS comparison are consequently not claimed or promoted.

Evidence directories:

- `artifacts/sfm-native-proposals-20261019/`: eight paired replay/proposal/initializer
  trials, manifests, per-group reports and complete logs.
- `artifacts/sfm-ray-validation-20261019/`: ray and ray-plus-BA ablations; early
  reports omitted sub-threshold fit counts, which the final run corrects.
- `artifacts/sfm-ray-final-20261019/`: final ray-plus-BA trials with explicit
  sub-threshold consensus and fit/holdout observation counts. Real proposals
  reject before BA; synthetic positive tests exercise BA.

## Other datasets and default behavior

`git diff 18c372b -- photara/src/sfm photara/include/sfm` is empty. New implementation
headers are called only by `photara_sfm_stable_resection_probe --groups` and tests.
No experimental environment variable is read by the default reconstruction path.
The default benchmark executable is byte-identical to the pre-change frozen
control (`b6571747df100c99ef7e11063788f2988fd03fd84a11944dfd796221cce59a69`).

Additional complete default-path smoke reconstructions from fixed frontends:

| Dataset | Registered | Center P95 / reference steps | Rotation P95 / degrees |
| --- | ---: | ---: | ---: |
| ori_img | 76/76 | 0.012446 | 0.392418 |
| Berlin | 1511/1511 | 0.054323 | 0.350283 |

The ori_img reference scores reproduce the historical scores to numerical
precision. Berlin keeps full coverage and has similar reference pose tails to
the previous 1511-camera result. This is a smoke regression on an unchanged
default executable, not evidence of gains caused by the experimental recovery
or a universal quality guarantee for that recovery. Evidence and resources are
in `artifacts/sfm-default-regression-20261019/manifest.json`.

Seven relevant CTests pass: two-view, equirectangular, mapping, checkpoint,
submap recovery, hierarchical and Studio alignment. New synthetic cases cover
native pinhole/fisheye weak-cycle proposals; independent ray alignment without
stable-only depth; frozen-boundary BA; corrupted disjoint holdout; single-anchor
and coincident-anchor scale ambiguity; planar local geometry; conflicting feature
assignments; and wrong boundary matches.

The developer probe and submap test now match SfM/BA's AVX2 configuration when
MSVC native architecture is enabled. This fixes a reproduced Eigen allocation
ABI failure introduced by header-instantiated Ceres/Eigen code in these tools;
production targets and their build configuration are unchanged.

## Reproduction and next integration decision

Build `photara_sfm_stable_resection_probe` and `photara_submap_recovery_test` in
Release. With the recorded local caches present:

```powershell
python scripts/sfm_group_boundary_trial.py NEW_OUTPUT replay-control native-raw native-cycle native-star
python scripts/sfm_group_boundary_trial.py ANOTHER_NEW_OUTPUT ray ray-ba
```

The runner refuses an existing output folder. Logs/CSV distinguish initialization,
shared-depth alignment, generalized consensus, withheld validation and BA failures.
Ray acceptance is recorded separately from shared-3D fit counts.

Before default integration, prioritize the associations and local shape responsible
for Alameda's low ray consensus, and track-level matching/growth at London's
boundaries. Require an independently validated real coverage/precision gain, paired
whole-segment external error checks, and repeated full-frontend reconstruction on
both target scenes plus unrelated datasets. Until then, retaining these variants
as opt-in experiments preserves the default behavior.

The subsequent direct structure-less, track-completion, overlapping-window and
independent candidate-precision experiments are documented in
[SFM_BOUNDARY_SEQUENCE_EXPERIMENTS.md](SFM_BOUNDARY_SEQUENCE_EXPERIMENTS.md).
One native-proposal single-image resection passes existing gates but has a large
independent pixel-error gap and is not promoted; this is distinct from the earlier
zero-acceptance group experiments above.
