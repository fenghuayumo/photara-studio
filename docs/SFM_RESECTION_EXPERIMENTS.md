# Registration failure experiments — 2026-10-18

## Scope and controls

Follow-up to `SFM_FAILURE_ANALYSIS.md`. Complete alignment trials start from
the same saved frontend tracks and include global positioning, BA, recovery
and final audits. They do not re-extract the full frontend. References enter
post-run evaluation only. Frozen executables, source/input hashes, command
lines, resource samples and results are recorded under
`artifacts/sfm-resection-analysis-20261018/`.

Two concrete resection problems were reproduced and fixed:

- A rejected feature/landmark assignment must not claim the feature and block
  its valid alternative. The regression reproduces the old failure and checks
  recovery of the true camera center with the fix.
- Failure of the strongest correspondence-count band must not prevent trying
  lower-count eligible cameras. The regression contains an invalid high-count
  camera and a valid lower-count camera; only the latter must register.

Both retain all original PnP/geometry acceptance requirements. Public
`ResectionConfig` switches permit ablation; both default on and enter the
reconstruction cache key. The benchmark accepts
`PHOTARA_BENCHMARK_RESECTION_VARIANT=baseline|conflicts|band|both`.

## Complete alignment results

| Scene / variant | Registered | Alignment seconds | Conflict cases | Exhausted candidate bands |
| --- | ---: | ---: | ---: | ---: |
| Alameda / legacy resection control | 1711/1734 | 288.97 | 0 | 0 |
| Alameda / conflict correction | 1711/1734 | 290.32 | 0 | 0 |
| London / legacy resection control | 1858/1874 | 152.45 | 0 | 0 |

The six Alameda and three London images that enter incremental resection
have **zero selected 2D–3D correspondences**. They never reach either defective
path. The Alameda control confirms no registration gain; tiny numeric
differences in GPU/BA results are not evidence that the correction helped.
The remaining unexercised ablations were skipped rather than treated as
completed experiments. Both code corrections remain covered by the targeted
regressions. They are not the measured cause of these datasets' missing runs.

## Cross-boundary evidence audit

This audit reads the final checkpoint without changing geometry. It counts
potential links from every preserved relative-pose edge, including inactive
edges, to inlier observations of the registered map. A feature associated with
multiple triangulated landmarks is excluded as ambiguous. Counts are search
evidence, not certified independent depths or accepted camera observations.

| Diagnostic for remaining missing images | Alameda | London |
| --- | ---: | ---: |
| Missing images | 23 | 16 |
| No preserved edge to any registered neighbor | 6 | 5 |
| Fewer than 30 unambiguous features linked to map landmarks | 18 | 13 |
| At least 30 linked features whose landmarks have 3 registered views | 3 | 3 |

For example, Alameda's `DSC07334`–`DSC07338` each has nine other missing
neighbors, but only 6, 4, 2, 1 and 1 unambiguous map links respectively.
That is direct evidence of strong internal image connectivity without enough
usable boundary depth. Repeated single-view PnP cannot create absent depth.

## Were the withheld poses actually inaccurate?

The checkpoint preserves some unsupported poses after clearing registration.
An evaluation-only pose snapshot makes those available to the reference
evaluator; it contains no points and **is not a registered reconstruction**.
These are retained final poses, not an exact checkpoint immediately before
withdrawal, so the comparison does not attribute all error to a particular
earlier stage.

- Alameda's retained `DSC07333`–`DSC07342` have center differences of about
  0.20–0.93 reference median steps and rotation differences around 0.8°.
  `DSC08181`/`DSC08182` differ by roughly 3.36/3.10 steps and 20.6°/20.7°.
- London's retained missing poses differ by about 0.25–4.89 reference steps,
  while their rotation differences are only about 0.17–0.70°. Translation
  consistency is a much clearer problem here than gross rotation failure.

Thus blindly restoring their registration flags would restore demonstrably
poor retained poses. The branch gate should stay until supported replacements
are available. Reference agreement is a comparison, not an absolute truth test.

The positioning logs also show different upstream behavior: Alameda accepts
its point-bearing refinement; London rejects it for translation-graph
regression (layout change P90 about 4.39 local baselines) and retains the camera
warm start. Both runs select 20,000 positioning tracks with at least four views.
This motivates measuring per-branch constraint coverage and consistency, not
simply weakening the warm-start acceptance guard.

## Offline inactive-correspondence trial

Preserved inactive boundary matches were supplied as tentative 2D–3D search
links to the existing frozen-anchor resection validator. No inactive edge was
reactivated. Original registered poses remain exact. Existing landmark
coordinates/observations can be recomputed by this offline function, so this
trial does **not** claim frozen original landmarks.

- Alameda: one additional candidate, `indoor_DSC07327.JPG` (1712 in the offline
  result). It has 28 fit inliers and 14/14 local holdout inliers. Reference pose
  differences are about 0.0199 steps and 0.089°.
- London: zero additional registrations. `DSC07840` has 11/13 holdout inliers
  and remains rejected by the unchanged 12-inlier minimum.

The Alameda candidate was tested on every eligible reference track observed
by it and at least three baseline-registered cameras. It was excluded from
every point fit; no residual/depth filtering was applied. On 1123 observations,
native mean/median/P95 error is **4.052/1.875/9.441 px**, versus reference
**1.315/0.973/3.588 px**. The native RMS is 12.701 px, with a 233 px extreme.
This external test depends on both recovered and baseline camera geometry;
it cannot assign all error to the candidate alone. Nevertheless, local
acceptance and close pose agreement do not establish reference-quality
geometry. The inactive-link path remains offline and is not a default gain.

## Next experiment justified by these results

1. Trace per-camera positioning coverage and boundary-track survival through
   calibrated verification, rotation filtering, global track selection and
   weak-branch cleanup. Keep original-anchor ancestry distinct from subsequently
   recovered cameras.
2. Choose the connected Alameda missing run for a bounded independent local
   initialization with stable-map connections at its available boundaries.
   Preserve inactive matches as proposals and distinguish initialization
   failure from stable-map alignment failure. A chain-compatible initializer
   should be tested against the prior star-initialization failure.
3. Require whole-segment, paired external reprojection and pose-tail checks
   before adopting a coverage gain. Existing anchor errors and candidate errors
   must be separated; a small local holdout alone has already proved inadequate.

Do not prioritize more global BA iterations, a wider ratio threshold or
lowering acceptance gates: none addresses the measured absence/inconsistency
of boundary depth. Validated product coverage remains 1711/1734 and 1858/1874.

Mapping (including both new failure regressions), submap recovery, checkpoint
and Studio alignment CTests pass: four tests, zero failures. `summary.json`,
`manifest.json`, `audit_manifest.json` and per-scene CSV/logs retain the evidence.

## Follow-up: branch tracing and joint initialization

The next experiment is completed in [SFM_BRANCH_CONSTRAINT_EXPERIMENTS.md](SFM_BRANCH_CONSTRAINT_EXPERIMENTS.md). It separates real stable-map boundaries from other withheld neighbors, attributes London's positioning starvation to short-track qualification rather than the 20k cap, and validates joint local initialization without counting unanchored local cameras as registrations. Real-data accepted recovery remains zero in this follow-up. The full traced reruns also reveal one fewer registration per dataset and record that reproducibility limitation explicitly.
