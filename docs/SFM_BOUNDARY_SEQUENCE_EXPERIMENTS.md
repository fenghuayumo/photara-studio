# Ordered boundary recovery experiments

Date: 2026-10-19. Follows [the upstream comparison](SFM_UPSTREAM_COMPARISON.md).
All new solvers and diagnostics are developer-tool experiments. Default SfM
implementation is unchanged relative to `18c372b`.

## Outcome

No candidate is promoted to the default reconstruction or counted as a validated
coverage gain. Parent coverage remains London 1858/1874 and Alameda 1711/1734.
The two cached main missing groups cover 11 London and 15 Alameda target cameras;
these trials do not examine every remaining missing camera or run a fresh frontend.

Direct structure-less estimation produces no accepted real pose. Track completion
adds a few observations but does not explain a recovered camera. One Alameda camera,
DSC08181, passes the existing stable-depth resection in both the native-proposal
control and completion variants. Independent evaluation reveals a substantial pixel
precision gap, so its diagnostic 1712-camera snapshot is not promoted.

## 1. Direct 2D–2D registration

Implementation: `experiments/sfm_structureless_probe.hpp`, exposed by
`photara_sfm_stable_resection_probe --structureless` with the saved local cache.

The estimator uses PoseLib's generalized relative 5+1 solver with registered camera
origins and world rays on one side and target-camera rays on the other. Translation
is normalized by anchor baseline. It never reads a retained target pose or existing
target 3D point. This is a prototype inspired by the reviewed COLMAP route; it does
not reproduce its complete estimator, and no 6-point fallback is implemented yet.

Candidate collection excludes conflicting feature associations and deduplicates
target-feature/anchor pairs. Every fifth target feature is entirely held out,
including all its anchor observations. Sampling uses five distinct features from
one anchor plus a different feature from another anchor. Deterministic RANSAC uses
4096 trials, a 4 px two-ray reprojection proposal gate, positive depth and at least
one-degree parallax. At least 30 proposal inliers and 25% fit support are needed to
attempt epipolar refinement. Initial hypothesis generation can run with 30 fit
constraints, 40 unique total features and 12 held-out constraints.

Certification remains stricter: 60 fit observations at 2 px; 80% observation and
whole-feature consensus; at least 12 held-out features and 80% held-out consensus;
three supported anchors in each split; target spatial coverage; a conditioned
six-parameter information matrix; and nonplanar triangulated structure. Anchor
poses and calibration are fixed. Acceptance is a pose hypothesis, not automatic
production registration. Held-out triangulation uses the candidate pose, so this
is independent pose-data validation, not independent stable-only depth validation.

London's final paired control results:

| Target | Unique features | Anchor cameras | Fit observations | Best 4 px proposal support | Corresponding 2 px support |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSC04920 | 116 | 7 | 94 | 22 | 14 |
| DSC05000 | 62 | 4 | 49 | 16 | 12 |

These two fail before refinement. The other nine targets fail support requirements;
many see only one anchor. More accepted pair hypotheses therefore do not imply a
jointly consistent, observable target pose. Both control and completion runs accept
zero structure-less poses. Zero holdout/condition fields in early-rejection rows
mean the corresponding stage was not reached.

Evidence: `artifacts/sfm-boundary-sequence-20261019/london/{control,completion}/`.
Per-target CSV records feature counts, baselines, coverage and rejection stages;
`.anchors.csv` records the individual anchor support and fit/holdout counts.

## 2. Geometry-gated completion, merging and retriangulation

Implementation: `experiments/sfm_track_completion.hpp`, enabled with
`PHOTARA_TRACK_COMPLETION=1` in the developer probe.

The local control rebuilds tracks from the same native-calibration verified pairs
and triangulates with frozen registered cameras. Completion traverses their
correspondence graph up to five hops and adds only registered observations whose
actual projection is within 2 px. A merge requires unique image observations and
all existing inlier observations to pass against the merged point. Ambiguous
ownership and conflicting image features are rejected. Untriangulated tracks are
revisited with at least three inliers and one-degree parallax. Unregistered poses
never supply depth. The existing stable resection validator runs on separate copies
before and after completion, without changing the parent reconstruction.

| Cached group | Added stable observations | Merged tracks | Retriangulated tracks | Target depth support change |
| --- | ---: | ---: | ---: | --- |
| London 04895–04920 | 0 | 0 | 0 | None |
| London 04985–05010 | 7 | 0 | 0 | None |
| Alameda 07333–07342 | 3 | 0 | 0 | None |
| Alameda 08180–08184 | 5 | 0 | 0 | DSC08182: 250 → 251 |

Counts are potential unique feature/landmark pairs, not certified independent
depths. The stricter existing resection check finds fewer usable independent
depths. London resection accepts no camera. Alameda DSC08181 is accepted before
and after completion; its pose-only ASFM exports are byte-identical. This isolates
the result to the native-proposal single-image route, not completion.

These local trials measure gains above a fresh native-pair track rebuild. They
do not establish whether preserving the full production correspondence history
across every BA/filtering stage would help. No inactive pose edge is globally
reactivated, and the persistent-graph production integration remains unimplemented.

## 3. Alameda windows and anchor diagnostics

`experiments/sfm_boundary_windows.hpp` runs independently initialized overlapping
windows and leave-one-anchor-out fits. `sfm_ray_group_alignment.hpp` can now export
per-track local pixel error, target-camera span, anchor identities, stable-only
triangulation support/parallax, normalized depth difference and boundary residual.
Residual export does not change the existing acceptance decision.

| Window | Boundary tracks | Fit ray observations | Best fit inliers | Result |
| --- | ---: | ---: | ---: | --- |
| Whole ten-camera run | 120 | 149 | 67 (45.0%) | Consensus fails |
| First six cameras | 118 | 141 | 72 (51.1%) | Consensus fails |
| Middle six cameras | 18 | 17 | — | Insufficient support |
| Last four cameras | 0 | 0 | — | Insufficient support |

No single-anchor removal restores the 80% consensus requirement; the strongest
remaining ratio among fits reaching RANSAC is 47/90 (52.2%). Some removals instead
lose spatial support. Splits change local reconstruction and eligible track sets,
so these ratios are descriptive, not a paired per-point causal test.

The whole-run diagnostic has 120 boundary tracks / 179 observations including
holdout. Of these, 82 tracks see only one anchor and 36 obtain a stable-only depth
under the 2 px triangulation gate. Local internal maximum-error median is 0.743 px
and P95 is 1.735 px, but these are selected triangulated inliers and do not prove
correct local geometry. Boundary disagreement persists across several anchors.
The results neither isolate a single bad anchor nor distinguish depth conditioning,
repetitive-texture associations and local deformation conclusively. Stable boundary
support is particularly poor toward the end of the run.

Evidence: `model.asfm.windows/892/windows.csv` and per-trial residual CSVs under
`artifacts/sfm-boundary-final-20261019/alameda/completion-windows/`.

## Independent check of DSC08181

The unchanged stable resection accepts this camera with 38/72 fit inliers and
15/36 held-out correspondences. These are internal matching/depth checks. The
pose-only snapshot has 1712 registered flags, but is not a complete reconstruction.

Reference comparison gives center error 0.048896 reference median camera steps and
rotation error 0.160921 degrees. Those modest pose differences do not establish
pixel precision. A separate test fits fresh points only in at least three original
registered cameras, excludes the added camera entirely from those point fits,
and evaluates every eligible reference-selected target observation without residual
or depth filtering. Reference intrinsics and observations use the native image grid.

| Same 603 held-out observations | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Candidate | 7.113 px | 6.999 px | 13.706 px |
| Reference | 1.541 px | 1.192 px | 3.981 px |

No nonfinite observations are omitted. This reference-selected comparison is not
absolute ground truth, but it demonstrates a material precision gap on a reasonably
large independent set. Do not promote this candidate or advertise a coverage gain.
Independence here means reference files are evaluation-only and the fresh 3D point
fits exclude the added camera. The camera pose was already estimated beforehand;
we have not proved that every reference-selected target pixel is distinct from
every feature measurement used during registration. These are not 603 certified
unseen pose-estimation measurements.
It also shows that passing the current low-support resection holdout is insufficient
to certify precision in this case. The direct structure-less validator rejects it.

The holdout script initially rejected an unrelated older bridge model as a frozen
control: its 1711 registered flags match the replay parent, but its poses differ.
The exact replay baseline and the new parent export agree. All 1711 original poses
in the candidate snapshot have exactly zero parsed pose difference from that parent.
Use the exact checkpoint/export pair for future comparisons rather than assuming
equal coverage identifies an identical control.

Independent metrics are in `artifacts/sfm-boundary-final-20261019/candidate-holdout.json`
and `candidate-reference.csv`. `validation.json` records equality of the final
exports and the independently evaluated earlier exports, so metrics can be reused
without rerunning an identical evaluation.

## Validation, reproduction and next work

Synthetic tests cover native pinhole/fisheye structure-less pose recovery with no
existing points and an intentionally wrong retained pose; single/coincident-anchor
degeneracy; conflicting feature assignments; corrupted whole held-out features;
planarity; valid completion/merge/retriangulation; and geometrically invalid unions.
Seven relevant CTests pass; the final export changes also pass the focused submap
test. The analysis loader now accepts empty COLMAP observation lines, necessary for
pose-only snapshots; the independent evaluator successfully reads those snapshots.

The default benchmark SHA256 remains
`b6571747df100c99ef7e11063788f2988fd03fd84a11944dfd796221cce59a69`.
Default SfM source diff is empty. This isolates current experiments from unrelated
datasets; it is not a quality guarantee for a future production integration.

```powershell
cmake --build build --config Release --target photara_sfm_stable_resection_probe photara_submap_recovery_test
python scripts/sfm_boundary_sequence_trial.py NEW_OUTPUT
# Optional single-scene replay:
python scripts/sfm_boundary_sequence_trial.py ANOTHER_OUTPUT alameda
```

Manifests record binary/source/checkpoint/cache hashes, process resources and per-
target outcomes. Earlier whole-sequence results, the attribution replay and final
exports remain in separate artifact directories. No parent checkpoint is overwritten.

Next priorities are additional verified boundary matches to distributed stable
anchors, especially for London's sparse targets and Alameda's tail; independent
cross-anchor precision checks for weak resection; and distinguishing associations
from depth/shape errors before deformation refinement. More BA iterations or
loosening final gates is not supported by these results. Reference observations
remain evaluation-only and must not become a reconstruction prior.

Follow-up: [cross-anchor precision experiment](SFM_CROSS_ANCHOR_PRECISION.md) adds
component-disjoint pose fitting and a withheld stable-anchor depth check, together
with an inventory of how little independent validation support the caches provide.

The later [cold frontend analysis](SFM_COLD_FRONTEND_ANALYSIS.md) starts over from
all original images and identifies calibration-order and correspondence-selection
losses. Its London control is 1857, distinct from this report's 1858-camera frozen
parent; its rejected candidate must not be counted as a validated coverage gain.
