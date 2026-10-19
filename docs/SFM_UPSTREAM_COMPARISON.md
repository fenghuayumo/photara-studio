# Alameda / London: COLMAP and OpenMVS source comparison

Date: 2026-10-19. Photara baseline: `18c372b`; experimental changes remain opt-in.
The ordered prototypes and measured follow-up are now recorded in
[SFM_BOUNDARY_SEQUENCE_EXPERIMENTS.md](SFM_BOUNDARY_SEQUENCE_EXPERIMENTS.md).
This is a source review and Photara cache replay, not a new upstream reconstruction
benchmark. The software/version/settings used to create existing reference models
are unknown; their coverage cannot be attributed to the mechanisms below.

## Reproducible source scope

Official repositories, pinned during this review:

- COLMAP `main`: `f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36`.
- OpenMVS `develop`: `610562a4f033be2aaf4f240512c93e1038186768`.

Downloaded files and SHA256 values are recorded in
`artifacts/sfm-upstream-review-20261019/manifest.json`. These describe the pinned
development sources, not every released version. An optional fetch of a nonexistent
COLMAP `scene/reconstruction_alignment.cc` returned 404; it is not used as evidence.

## COLMAP: grow structure and revisit registration

The incremental mapper retains a geometrically verified correspondence graph.
Ordinary registration collects deduplicated 2D–3D candidates through registered
neighbors, then robustly estimates and refines the target pose. Registration is
interleaved with triangulation and local BA. Global refinement completes/merges
tracks and retriangulates before BA, repeats completion/filtering, and gives a
stalled reconstruction another refinement-and-registration opportunity.

`incremental_triangulator.cc` makes the distinction important: completion traverses
the persistent correspondence graph, admits only registered valid cameras, and
checks actual projection before adding an observation. Merging checks observations
from both tracks against the merged point. Retriangulation revisits underused
verified pairs. Defaults include completion transitivity 5, completion/merge 4 px,
and retriangulation when triangulated/verified matches falls below 0.2. This is not
blind union of all raw matches. Defaults also ignore two-view-only tracks, so
claiming COLMAP succeeds simply by accepting every short track would be incorrect.

Sources: [mapper](https://github.com/colmap/colmap/blob/f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36/src/colmap/sfm/incremental_mapper.cc),
[triangulator](https://github.com/colmap/colmap/blob/f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36/src/colmap/sfm/incremental_triangulator.cc),
[defaults](https://github.com/colmap/colmap/blob/f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36/src/colmap/sfm/incremental_triangulator.h).

### A separate route when existing 3D support is absent

The pinned pipeline enables `structure_less_registration_fallback`. Its mapper
can estimate a single target pose directly from target-to-registered-image 2D–2D
matches and the registered camera poses. It does not require a complete local
target submap or preexisting common 3D points. It then continues existing tracks
or triangulates new points, and refines the pose using the resulting structure.

At the pinned defaults it needs at least two registered images and 60 inlier
2D–2D constraints (twice the ordinary 30-inlier requirement), with a 0.25 minimum
inlier ratio and 6 px epipolar error threshold. Those are constraint counts, not
necessarily 60 unique target features. Epipolar errors are not interchangeable
with our final pixel reprojection or independent validation thresholds. The
generalized estimator uses 5+1 or 6-point solvers; this differs from the gp4ps
local-3D-to-anchor-ray group alignment we have already tested.

Sources: [structure-less mapper](https://github.com/colmap/colmap/blob/f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36/src/colmap/sfm/incremental_mapper.cc#L642),
[generalized estimator](https://github.com/colmap/colmap/blob/f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36/src/colmap/estimators/generalized_pose.cc),
[pipeline options](https://github.com/colmap/colmap/blob/f9346834bd8bf5f9d59b8f9cc2fddc35aa521b36/src/colmap/controllers/incremental_pipeline.h).

## OpenMVS: weak seams, hierarchical alignment, and resection

The pinned development tree contains native SfM in `libs/SFM`, including hierarchical
reconstruction; describing this version as only consuming another tool's poses
would miss relevant code. The hierarchical path clusters images, reconstructs local
scenes, aligns them, and runs further resection/refinement. Star initialization
defaults to at least four views. Resection uses existing 2D–3D support, triangulates
between growth steps, performs local/full BA, filters, and retries after progress.
It also stops if a selected batch registers no images; upstream is not immune to
candidate-selection or insufficient-support failures.

`SceneCluster.cpp` explicitly splits thin-waist clusters and avoids merging across
weak interfaces. This addresses a local reconstruction containing two strongly
connected blocks with a poorly constrained seam, where a single Sim(3) may not
correct internal deformation. It suggests a diagnostic for Alameda; it does not
prove that Alameda's failures are caused by this deformation.

`GlobalAlignment` obtains Sim(3) from shared triangulated points, with defaults of
25 common tracks, 30% inlier ratio and a threshold based on 1% of destination bounding
box diagonal. It validates averaged placement through cycle consistency and can
discard a subscene's placement while preserving observations for resection. Track
merging limits reinsertion of rejected internal observations and reprocesses cross-
subscene pairs. Its lower nominal ratio cannot be copied into Photara: point sets,
spatial thresholds, conditioning checks and validation differ.

Sources: [resection](https://github.com/cdcseacave/openMVS/blob/610562a4f033be2aaf4f240512c93e1038186768/libs/SFM/Resection.cpp),
[clustering](https://github.com/cdcseacave/openMVS/blob/610562a4f033be2aaf4f240512c93e1038186768/libs/SFM/SceneCluster.cpp),
[alignment](https://github.com/cdcseacave/openMVS/blob/610562a4f033be2aaf4f240512c93e1038186768/libs/SFM/GlobalAlignment.cpp).

The global positioner allows three-view tracks by default, whereas our selected
positioning constraints require four views. London has 116 three-view cross-cohort
track incidences in the earlier diagnostic, but only four eligible four-view ones.
These are incidences, not 116 independent good landmarks. A conditional boundary
three-view experiment is justified; globally lowering the requirement is not yet.

Source: [positioning defaults](https://github.com/cdcseacave/openMVS/blob/610562a4f033be2aaf4f240512c93e1038186768/libs/SFM/GlobalPositioning.h).

## What the Photara experiments establish

Validated parents remain Alameda 1711/1734 and London 1858/1874. Restoring raw
native-calibration proposals did not register additional images. The group ray
method also rejected the real segments: London's local reconstruction supplies
only seven usable boundary tracks; Alameda's ten-camera segment has 120 candidate
ray tracks but only 67/149 fit observations agree with its best transform, below
the required consensus. Segment BA is never reached for these failures. Its
synthetic success therefore cannot be claimed as a real-data improvement.

The corrected read-only calibration audit explicitly trusts both sets of supplied
intrinsics, keeping solver selection constant:

| Segment | Tested boundary pairs | Existing subset / native 15 | Raw / native 15 | Raw / approximate 15 | Raw / native 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alameda 07333–07342 | 320 | 6 | 24 | 25 | 6 |
| Alameda 08180–08184 | 160 | 28 | 44 | 44 | 28 |
| London 04895–04920 | 160 | 0 | 13 | 13 | 0 |
| London 04985–05010 | 192 | 0 | 10 | 9 | 0 |

These are accepted pair hypotheses, not certified registrations or unique feature
counts. Old audit values mixed camera trust flags and calibration; outputs are
preserved, and the plan's table is corrected. Actual native proposal experiments
already used trusted calibration and are unaffected. Evidence:
`artifacts/sfm-calibrated-audit-20261019/summary.json` and per-pair CSV; both logs
confirm `parent_geometry_unchanged=1`.

## Revised implementation priorities and acceptance

1. **London: direct structure-less single-image registration.** First export unique
   target-feature counts, registered-anchor coverage/baselines and ambiguity from
   the corrected native proposals. Then prototype 5+1/6-point robust estimation with
   frozen anchors. Deduplicate constraints, exclude invalidated poses, test scale
   observability and repetitive/planar cases, triangulate only after a pose hypothesis,
   and require disjoint validation before committing. This bypasses the requirement
   that all missing cameras already form a complete local 3D submap. It cannot bypass
   genuinely insufficient or false boundary matches.
2. **Geometry-gated track completion and retriangulation.** Retain verified match
   provenance separately from the active pose graph. Complete/merge using current
   stable poses, pixel and depth/parallax checks; measure added stable 3D support
   before PnP. Photara already extends some existing tracks, but lacks the same
   persistent-graph completion/merge loop. Never reactivate every invalidated edge
   as a pose constraint or blindly union tentative matches.
3. **Alameda: diagnose and split weak boundary windows.** Export per-track anchor
   identities, triangulation angles, ambiguity, pixel errors and transform residuals;
   compare leave-one-anchor-out fits and shorter overlapping windows with stable
   context. Depth-sensitive residuals indicate conditioning; window-dependent
   transforms suggest shape drift; feature conflicts suggest associations. Only
   after valid placement run frozen-boundary segment BA and independent holdout.
4. **Conditional three-view boundary constraints** as a separate ablation, with
   parallax, spatial coverage and cycle checks. Preserve the global four-view
   default until paired target and unrelated-dataset trials show acceptable results.

For promotion, compare identical cached features/matches/calibration and repeat
full-frontend runs. Report exact gained/lost cameras, independent pixel error tails,
shared-camera pose tails, cost and frozen-anchor equality. A gain in registration
count alone is insufficient. Existing references remain evaluation-only. No default
solver behavior was changed by this source review or the audit correction.
