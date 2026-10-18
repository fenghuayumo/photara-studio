# SfM orbit prior and alignment quality

Default reconstruction leaves camera trajectories unconstrained during final
bundle adjustment. Detecting a complete circular capture does not establish
constant angular speed, constant height, a fixed radius, or a common look-at
point. Handheld motion and uneven turntable sampling must remain recoverable
from image observations.

`GlobalPositioningOptions::regularize_complete_orbits` is therefore **false** by
default. This applies to the CLI, Studio, and hierarchical mapping. The existing
hard orbit projection remains an explicit C++ experiment for captures with a
known uniform circular trajectory. It is not a general camera repair strategy.

On the 109-image `chuan` capture, fixed-frontend ablation reduced shared-track
held-out mean reprojection error from 2.322 px to 0.919 px when this prior was
disabled; the supplied COLMAP binary model scored 0.682 px on the same test.
All 109 cameras remained registered. This validates a geometry improvement,
not a measured 3DGS rendering improvement or equivalence to COLMAP.

`photara_mapping_test` includes a nonuniform 24-view circular capture. It checks
registration, retained observations, reprojection error, unequal camera steps,
and relative rotations using the production defaults.

Reconstruction checkpoint keys include this option and the SfM implementation
fingerprint, so an old reconstruction is not reused when Align Photos runs with
the updated defaults. Previously saved `.asfm` / `.ascan` cameras remain saved
results: rerun Align Photos to replace their alignment before training.

For offline comparisons, `photara_sfm_reconstruction_benchmark` supports:

- `PHOTARA_BENCHMARK_ENABLE_ORBIT=1`: reproduce the former hard orbit prior.
- `PHOTARA_BENCHMARK_DISABLE_ORBIT=1`: explicitly disable it (also the default).
- `PHOTARA_BENCHMARK_EXPORT_COLMAP=NEW_DIRECTORY`: export the result for evaluation.

If both orbit switches are set, disable takes precedence. No reference model
enters reconstruction. `scripts/sfm_quality_compare.py` evaluates binary models
in preference to text and measures common camera poses and shared-track errors.
Its held-out observations are excluded from each fresh point fit, but the camera
solutions themselves were optimized beforehand. Reference-selected tracks and
COLMAP comparisons do not establish absolute ground truth.
The evaluator also reports adjacent relative rotations, center differences
normalized by the RMS camera radius, and per-camera held-out observation errors
with sample counts. These complement the trajectory and average-error summaries;
a small mean error does not rule out a few badly aligned cameras. Alternating
fit/test assignment groups observations by camera, so duplicate observations
from one view cannot leak between a point's fit and test sets.
The 2000-track random sample is supplemented to test every common camera with
eligible reference tracks; cameras without eligible tracks remain explicit in
the coverage counts and per-camera CSV.
Before comparing, reference intrinsics and feature coordinates are rescaled
together to the native model's image grid. Model dimensions can differ from the
actual supplied JPEG dimensions, so unscaled pixel errors or focal lengths must
not be compared across those grids. The transforms are recorded in metrics.json.

The final global BA stage also relaxes the untrusted initialization focal prior
to weight 0.05, uses a 1 px Huber transition, and permits up to 100 iterations
before the existing bounded continuation checks. Early geometric BA retains
its stronger prior and calibrated intrinsic groups remain locked. This avoids
turning an inaccurate view-graph focal into a persistent calibration bias.
The synthetic regression includes varied camera height and scene depth for
self-calibration; a planar yaw-only orbit cannot independently identify fy
and the scene's vertical scale.

For a downstream 3DGS check, train with `--splat-eval-split-every 0` to include
every image, then use `photara_splat_quality_eval DATASET IMAGES MODEL OUTPUT 1`
to render and score every camera. These are training-set fit metrics; the CLI's
default three-view summary is not an all-image score. Camera calibration may
also change the undistortion grid, so compare against the original image grid
when attributing PSNR differences across camera solutions.

## Large-scene positioning and missing views

The existing large-scene camera warm start is now retained when point-bearing
refinement materially degrades the translation graph **and** moves the bulk
camera layout by more than one typical local graph baseline, without a
commensurate independent point-fit gain. Layout movement is measured after a
trimmed similarity fit, so a gauge change or one isolated bad camera cannot
trigger a whole-scene fallback. Small layout refinements may still move away
from noisy relative translations. Held-out angular regression tolerances also
respect the camera's pixel scale. These checks compare native candidates;
they do not read a reference reconstruction or certify absolute accuracy.

Late missing-view recovery retains a fixed 16-anchor budget, combining verified
visual overlap with temporal neighbors. Inactive verified edges can nominate
search neighbors, but cannot directly register a camera. Missing runs are
visited from their boundaries, keeping separate runs together for bounded-cache
reuse. A small PnP consensus only seeds guided descriptor matching. Its 8 px
search radius is not an acceptance threshold: full poses must still
pass the existing 2 px resection, independent-depth, spatial-conditioning and
held-out support checks. Local descriptor-map recovery requires at least 18
fit inliers, 12 held-out inliers, 30 final observations, and three depth anchors
in both fit and validation. Depths use frozen registered cameras only.

Local descriptor-map fit/validation observations are stratified by the cameras that supplied each
independent depth. The split is deterministic and uses no candidate residuals;
it prevents a rare anchor group from disappearing from one side of the split.
Ordinary recovery keeps its existing shuffle and three-camera-per-depth rule.
Failed candidate poses never change the frozen anchors or calibration.

Up to 32 weak registered leaves with extreme local scale breaks can undergo
the same independent re-localization, with all other registered cameras fixed.
Failed proposals restore their original poses; the normal final registration
audit remains responsible for removing unsupported cameras. This keeps a
nomination heuristic from silently accepting a replacement or deleting a view.

`photara_sfm_stable_resection_probe ... --groups` is a separate offline
experiment for jointly reconstructing missing connected runs. It uses native
calibration and the existing independent submap alignment/holdout checks, and
does not run in Photara's default reconstruction path.

## Missing-run anchor search (2026-10-18)

Further bounded boundary recovery trials are recorded in
[SFM_BOUNDARY_RECOVERY_EXPERIMENTS.md](SFM_BOUNDARY_RECOVERY_EXPERIMENTS.md).
They increase Alameda's independent shared-point count but fail geometric
consensus; no additional default registration gain is established.

Late recovery now also nominates registered anchors through a verified
target-to-missing-to-registered graph path. Its score is half the weaker edge
score. This changes search candidates only: no bridge pose, depth, or reference
geometry enters registration. The total anchor budget stays at 16, with at
least half the slots available for temporal neighbors. Every nominated anchor
is freshly descriptor-matched and the existing absolute-pose, independent-depth,
spatial and held-out checks remain in force.

On the saved native final graphs from the `layout` baseline, one production
missing-view recovery invocation produced:

| Scene | Before | After | Remaining |
| --- | ---: | ---: | ---: |
| London | 1856/1874 | 1858/1874 | 16 |
| Alameda | 1706/1734 | 1711/1734 | 23 |

London added `indoor_DSC06325.JPG` and `indoor_DSC06330.JPG`; Alameda added
`indoor_DSC07328.JPG` through `indoor_DSC07332.JPG`. The probe checked all
original registered camera poses, intrinsics, and original landmark/observation
contents for exact preservation. Post-run comparison to the supplied reference
gave the seven added cameras center errors of 0.0052–0.0363 reference median
steps and rotation differences of 0.043–0.225 degrees. References were read
only by evaluation, never by recovery. These are incremental final-map recovery
measurements, not fresh end-to-end frontend or 3DGS quality measurements.

The wider descriptor-ratio experiment, geometry-filtered proposal/multi-view
depth candidate, and relative-ray search seed each added zero views on these
two saved graphs before the two-hop anchor change. Increasing the number of
tentative descriptor matches alone did not fix these missing runs.

Evidence is under `artifacts/sfm-recovery-20261018/`: `bridge_manifest.json`
records inputs, binary/source hashes, execution results and reference scores;
each scene has the saved model, accepted-name list and full recovery log.
`new_view_holdout.json` evaluates every eligible reference track seen by a new
camera and at least three original cameras, fitting points with original
cameras only and excluding every newly recovered camera. This avoids the very
small per-camera sample counts in the standard random-track summary. Neither
test removes observations based on residuals. Changing coverage also changes
the standard random-track sample, so its before/after RMS is not a paired
quality comparison; original camera solutions remain exactly unchanged.

The dedicated new-camera test has 50/45 observations for London's additions:
native mean 2.50/1.49 px versus reference 2.27/1.42 px. Alameda's five additions
have 408–917 observations each: native mean 2.92–3.60 px versus reference
1.26–1.78 px. Thus the extra coverage has useful pose agreement, but Alameda's
recovered segment still has a material independent reprojection gap. The new
cameras have not been shown to match reference quality or improve 3DGS output.

Mapping, submap recovery, checkpoint and Studio alignment CTests pass. The
mapping regression checks that a missing bridge can nominate a distant
registered anchor within the fixed budget, while never becoming an anchor
itself. Search-seeding header changes now participate in both frontend and
reconstruction cache fingerprints.
