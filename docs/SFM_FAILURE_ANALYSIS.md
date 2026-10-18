# Alameda / London registration failure analysis — 2026-10-18

This is a code and existing-evidence review, not a new reconstruction trial.
The follow-up measured trials are recorded in [SFM_RESECTION_EXPERIMENTS.md](SFM_RESECTION_EXPERIMENTS.md).
The baseline here is the validated two-hop anchor recovery (`bridge`), not
the subsequently rejected disposable-depth refinement (`joint`). References
are comparisons, not ground truth, and do not enter reconstruction.

## Assessment

| Scene | Registered | Missing | Missing fraction | Largest missing run |
| --- | ---: | ---: | ---: | ---: |
| Alameda | 1711/1734 | 23 | 1.33% | 10 input images |
| London | 1858/1874 | 16 | 0.85% | 6 input images |

An absolute limit of ten missing images is not a useful general quality rule:
image redundancy, input size, geometric conditioning and spatial coverage all
matter. However, these images occur in the supplied reference reconstructions,
and 32 of the 39 remaining failures lie in multi-image missing runs (19 Alameda,
13 London). That makes the gaps actionable evidence of a pipeline limitation,
not grounds to dismiss them as unavoidable bad input. Reference success does
not establish that all views can pass Photara's current quality requirements.

The full alignment logs also change the diagnosis materially: the initial
global solve contained 1728 Alameda and 1871 London cameras. Subsequent support
pruning removed 6/7, and weak-branch validation withheld 21/13 respectively,
before later recovery. Of the **currently remaining** 23/16 views, 16/11 appear
in the weak-branch withheld lists, and another 2/2 in final-support prune lists.
Thus 27/39 remaining failures had poses before independent branch validation
withdrew them. The other eight have no per-image attribution in these drop
logs. It would be wrong to equate these failures with absent SIFT detections,
or to bypass branch validation merely to retain the initial high count.

The previous gain of seven cameras came from changing anchor search while
retaining acceptance thresholds. This is stronger evidence for missing usable
cross-boundary support than for globally over-strict numerical thresholds.

## Pipeline reviewed

1. SIFT extraction, spatial sampling, descriptor compression and optional masks.
2. Sequential + BoW candidate retrieval, descriptor matching and E/F/H checks.
3. Weak-image feature/pair rescue, focal calibration, calibrated re-verification.
4. Weighted track unions, rotation averaging and a second track build at weight 3.
5. Coverage-aware capped global positioning, triangulation and staged global BA.
6. Incremental resection, weak-branch and stable-depth recovery, final pruning.
7. Late local descriptor-to-map recovery and final support/position audits.

The reviewed entry points are `frontend.cpp`, `retrieval.cpp`, `geometry.cpp`,
`tracks.cpp`, `global_rotation.cpp`, `global_positioning.cpp`, `triangulation.cpp`,
`resection.cpp`, `star_init.cpp`, `submap_recovery.cpp`, `bundle.cpp` and
`reconstruct.cpp`, with their public configuration headers. This is a focused
review of the alignment path, not an exhaustive audit of the application.

## Existing failure evidence

`artifacts/sfm-recovery-20261018/analyze_failures.py` reads the existing bridge
logs and evaluation lists and writes `failure_analysis.json`.

| Last logged late-recovery rejection | Alameda | London |
| --- | ---: | ---: |
| Pose consensus failed | 12 | 6 |
| Insufficient independent depths | 7 | 10 |
| Insufficient fit/inlier/validation anchor diversity | 3 | 0 |
| Held-out consensus failed | 1 | 0 |

These are **last rejection reasons**, not first-cause attribution. A view has
multiple attempts, feature budgets and proposal sets. In particular, a logged
PnP inlier count of zero does not prove zero geometric inliers: the absolute
pose wrapper returns an empty failure result before copying sub-threshold
PoseLib statistics. Existing logs also do not provide a complete per-image
history from frontend through every prune/recovery stage.

Alameda has a ten-image gap `DSC07333`–`DSC07342` and a five-image terminal gap
`DSC08180`–`DSC08184`; London has five- and six-image gaps near `DSC049xx`.
Run lengths refer to consecutive supplied input images, not numeric filename
increments or measured distance in space.

## Prioritized directions

### 1. Diagnose the weak-branch boundary and loss of trustworthy support

This has the strongest direct scene evidence (27/39 remaining views were
withheld). Capture the pre-withdrawal poses and compare them independently to
the retained map/reference in evaluation. Count surviving cross-boundary
matches and tracks before/after calibrated geometry, rotation filtering,
the weight-3 track rebuild and weak-branch cleanup. That distinguishes actual
branch drift, missing independent support and false-positive withdrawal.

Weak-branch recovery tries stable-depth propagation before withdrawal; afterward
it deactivates all edges incident to unregistered views and rebuilds tracks.
Final late recovery can use inactive edges as search nominations but cannot
reuse them as registration evidence. This interaction is central to inspect.
Keep the validation gate until a replacement has independent evidence.

### 2. Correct correspondence bookkeeping and candidate-band starvation

Two concrete implementation problems deserve targeted tests before more
expensive algorithm changes:

- `collect_pose_correspondences` inserts a feature into `used_features` before
  finding out that its proposed track is already used. A rejected `(feature,
  track)` can therefore block a later valid alternative for that feature.
  Example: accept `(f1,t1)`, reject `(f2,t1)`, then incorrectly reject `(f2,t2)`.
  Membership checks should be non-mutating until both endpoints are free.
- `select_next_images` retains only candidates with correspondence counts at
  least 0.3 of the current maximum. If every candidate in this band fails,
  `register_images` exits when no registration progressed. Lower-count but
  geometrically valid candidates can remain unattempted. A fallback over the
  remaining eligible candidates after band exhaustion can retain all existing
  PnP acceptance requirements.

The control-flow consequences follow directly from the code; their contribution
to these 39 failures has not yet been measured. They are small, testable fixes,
not a claim that they alone will close the gaps.

### 3. Build cross-boundary multi-view support for connected missing runs

This is the strongest dataset-specific hypothesis. Late recovery operates on
one missing target plus up to 16 registered anchors. Other missing images do
not contribute independent depth, so the interior of a run can remain without
enough anchor-only landmarks even when the run has useful internal overlap.

Prefer verified overlap, depth diversity and multiple independent connections
to the stable map over simply increasing nearest temporal neighbors. Audit
which useful pair/track evidence was lost at calibrated re-verification,
rotation filtering or the mapping track-weight threshold before changing it.
Do not make inactive edges into pose certificates.

A bounded independent submap for a genuinely connected missing run is a
plausible next algorithm experiment. It must reconstruct without inherited
target poses and pass independent stable-map fit/holdout checks. The earlier
offline group trial failed initialization, not similarity validation: the
star initializer requires four views and strong central neighbors (50 matches
per neighbor, 100 initial tracks). A sparse chain is not necessarily a star.
Investigate alternative supported seeds/growing initialization; indiscriminately
lowering acceptance thresholds is not supported by that result.

### 4. Separate registration propagation from precision refinement

Newly recovered cameras immediately enter later targets' anchor selection and
depth construction. Their geometry and ancestry are correlated. A refinement
that helps one camera can change subsequent tracks, correspondence conflicts
and poses throughout a missing run.

The disposable fit-only depth refinement provided direct evidence: London
new-view holdout mean improved 2.021 to 1.981 px, but Alameda regressed 3.290
to 3.351 px on the same 3803 external test observations. Small local holdout
improvements therefore did not certify segment-wide precision. That integration
has been removed from production; the helper/test and frozen binaries remain
offline evidence only.

Before another precision trial, keep original-anchor ancestry explicit and
evaluate the entire recovered segment. Candidate directions are delayed
promotion/refinement, independent original-map checks and a boundary-constrained
segment solve. None is yet established as better. The boundary itself must be
tested for consistency; a fixed biased boundary can preserve residual bias.

### 5. Improve frontend evidence only where the diagnosis warrants it

The default frontend uses a sequential window of 3 and retrieval top-50;
weak-view rescue has wider temporal/retrieval proposals and bounded budgets.
Its weak criterion uses local degree/nonplanar neighbors; a well-connected
isolated pocket can still have poor connections to the stable scene. Structural
rescue exists but is deliberately off after earlier quality failures.

First measure retrieval recall, feature counts, usable cross-boundary matches
and depth-support survival for the failed runs. Then choose among better
cross-component proposals, guided matching, or stronger feature matching.
The earlier wider-ratio experiment recovered zero additional views; more
candidate matches alone are not evidence of better geometry. Changing camera
models, raising global feature budgets or increasing BA iterations is lower
priority until there is corresponding diagnostic evidence.

## Before the next reconstruction experiment

Record per-image stage transitions and rejection reasons; retain sub-threshold
PnP best-support statistics; count correspondence alternatives lost to conflicts
and candidates skipped by the top-count band. Summarize per missing run as well
as per image. Some of this needs added diagnostics; the current logs cannot
retrospectively answer every question.

First validate the two bookkeeping/control-flow issues with targeted synthetic
cases. Then use one change at a time against fixed inputs with fresh cache keys.
Evaluate complete production alignment, not only incremental saved-map rescue.
Judge coverage, remaining run lengths, original-camera non-regression, recovered
pose tails and paired independent reprojection errors together. Reference paths
stay outside reconstruction. No new alignment trial was started for this review.
