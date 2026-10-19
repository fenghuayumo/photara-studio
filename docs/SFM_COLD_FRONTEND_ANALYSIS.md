# Cold frontend analysis and calibration ordering

Date: 2026-10-19. Original inputs: `D:/Models/london/images` (1874 images) and
`D:/Models/alameda/images` (1734 images).

## Scope

The control starts from original images in new, empty cache directories. It
re-extracts features, trains retrieval, matches descriptors, verifies geometry,
builds tracks and completes reconstruction. No previous features, matches, tracks
or reconstructed geometry are loaded. Reference models enter only subsequent
evaluation processes. Image content inventories, commands, source snapshots,
binary hashes, logs and resources are under
`artifacts/sfm-frontend-cold-20261019/`.
"Cold" here means no application frontend/reconstruction cache reuse; OS file
and driver caches are not flushed, so individual timings are not isolated
hardware performance measurements.

| Cold control | Registered | Frontend seconds | Complete reconstruction seconds | Center P95 / reference steps | Rotation P95 / degrees |
| --- | ---: | ---: | ---: | ---: | ---: |
| London | 1857/1874 | 314.916 | 490.342 | 0.050557 | 0.133500 |
| Alameda | 1711/1734 | 532.190 | 820.200 | 0.033925 | 0.163597 |

Both processes and their separate reference evaluations exit zero. Their logs
contain no checkpoint hits. Intermediate Alameda registration reached 1728;
final filtering/recovery finishes at 1711, illustrating why transient counts
cannot stand in for the final output.

The candidate uses the same original images and another empty cache, under
`artifacts/sfm-frontend-calibrated-20261019/`. It changes production frontend
ordering rather than invoking missing-group experimental probes. Descriptor
ratio, mutual check, epipolar/pixel/parallax gates, minimum pair support and final
registration checks retain their defaults. It is a candidate until the complete
paired results establish whether it should remain in the default workflow.

## What the cold London control establishes

The control registers 1857/1874, rather than the earlier fixed-track replay's
1858. Its center P95 is 0.050557 reference median camera steps and rotation P95
0.133500 degrees. This cold control is the comparison for the candidate; an
additional camera relative to it is not automatically a gain over historical
best coverage. Reference models are comparisons rather than absolute truth.

Primary extraction produces 1141–27000 features per image, with median 17459;
669 images reach the 27000-feature cap. The eleven previously investigated
London targets have 5983–18741 features; all occupy
16 frame cells. This measures detections, not independent or useful scene points.
Increasing the weak-view feature cap to 54000 adds **zero** features to every
one of these eleven targets. The pass appends 2162600 features to 181 other
images, but cannot increase detections in targets that were already below the cap.

Primary matching attempts 64815 pairs and retains 10520 provisional pairs.
Weak-view expansion attempts 38141 pairs at the existing rescue policy, accepts
3986 new pairs and strengthens 930. It does not follow that these provisional
edges provide correct geometric support after calibration.

Auto model selection finds only two informative probe pairs and has insufficient
confidence to replace the initial pinhole camera: focal remains 2457.6 px. Weak
image nomination, feature augmentation and rescue verification run at this prior.
Only afterwards does graph self-calibration update focal to 892.576 px. Strict
calibrated re-verification then updates 12903 of 14506 pairs and rejects 1603.
It operates on already accepted pair inliers; rejected raw proposals do not
receive another opportunity under the new calibration.

Reading the actual schema-9 **tracks** checkpoint, without altering its stage,
confirms that DSC04900, DSC04910, DSC04915, DSC04920, DSC04950, DSC04990,
DSC05005 and DSC05010 have no active edge to a camera registered in the cold
control. These losses exist before mapping and BA. Some other missing images
have one or two surviving anchor edges; their failures require separate depth
and multi-view support analysis.

## What the cold Alameda frontend establishes

Extraction produces 8402–27000 primary features per image, with median 27000;
1511 of 1734 images reach the cap. Unlike London, additional detection changes
the investigated target sets: DSC07333, DSC08181 and DSC08184 each grow from
27000 to 54000 features; DSC07342 grows to 29974. This establishes that the
budget is binding here, not that those additional features are correct matches.

Primary matching attempts 61020 pairs and accepts 9748 provisional pairs.
Rescue attempts 35601 pairs and accepts 5345 provisional hypotheses, including
retries of existing pairs. The pre-track graph contains 13454 distinct pairs.
Initial camera model selection is again inconclusive (pinhole score 0.222586,
fisheye 0.188768), retaining focal 3346.8 px. Graph self-calibration occurs after
rescue and updates focal to 1208.975 px; 1252 pairs are subsequently rejected,
leaving 12202 active pairs and 1445475 tracks at the frontend checkpoint.

The numerical similarity of the focal correction in these two datasets supports
testing the ordering change, but does not establish it as the sole source of
the remaining missing views or internal submap errors.

## Tested candidate and decision

**Do not enable this candidate by default.** Its London shared-camera precision
improves, but it loses registrations and retains a severe false placement. The
production ordering change is reverted; optional stage diagnostics remain.
The candidate executable and exact source snapshot remain in its artifact folder.
The Alameda candidate is not run after this failure; Alameda's complete cold
control and stage analysis are retained.

| London | Cold control | Candidate |
| --- | ---: | ---: |
| Final registered flags | 1857 | 1855 |
| Reference center P95 / steps | 0.050557 | 0.021381 |
| Reference rotation P95 / degrees | 0.133500 | 0.105582 |
| Same 2856 shared target pixels: mean / px | 1.409254 | 0.974398 |
| Same shared pixels: P95 / px | 3.376475 | 2.853142 |
| Complete reconstruction seconds | 490.342 | 605.475 |

The paired pixel comparison uses 1854 common cameras, 2000 seeded random
reference tracks plus coverage supplements, 2845 point-fit jobs and 2856 test
observations. Each test camera is excluded from its fresh point fit; all three
models use the same fit/test observations. Reference mean is 0.905025 px and
P95 2.539873 px. All errors are finite and no shared test point is behind its
test camera. The reference pose metrics above are per-model robust alignment
evaluations with different registered sets; the pixel comparison is the paired
same-observation result.

The candidate gains only DSC05050 and loses DSC06325, DSC07295 and DSC07300.
For the 121 eligible DSC05050 observations, fresh depths fitted only in at least
three common original cameras give candidate mean **633.476 px**, median
432.875 px and P95 1169.478 px. Reference mean is 1.445 px and P95 3.664 px.
This is a severe failure, not a coverage improvement. In contrast its own fitted
native tracks report 434 observations with mean 0.250 px and P95 0.692 px, and
the final audit retains it. Small self-reprojection residuals and graph support
therefore do not certify this registration's physical correctness.

DSC06325's lost control placement is relatively sound on the independent set:
mean 2.191 px versus reference 1.950 px (56 observations). The two other lost
control cameras have larger mean gaps (4.001 versus 1.782 px for DSC07295,
4.897 versus 1.317 px for DSC07300), but these losses cannot be silently removed
from the denominator. The candidate has a real shared-precision signal and an
unacceptable weak-camera outcome simultaneously.

Files: `london/paired-pixels.json`, `paired-pixels.csv`, `reference.csv`, full logs
and both manifests. No older DSC08181 evaluation is reused as a new result.

### Implemented candidate behavior (preserved only in artifacts)

After provisional primary verification, estimate graph/EXIF focal calibration
before nominating weak images. When calibration updates, re-verify **all original
primary raw correspondences**, using the current calibration, and rebuild the
accepted primary graph. Then choose weak views and perform rescue matching with
the calibrated geometry. Camera groups without an updated/known calibration
retain their original verification path.

The calibrated verification uses temporary copies of camera intrinsics. Persisted
`trust_intrinsics` flags remain unchanged, so inferred calibration does not lock
BA or masquerade as externally supplied calibration. No reference intrinsics or
poses enter this change.

For London, the candidate's newly extracted feature payload SHA256 equals the
control's (`43734c1ba62502e0c3cad8273a99a679f0341b9676283c5d95b0bbf20ba77400`).
The independently rematched raw-pair payload SHA256 also equals the control's
(`05ece91af919d042bd01ec6069e686f80b981e81f4eae44c50c815b7de054450`).
Original image inventories agree, and provisional pair diagnostics agree apart
from the clarified failure label. These are independently recomputed stage
outputs, not loaded controls; `input-equivalence.json` records the comparison.

The early calibration estimates 892.883 px from the primary graph. Re-verifying
its raw correspondences retains 10776 primary pairs versus 10520 provisional
pairs: 412 enter and 156 leave. Weak nomination changes from 479 to 525 images,
with 40986 rescue attempts versus 38141. These are stage effects, not certified
registration/quality gains. Several investigated missing targets still have
only internal links after the calibrated primary pass.

The candidate's final pre-track graph has 15590 pairs / 14156 active pairs.
Its calibrated focal remains 892.883 px in the later focal pass. Among the
accepted rescue hypotheses, **1366** have 20–29 inliers. All 1366 are present but
inactive in the actual tracks checkpoint: the rescue pass uses minimum 20,
while the later uniform re-verification uses minimum 30. This is a measured
policy/stage difference; it may reflect intentionally conservative staging and
does not prove that retaining these edges as pose constraints would be safe.
No acceptance threshold is lowered in this work.

The next algorithmic work should preserve correspondence provenance separately
from active pose constraints, and test weak-camera placements using additional
anchor observations and fresh target-excluded depths before admitting them.
The missing London targets' nonbinding feature caps and the catastrophic
DSC05050 example argue against using a larger generic feature budget or a
lower final threshold as the next intervention.

## Diagnostics and limitations

`--frontend-diagnostics-dir DIR` writes optional per-image feature/calibration
CSV files and per-attempt primary/rescue/calibrated pair counts. It does not
change selection or thresholds. A read-only schema-9 inventory script joins
the real pre-reconstruction graph with these stages and final camera names.
Only executed stages emit detailed CSVs; complete traces require an empty cache
or disabled cache rather than a frontend cache hit.
These detailed files cover the descriptor-based frontend; the fused LightGlue
pipeline retains its existing aggregate logging.

The control's `ransac_support` CSV label is coarse: some failed estimator/pose
recovery paths return before the RANSAC count is populated. It must not be read
as proof that RANSAC itself found zero inliers. The candidate labels these cases
`estimation_or_pose_recovery`. Likewise primary untrusted-focal F support defers
strict filtering, so its RANSAC and filtered count columns can be identical;
the tracks checkpoint provides the actual later active graph.

An auxiliary OpenCV 4.11 SIFT/RootSIFT check on DSC04890–DSC04900 obtains only
39 mutual ratio matches and 14 fundamental-matrix inliers at 4 px. Its extraction
and distance conventions differ, so it is not a controlled replacement benchmark.
It supports investigating texture/viewpoint ambiguity rather than attributing
every missing correspondence to a CUDA implementation defect.

Candidate validation includes full cold reconstruction, exact gained/lost image
sets, reference pose errors and an identical-observation pixel comparison with
fresh points fitted without the test camera. Passing existing unit tests or
increasing provisional pair counts alone is insufficient to adopt the change.
Reference-selected target pixels may overlap pose-estimation measurements;
the pixel comparison guarantees exclusion of test cameras from fresh depth
fits, not that every tested pixel was unseen during pose estimation.

## State after calibration-order rollback

The only retained production changes are the optional diagnostic directory in
`FrontEndOptions`, its CLI flag, and stage CSV writers/calls. Comparing restored
`frontend.cpp` with the archived cold-control source yields exactly one further
line: the clarified zero-count failure label. Selection, calibration, matching
and registration behavior is restored to the control. Existing unrelated working
tree changes are preserved.

The restored CLI is rebuilt. `photara.sfm.two_view` and `photara.sfm.mapping`
pass after rollback (2/2). No full reconstruction is repeated after this
diagnostics-only rollback. `artifacts/sfm-frontend-calibrated-20261019/validation.json`
verifies both archived executable/source snapshot hashes, successful fresh-run
processes, absence of cache hits, and source equivalence after rollback.

Replay archived binaries with the same arguments into a **new** output/cache
directory; do not rerun the artifact launchers over their existing result folders.
The two source snapshots distinguish the control and rejected candidate, so the
current reverted source must not be described as the tested candidate.

## Correspondence attribution: omitted revisits

The next investigation reuses the frozen cold feature, raw-match and tracks
checkpoints read-only. It does not rerun the previous missing-group experiments
or rewrite checkpoint stages. `artifacts/sfm-correspondence-audit-20261019/`
contains the reader, per-pair CSVs for both scenes, exact-byte diagnostic
rematches and independent reference-only evaluations.

For each missing image, the audit joins primary proposals, rescue attempts and
the actual final pre-mapping graph. Reference tracks nominate diagnostic oracle
pairs only; neither their identities nor their geometry enter reconstruction.
The nearby-detection counts use a 2 px nearest-keypoint radius and do not imply
descriptor identity. Epipolar agreement is also a diagnostic rather than proof
of a correct match.

Several useful London revisits never reached either matching pass. Matching the
saved original native byte descriptors at the original angular 0.8 mutual-ratio
policy produces the following results. Torch float32 dot products use exact
integer-valued byte sums with TF32 disabled, followed by the native double-acos /
float-rounding acceptance rule. Two control pairs reproduce the saved native
CUDA match sets exactly (560/560 and 38/38); this parity check covers those two
pairs, not every possible tie case.

| Images | Primary/rescue attempted | Native-byte matches | OpenCV F inliers / 4 px | Reference epipolar agreement / 2 px |
| --- | --- | ---: | ---: | ---: |
| DSC04710–DSC04920 | neither | 490 | 422 | 330 |
| DSC04705–DSC04920 | neither | 588 | 517 | 447 |
| DSC03720–DSC04990 | neither | 173 | 153 | 121 |
| DSC03715–DSC04990 | neither | 211 | 182 | 147 |
| DSC04555–DSC05050 | neither | 198 | 166 | 163 |
| DSC04560–DSC05050 | neither | 308 | 223 | 211 |

This isolates an image-pair nomination failure for these examples: useful
descriptor correspondences already exist, but the selected candidate lists
omit them. It does not establish that every missing image has the same cause,
or that inserting oracle-selected pairs would constitute a deployable algorithm.
The OpenCV estimator here is an attribution check, not the production verifier.

### Vocabulary training cap and rejected global replacement

The established vocabulary trainer concatenates sampled image descriptors in
image order and stops after 200000 rows. In the actual cold inputs this uses
exactly the first 100 images in each scene, ending at London DSC00530 and
Alameda DSC06535. Later rooms do not contribute training descriptors. A
deterministic systematic sample across the full collection retains the same
200000-row budget while covering all 1874/1734 images (London 61–107 rows per
image; Alameda 115–116). This is a concrete training-distribution bias, but its
effect must still be measured through the entire pipeline.

The first candidate replaces primary/rescue vocabulary training with this
whole-scene sample. It runs London completely from original images in another
empty cache. Image inventories match, independently extracted feature payloads
are byte-identical to the control, both processes exit zero and there are no
checkpoint hits. Its source snapshot and executable remain frozen under
`artifacts/sfm-frontend-vocabulary-20261019/`.

**Reject the global replacement.** It registers 1846 versus 1857: 8 gained,
19 lost. On 2886 identical held-out-depth target pixels in 1838 common cameras,
control mean/P95 are 1.240664/3.320683 px and candidate mean/P95 are
1.769049/3.582546 px. These observations differ from the earlier calibration
candidate's evaluation cohort; means must only be compared within each paired
evaluation. Candidate center/rotation P95 are 0.148861 steps / 0.169204 degrees.

Some recovered cameras have useful precision: DSC04990 gives mean 1.541 px on
135 observations versus reference 1.320 px, DSC04985 1.405 versus 1.061 px,
DSC04995 1.563 versus 1.461 px, and DSC05000 1.654 versus 1.522 px. Others fail:
DSC04945 gives 27.333 versus 0.978 px and DSC04950 41.479 versus 1.088 px.
These are cross-view consistency checks with target-excluded fresh depths;
they do not isolate all error to the test camera when fit anchors are also
imperfect. Neither eight registration flags nor six apparently useful recovered
cameras offset the lost cameras and degraded shared-model precision.

The second candidate preserves the established primary and rescue candidate
selection, then spends only remaining weak-view pair budget on a supplementary
whole-scene vocabulary. No existing candidate is removed, the per-view pair cap
is unchanged, and geometry/resection acceptance thresholds remain unchanged.
The additional vocabulary does incur another training pass of at most 200000
descriptors; it is not a zero-cost change. Its results and
final decision are recorded below after complete evaluation.

### Supplemental retrieval and final independent admission

The supplemental candidate preserves the established 64815 primary London
proposals. Independently extracted feature payloads and raw primary matches are
byte-identical to the cold control; primary diagnostics also agree after
normalizing the clarified failure label. All 38141 original rescue candidates
remain, with identical raw match counts; 1687 additional candidates use leftover
per-view budget. In particular DSC03720–DSC04990 now receives 240 rescue matches
and 172 provisional geometric inliers, using the existing rescue policy.

This complete cold reconstruction registers 1860: it gains DSC04990, DSC05050
and DSC06335, losing no control registrations. **Do not count all three as a
quality gain.** Target-excluded reference-observation evaluation gives DSC04990
1.478 px versus reference 1.320 px (135 observations), DSC06335 1.535 versus
1.515 px (38), but DSC05050 **1144.445 versus 1.445 px** (121).

The native-only depth audit does not use reference observations or poses.
DSC05050 has 466 fitted native observations, 437 with only one other registered
anchor and 29 with two; none have three. Those 29 anchor-only depths even predict
the target at mean 1.407 px after anchor fit/parallax checks, despite its gross
cross-map inconsistency. Its native self-reprojection mean is only 0.265 px.
Thus a small local residual, even with two-view fresh depths, cannot alone
establish a correct placement. Lowering the existing independent-support count
would be particularly unjustified here.

The existing `analyze_alignment_observability` already excludes DSC05050 from
its reliable mask. Production previously left that result diagnostic-only.
`prune_unobservable_registrations` now applies this existing predicate at final
admission, after all recovery, in both fresh and resumed reconstruction paths.
It preserves all poses/intrinsics of retained cameras, removes rejected image
observations, invalidates orphaned triangulated points and rebuilds track
indices. It does nothing without an established component of at least four
cameras. The separate stricter pair-only position audit remains an uncertainty
annotation; its large `unverified` population is **not** indiscriminately pruned.
This does not guarantee that every retained camera is correct.

For London, the production gate is replayed on the frozen **complete** cold
candidate and control checkpoints, rather than repeating extraction/mapping.
The replay mode invokes the same library gate and does not run BA or recovery.
Input hashes and retained-pose equality checks are recorded, and actual ASFM
outputs receive separate reference evaluations. This is a final-stage replay,
not a third London cold run. The executable, source snapshots and audit-tool
manifest are under `artifacts/sfm-frontend-vocabulary-guarded-20261019/`; the
equally screened controls are under
`artifacts/sfm-frontend-vocabulary-control-guarded-20261019/`.

| London, same final admission | Control | Supplement + gate |
| --- | ---: | ---: |
| Registered | 1855 | 1857 |
| Center P95 / reference steps | 0.050299 | 0.026162 |
| Rotation P95 / degrees | 0.133522 | 0.116264 |
| Same 2885 target pixels: mean / px | 1.187158 | 0.978948 |
| Same target pixels: P95 / px | 3.043880 | 2.710076 |
| Same target pixels: RMS / px | 3.652392 | 1.876342 |

The paired cohort has 1854 common cameras and 2880 point-fit jobs; no tested
pixel is filtered by its target residual. Reference mean is 0.889312 px. The
candidate still has a 69.906 px maximum, so aggregate improvement is not proof
that all remaining weak regions are solved.

Relative to the equally screened control, the candidate admits DSC04990,
DSC06335 and DSC08310, and excludes DSC05045. DSC08310 was registered but
unobservable in the original control; it is not a newly registered image from
the raw count perspective. Its independent mean is 1.905 px versus reference
0.924 px (265 observations). The excluded control DSC05045 gives 2.602 versus
0.961 px (86); lack of independent support is not proof that its pose is wrong.
The two newly recovered missing images with close reference precision are
DSC04990 and DSC06335. The previous raw 1857 count and the new screened 1857
count apply different acceptance rules and must not be treated as identical
coverage or advertised as 1860 reliable cameras.

The guard rejects DSC04890/DSC08310 in the cold control and
DSC04890/DSC05045/DSC05050 in the supplemental candidate. Re-analysis of the
remaining London models finds no retained alignment-unobservable camera. The
same guard on Alameda's existing cold control rejects DSC07483, yielding 1710;
this is the comparison for the new complete Alameda run, rather than silently
comparing a screened candidate to 1711 unfiltered flags.

Regression checks cover whole-scene capped vocabulary sampling, rejecting a
floating block despite its local graph cycle, and clearing its orphaned depth
observations. `photara.sfm.mapping` and `photara.sfm.vocabulary` pass (2/2).
Vocabulary format v2 records the sampling policy and the reader accepts v1.

### Alameda complete validation and adoption decision

Alameda runs the supplemental retrieval and final guard together from original
images in a new empty cache. No checkpoint hits occur; reconstruction and the
separate reference evaluation exit zero. Independently extracted features and
raw primary matches are byte-identical to the control, and all 61020 primary
pair diagnostics agree after failure-label normalization. All 35601 old rescue
candidates remain with unchanged raw match counts; 871 candidates are appended.
The run takes 861.466 s inside reconstruction (control 820.200 s); these are
single runs without OS/driver cache isolation.

| Alameda, same final admission | Control | Supplement + gate |
| --- | ---: | ---: |
| Registered | 1710 | 1709 |
| Center P95 / reference steps | 0.033847 | 0.035155 |
| Rotation P95 / degrees | 0.162845 | 0.157352 |
| Same 2702 target pixels: mean / px | 2.221425 | 2.516235 |
| Same target pixels: P95 / px | 4.599257 | 4.611517 |
| Same target pixels: RMS / px | 10.586058 | 15.085404 |

The paired comparison uses 1709 common cameras and 2700 point-fit jobs.
Reference mean is 1.165489 px; no nonfinite target errors occur. No cameras are
gained, and DSC07403 is lost. Its old-model independent error is 23.198 px versus
reference 1.444 px (624 target observations), despite native self-reprojection
mean 0.928 px. This flags cross-map inconsistency, but does not excuse the
candidate's regression on common cameras. The final native admission guard
removes DSC07483, the same camera removed from the control. Neither final
Alameda model retains an alignment-unobservable camera according to the same
existing predicate; remaining high errors show this predicate is not sufficient
to certify all geometry.

**Keep supplemental retrieval opt-in.** Final production code exposes
`FrontEndOptions::whole_scene_rescue` and CLI `--whole-scene-rescue`, default
false. This preserves primary retrieval and the old rescue proposal policy by
default. London demonstrates a useful recovery mechanism, while Alameda does
not support enabling it for all datasets. The rejected global vocabulary
replacement is not part of the default implementation.

The final independent-alignment admission guard remains enabled after recovery
for fresh and resumed reconstruction. This intentionally changes coverage
semantics: the historical 1857/1711 cold registered flags are not all certified
under the stricter rule. Default control results after the isolated gate are
1855/1710; London with supplemental retrieval and the same gate is 1857.
Do not describe the raw 1860 result as reliable coverage or claim an Alameda
registration improvement.

The final CLI adds an opt-in switch around the exact tested supplemental block;
the archived full-run binaries predate the switch and execute that block
unconditionally. Replaying those exact runs uses the archived executables.
Reproducing their algorithm using the final CLI requires `--whole-scene-rescue`.
No full cold run is repeated merely for this default-off switch. The final
source, build/test evidence and equivalence checks are recorded in
`artifacts/sfm-correspondence-audit-20261019/final-validation.json`.

Final build succeeds; `photara.sfm.two_view`, `photara.sfm.mapping` and
`photara.sfm.vocabulary` pass (3/3, `final-tests.log`). CLI help confirms the
opt-in flag. Archived cold executable/source hashes and both scenes' primary
feature/match equivalence checks pass. The final source snapshot and executable
are retained beside the validation manifest; unrelated working-tree edits are
preserved.

## Final cross-view pose refinement

**The first candidate described in this subsection is superseded.** A stronger
synthetic regression found that a biased anchor could induce a small update in
an otherwise accurate camera. Its attempted full cold run was stopped and is
marked rejected under `sfm-quality-crossview-cold-20261019/REJECTED.json`.
The subsequent jackknife variant and its final decision are recorded below;
the first candidate's four-camera gains must not be attributed to that variant.

A subsequent quality pass reuses the frozen complete cold reconstructions:
London's supplemental-retrieval result and Alameda's original-retrieval
control. Both receive the same final alignment-observability admission gate
before comparison. The rejected Alameda supplemental candidate is not the
baseline for the new quality claim. Artifacts are under
`artifacts/sfm-quality-crossview-20261019/`.

`refine_crossview_poses` proposes bounded final pose changes using existing
native correspondences. For each registered observable camera, it samples at
most 512 associated tracks and keeps at most 192 qualifying points. Each point
is freshly triangulated from at least three **other** registered observable
cameras, with all anchor observations within 2 px and at least 2 degrees of
parallax. The inherited landmark position and the target camera's observations
do not enter this point fit. Camera poses are frozen for all proposals in the
pass, so a newly refined camera cannot supply depths for the next proposal.

Track IDs deterministically reserve one third for validation. Fitting requires
at least 40 depths, validation at least 20 and fitting pixels at least four
4x4 image cells. PnP requires 80% fitting consensus at 2 px. Acceptance requires
improvement on fitting cost and independent validation cost/median/P95, no loss
of 2 px validation support, at least 80% validation support, rotation change at
most 2 degrees and center change at most 5% of median target-to-point distance.
These are conservative candidate gates, not a universal accuracy certificate.
Only accepted camera poses change; intrinsics and registration flags stay
fixed. Affected native tracks are then retriangulated. There is no joint BA
that could silently move the other cameras afterwards.

The native validation tracks are excluded from this new PnP fit. They are not
guaranteed to be unseen by earlier joint BA, and their anchor cameras come from
that existing reconstruction. Thus they provide a useful check against
overfitting the new pose update, not fully independent capture evidence.

London tests 302 candidates after support/old-error gates and accepts one;
Alameda tests 423 and accepts three. Registration sets remain 1857 and 1710.
Separate reference-observation evaluation uses fresh target-excluded depths
and identical fit/test pixels for the compared models, with no target-error
filtering. This is the evaluation cohort for this pass, distinct from earlier
tables; absolute means from different cohorts must not be compared.

| Changed camera | Target observations | Before mean / px | After mean / px | Reference mean / px | Before/after P95 / px |
| --- | ---: | ---: | ---: | ---: | --- |
| London DSC01780 | 142 | 1.092887 | 0.824949 | 0.859327 | 3.788514 / 2.061831 |
| Alameda DSC07249 | 381 | 4.671978 | 2.313815 | 1.6383 | 16.3115 / 5.4531 |
| Alameda DSC07530 | 804 | 8.1388 | 8.2563 | 1.3380 | 41.9841 / 41.8620 |
| Alameda DSC07929 | 824 | 1.2133 | 1.2069 | 1.1846 | 3.2682 / 3.1558 |

The Alameda three-camera aggregate (2009 pixels) improves from mean
4.640790 to 4.237975 px, P95 22.623655 to 15.078848 px. DSC07530's mean
**regresses** despite passing native held-out validation; this limitation must
not be hidden behind the aggregate. Its large tail also involves inconsistent
fit anchors: independent fits passing an anchor-max-error diagnostic give a
different subset and must not replace the unfiltered paired result above.

The shared-camera samples show only a small whole-model effect, as expected
when four of 3567 registered cameras move. London (2882 pixels) changes mean
0.988093 to 0.988044 px; Alameda (2682 pixels) changes 2.042033 to 2.041560 px.
The existing 69.906/238.215 px maxima remain essentially unchanged. This is a
localized quality improvement, not a fix for the remaining global error tails.
No registration increase is claimed.

Reference-aligned pose comparison also improves DSC01780's rotation error
0.5303 to 0.2298 degrees and DSC07249's 1.2494 to 0.3282 degrees. These
evaluations robustly realign each entire model, so tiny apparent changes in
unchanged cameras' aligned errors are gauge-fit changes, not actual pose edits.

The first candidate temporarily integrated this pass after observability
pruning for fresh and resumed reconstruction, with a
`--crossview-refinement=false` switch and checkpoint-key configuration. This
integration and switch were subsequently **reverted**, as described below.
Supplemental whole-scene retrieval remains default-off.
Synthetic regression checks cover a displaced camera with deliberately stale
landmark positions, preservation of accurate anchor poses/intrinsics/flags,
and rejection of depths with fewer than three other cameras.

The completed stored-result experiments are final-stage replays, not new cold
runs. The attempted original-image Alameda run was aborted after the stronger
regression exposed anchor contamination; it provides no complete-run result.

### Anchor-depth stability correction

The regression scene has four accurate anchors and one displaced target, with
deliberately stale inherited landmark positions. The initial pass repaired the
target but also moved an accurate anchor: target exclusion alone does not stop
the remaining anchor set from containing a biased pose. The stronger regression
therefore required accurate anchors to remain unchanged. It failed on the first
candidate, and passed after the following correction.

For each fresh point, the new pass removes every anchor in turn and
retriangulates from the remaining anchors (at least two, using the same
reprojection/parallax checks). It predicts the pixel in the target camera from
each alternative depth. If any prediction differs by more than 0.5 px from the
full-anchor prediction, the point is excluded from both fitting and validation.
The target measured pixel does not enter this depth-stability gate. This is a
conservative bound on anchor-choice sensitivity, not a calibrated statistical
uncertainty estimate; earlier BA can still correlate anchor poses.

The new variant is frozen under
`artifacts/sfm-quality-crossview-jackknife-20261019/`, with separate binary,
source hashes, outputs and reference-only evaluations. It tests 40 London
cameras and accepts DSC09610; tests 102 Alameda cameras and accepts DSC07421,
DSC07553 and DSC07832. Its acceptance set is entirely different from the first
candidate. London/Alameda registration counts remain 1857/1710.

London DSC09610, on the same 262 target observations, improves mean
1.078968 to 1.022897 px and P95 2.306187 to 2.074417 px; reference mean is
0.966996 px. Maximum error increases from 4.381 to 4.663 px. The shared-model
sample mean changes only 0.988093 to 0.988092 px. The pass costs 92.282 s on
London and 92.973 s on Alameda, versus 7.478 s for the first London candidate.
Its precision and runtime tradeoff must be assessed on this actual variant,
not the superseded candidate's larger localized improvements.

### Final cross-view decision: do not adopt either candidate

The complete paired evaluation of the jackknife variant finds **no overall
quality gain**. London has a small localized improvement, but all three changed
Alameda cameras have worse mean errors. Same-pixel results are:

| Alameda jackknife target | Pixels | Before mean / px | After mean / px | Before P95 / px | After P95 / px |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSC07421 | 362 | 2.449344 | 2.623366 | 5.678708 | 6.293025 |
| DSC07553 | 1607 | 1.539806 | 1.552805 | 4.492239 | 4.377163 |
| DSC07832 | 1242 | 0.918273 | 0.939426 | 2.517263 | 2.560341 |
| All changed | 3211 | 1.401939 | 1.436245 | 4.267900 | 4.278352 |

The reference mean for the aggregate is 1.245576 px. On the same 2682 shared
model pixels, mean changes 2.042033 to 2.042188 px; this is essentially flat
with a small regression, not a demonstrated improvement. Registration sets are
unchanged in both datasets. Native fitting/validation residuals fall, yet the
reference-observation comparison worsens. A target-excluded fit, withheld
tracks and leave-one-anchor-out depth stability are still not sufficient
accuracy evidence when the native association/depth system itself is biased.

Both pose-refinement candidates are rejected. Their binaries, candidate code,
per-pixel results, manifests, failed/passing regression logs and aborted-run
record remain in artifacts. No additional full cold run is started for the
jackknife variant after this negative result. No registration or geometric
threshold is relaxed.

All five modified C++ source/header/test/tool files are restored to the
pre-turn algorithm, with normalized-text equality against the previous source
snapshot verified in `sfm-quality-crossview-jackknife-20261019/rollback.json`.
The new refinement API, CLI switch, automatic calls and prototype regression
test are not retained in the working product code. The previously validated
observability admission guard and opt-in whole-scene rescue remain exactly as
before this turn. Rebuild/check results after restoration are recorded below.

Restored CLI and recovery tool rebuild successfully. `photara.sfm.two_view`,
`photara.sfm.mapping` and `photara.sfm.vocabulary` pass (3/3, 10.06 s) in
`artifacts/sfm-quality-crossview-20261019/rollback-tests.log`. Final executable
hash, source-restoration checks and all four completed replay/evaluation exits
are recorded in
`artifacts/sfm-quality-crossview-jackknife-20261019/final-validation.json`.
No rejected refinement code remains in the formal product or test files.

This round establishes a limitation, not another quality claim. The next
recovery work needs genuinely new cross-anchor correspondences and validation
of their association/anchor independence, rather than more optimization of the
same internally consistent tracks. Reference pixels must remain evaluation-only.

## Detector-level sampling and budget attribution (2026-10-19)

The next investigation starts in candidate nomination, not camera refinement.
Artifacts: `artifacts/sfm-retrieval-path-20261019/`.

Read-only two-hop expansion of primary accepted geometry finds no unattempted
candidate for most missing cameras. In particular it cannot bridge London
DSC04920's disconnected neighborhood. Reusing existing graph edges is not a
substitute for finding new cross-window correspondences.

CUDA SIFT currently writes response = 0 for every exported keypoint. Native
spatial retrieval sampling ranks response times scale; all these scores tie,
so each spatial cell takes a prefix in detector order. This excludes later
scale/octave levels. The eight saved native rematches were examined without
changing their descriptors or correspondences. For DSC04705–DSC04920, only
34/588 correspondences have either endpoint sampled; systematic sampling
across the same cell lists raises this to 111/588. Both endpoints sampled
changes only 3 to 4: better endpoint coverage alone is not evidence of good
retrieval or reconstruction.

Native fixed-feature/fixed-vocabulary retrieval replays completed for both
scenes (all four exits 0). Original top-50 retrieval is a subset of the saved
primary proposal set for both datasets, establishing nomination parity.
Changing only histogram sampling retrieves DSC04705–DSC04920, but not the
other five known diagnostic missing edges. It proposes 814 London and 1056
Alameda previously unattempted pairs touching the baseline missing cameras;
these are candidates, not verified geometry.

A second causal obstruction is the rescue budget: DSC04920 already uses
88/96 pairs, while DSC04705 ranks 31st among its novel histogram proposals.
Simply enabling uniform sampling would still miss the useful edge. No full
cold run was launched for that sampling-only variant.

A reference-free coarse nomination check matches the full DSC04920 native
byte descriptors against 2000 systematic descriptors per proposed anchor.
All 46 novel top-50 histogram proposals are checked. DSC04705 yields 81 mutual-ratio matches and 51 OpenCV fundamental
inliers at 2 px; next-best geometric support is 11. This attribution uses no
reference model. OpenCV F here only audits the ordering; it does not replace
native geometry or admit a camera. The production candidate ranks by native
coarse match count, then runs unchanged full matching and geometric checks.

The candidate retains the original primary/rescue proposals and spends only
remaining per-weak-view budgets. Systematic sampling changes only the
supplemental histogram; vocabulary training and all original features stay
unchanged. Coarse subsample indices never enter tracks or pose estimation.
The optional `--balanced-rescue` switch stays off by default pending complete
reconstruction and paired quality validation. No pose or geometry threshold
is relaxed. A regression check verifies full detector-order coverage and
sparse-cell quotas, including empty and uncapped sampling.

### Complete London cold result and rejection of wholesale replacement

The real original-image, empty-cache London run completes with exit 0 and no
checkpoint hits. Feature and primary-match payload SHA-256 values equal the
cold baseline; primary diagnostics match numerically and in selection after
normalizing the previously corrected zero-inlier failure label. Every one of
the old 38141 rescue proposals remains with identical raw match count. There
are 1714 additional full-match proposals. Deriving the primary weak mask from
the diagnostics reproduces 479 views; maximum rescue degree is 96 and no view
exceeds the existing budget.

The native coarse pass checks 12261 novel proposals. The DSC04705–DSC04920
proposal has 88 coarse matches on the augmented feature sets and is selected.
Formal rescue at its unchanged 0.85 ratio obtains 826 matches and 650 native
geometry inliers. The earlier 588 rematches and 81 coarse matches used primary
features and ratio 0.8; these are different stages, not repeated-run drift.

Full global mapping plus existing final support screening registers 1869,
versus 1855 on the equally screened cold control, with 14 gained and none lost.
This number is **not adopted as reliable coverage**. Same-pixel evaluation
finds:

| London gained camera | Pixels | Candidate mean / px | Reference mean / px |
| --- | ---: | ---: | ---: |
| DSC03210 | 50 | 1.255 | 1.034 |
| DSC04890 | 66 | 1.698 | 1.510 |
| DSC04895 | 69 | 129.899 | 1.223 |
| DSC04900 | 102 | 196.963 | 1.563 |
| DSC04910 | 23 | 2.522 | 2.161 |
| DSC04915 | 77 | 1.587 | 1.478 |
| DSC04920 | 96 | 1.512 | 1.317 |
| DSC04985 | 49 | 1.932 | 1.061 |
| DSC04990 | 135 | 1.464 | 1.320 |
| DSC04995 | 97 | 1.675 | 1.461 |
| DSC05000 | 121 | 1.975 | 1.522 |
| DSC05005 | 102 | 2.233 | 1.398 |
| DSC06335 | 38 | 7.312 | 1.514 |
| DSC08310 | 265 | 1.896 | 0.924 |

On 2902 shared-camera pixels against the screened cold control, mean improves
1.194909 to 1.018206 px and P95 2.967101 to 2.762737 px. Against the stronger
previous whole-scene supplemental result, however, the same 2882 pixels change
mean 0.988093 to 1.015138 px and P95 2.770003 to 2.776711 px. That prior result
also has a much better DSC06335 pose. The global candidate does not replace it.

A native, reference-free diagnostic excludes every gained camera from depth
anchors. DSC04895 and DSC04900 then have zero old-map anchors on all their
tracks (84 and 60 observations). DSC06335 has only one qualified three-anchor
track. These newly connected cameras must not certify each other's placement.
This is why the next step reuses a retained model instead of promoting the
whole new global solution.

### Reuse the previous model: one validated frozen-map extension

`photara_sfm_extend` reads the retained full-feature ASFM and the completed new
correspondence checkpoint. It checks the full image set, extraction grid and
feature-index prefix exactly. It imports only active boundary correspondences;
new candidate poses, intrinsics, landmarks and registration flags are ignored.
It calls the existing ordinary `recover_stable_resections(scene, stable)`:
three old anchors per fresh depth, a 30-inlier fit budget, at least 12 held-out
inliers, and the existing spatial/rotation/direction checks. The stable mask
is frozen for the entire pass. No validation threshold changes.

Input baseline is the previously guarded London whole-scene result with 1857
registered cameras. This is a retained-result extension, **not another full
cold run**. The preceding new correspondences did come from a complete cold
original-image run. Both input hashes and the extension executable are archived
in `artifacts/sfm-balanced-frozen-extension-20261019/manifest.json`.

Only DSC04920 is accepted: 90 independent proposals, 35/60 fitting inliers,
23/30 held-out inliers distributed over six image cells. DSC04895 and DSC04900
fail independent-depth support; the good old DSC06335 remains frozen. Other
nominees do not receive relaxed admission. Result: **1857 to 1858**, no old
camera lost. This is a one-camera increase on the same retained baseline, not
a claim to exceed the separate historical fixed-track coverage count.

The exported ASFM independently verifies every camera record and all 1857 old
registered image records byte-for-byte against the baseline, including poses,
calibration, keypoints and flags. Track coordinates/observations may be updated
by the existing target-excluded triangulation/attachment routine; this is not
a claim that every landmark byte stays fixed. New poses are marked position
unverified in the product metadata rather than inheriting a blanket guarantee.

On 96 fixed reference observations, the added camera has mean **1.544834 px**,
P95 **3.705651 px**; reference mean is **1.317447 px**, P95 **3.055478 px**.
Reference observations are evaluation-only and target pixels do not determine
which depth is fitted. On 2882 shared-camera evaluation pixels, baseline and
extension means are 0.9880934016 and 0.9880934024 px. The sub-micro-pixel numeric
difference comes from decimal pose export/optimization numerics; actual old
poses and camera records are byte-identical. Existing large-error tails are
preserved, not claimed repaired.

Reviewable model:
`artifacts/sfm-balanced-frozen-extension-20261019/london/model.asfm`.
The supplemental global switch remains default-off. The retained product
change is optional descriptor-supported retrieval plus a native frozen-map
extension tool; no failed cross-view refinement code is reintroduced.
`photara.sfm.two_view`, `.mapping`, `.vocabulary`, `.submap_recovery` pass 4/4
(11.25 s). Builds, source/executable hashes, per-pixel results and frozen-record
proofs are retained with the runs.

### Alameda: incomplete run, no quality conclusion

Alameda starts from its 1734 original images and an empty cache. The original
61020 first-stage pair proposals are retained; 988 supplemental full-match
pairs are nominated after the old 35601 rescue proposals. The run fails at
36588/36589 rescue tasks with `bad allocation` before frontend finalization.
There is no completed reconstruction or coverage result for this variant.
Feature and primary-match checkpoints, image inventory and diagnostics are
retained; a later continuation can reuse them and must be labelled a resumed
run rather than a second cold run.

The user reports Blender was running concurrently. Resource contention is a
plausible explanation; the log alone does not establish the root cause or
implicate algorithmic quality. No threshold is relaxed and this failed run
provides no evidence for or against Alameda reconstruction accuracy. No
background retry is left running.
