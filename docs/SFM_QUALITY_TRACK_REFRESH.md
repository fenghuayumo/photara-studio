# Native track refresh quality improvement — 2026-10-19

This round starts from the previously retained `sfm-continued-20261019/{scene}-refined/model.asfm`, not the earlier cold/global baselines. Both datasets retain exactly the same registration flags and calibration records. Reference data is used only for evaluation.

## Retained method

The saved ASFM contains triangulated tracks, and its BA uses only their current inlier observations. Refining those alone leaves previously rejected or untriangulated native correspondences outside the solve. Restore associations from the balanced native reconstruction checkpoint, discard imported point positions and inlier labels, triangulate against the retained camera poses at 2 px / 1 degree, then jointly optimize poses and points with fixed calibration (Huber 1 px, up to 100 iterations). All retained cameras stay frozen during association import; the subsequent BA can move them.

```text
photara_sfm_extend previous-refined.asfm reconstruction-HEX.bin refreshed.asfm --fresh-tracks --merge-features
photara_sfm_refine refreshed.asfm final.asfm
```

The commands above describe the original offline experiment. The method is now integrated into the normal alignment path; see [Production integration](#production-integration). No downstream dense reconstruction or rendering improvement is claimed without testing that pipeline.

## Same-observation comparison

Every target camera is excluded from the fresh depth fit. The original deterministic cohort is reused without filtering by target residual. Verification reproduces every previous pixel error exactly before comparing the candidate.

| Dataset / metric (px) | Previous retained | Refreshed + refined |
| --- | ---: | ---: |
| London mean | 0.952902 | 0.943832 |
| London median | 0.666128 | 0.669099 |
| London rms | 1.544184 | 1.393062 |
| London p95 | 2.593031 | 2.579085 |
| London max | 39.032613 | 20.880447 |
| Alameda mean | 1.547204 | 1.515471 |
| Alameda median | 0.923403 | 0.923634 |
| Alameda rms | 4.541831 | 4.140370 |
| Alameda p95 | 4.095102 | 4.045134 |
| Alameda max | 125.473919 | 109.382963 |

The cohorts contain 2,868 London pixels and 2,681 Alameda pixels. RMS improves by 9.8% and 8.8%, respectively. Mean and P95 improve in both scenes. Medians increase slightly; this is not a uniform per-camera or per-pixel improvement. Alameda still has a 109 px tail.

## Additional non-overlapping pixels

A second deterministic seed (20261020) selects observations after the candidate solves. Every point/target pair in the original cohort is explicitly excluded. These pixels are never solver inputs. The models are not retuned after this check.

| Dataset | Pixels | Mean before → after | RMS before → after | P95 before → after |
| --- | ---: | ---: | ---: | ---: |
| London | 2321 | 0.902823 → 0.899753 | 1.251977 → 1.245396 | 2.643038 → 2.637782 |
| Alameda | 2351 | 1.364799 → 1.336078 | 2.931569 → 2.653172 | 3.643860 → 3.651390 |

The additional London cohort shows a smaller improvement than the original cohort: the original RMS gain is concentrated in a weak region, not a scene-wide 9.8% reduction. The additional Alameda cohort improves mean/RMS but its P95 increases from 3.643860 to 3.651390 px; this local regression is retained in the report. References are comparison reconstructions, not absolute ground truth.

## Native diagnosis and rejected alternatives

Before refresh, London DSC07280 had 290 inlier observations but only 33 tracks involving at least three views; after refresh these counts are 295 and 40. Alameda DSC07265 grows from 17 to 27 inlier observations, but still has only three multi-view tracks. Low native residual therefore does not establish an accurate pose. These counts motivate restoring constraints, rather than merely increasing solver work.

- Precision-only BA (500 PCG iterations, tighter tolerances, 150 outer iterations) takes 153 s / 222 s and gives no material pixel improvement. Rejected.
- Huber 0.5 px slightly improves London RMS but worsens its median/P95; Alameda RMS increases to 4.656604 px. Rejected.
- Native-support-selected resection with three-anchor depths, and the separately tested two-anchor path, fail to fix the main weak views. The latter adds three London registrations but slightly worsens shared-camera mean/RMS. Both experimental repairs are archived and removed from the tool source.
- Retriangulating only saved tracks before BA improves London RMS to 1.405030 and Alameda to 4.501279 px. This remains an optional `--retriangulate` offline-tool mode; full association refresh gives the larger retained gain.

The diagnostic `--precision` and `--huber=` flags are opt-in; default refinement settings are unchanged. The accepted outputs use neither flag.

## Validation and artifacts

All six comparison solves (precision, robust, saved-track refresh), both full association refreshes, and both final refinements completed successfully. An early Alameda full-refresh refinement launch encountered a missing input before solving because the preceding refresh had not yet saved; it was rerun after that stage completed, with the successful command recorded in `fresh-refined-manifest.json`.

Five BA/SfM tests pass; invalid/nonfinite Huber arguments are rejected. Saved output checks confirm identical camera calibration bytes and registration flags (London 1859; Alameda 1720). Complete pixel CSVs, independent cohort checks, native support audits, source snapshots, binary/input hashes, and run logs are in `artifacts/sfm-quality-round2-20261019/`.

Retained models:

- `artifacts/sfm-quality-round2-20261019/london-fresh-refined/model.asfm`
- `artifacts/sfm-quality-round2-20261019/alameda-fresh-refined/model.asfm`


## Production integration

The native quality pass is now part of `sfm::reconstruct_from_frontend`, shared by the normal CLI/Studio alignment path (`sfm::reconstruct`) and frontend/mapping API callers. Both global and hierarchical mapping finish with the same stage, after recovery and registration screening, before final position audit, diagnostics, summary, coloring and saving. Existing live associations are refreshed directly; no external executable or reference input is needed.

`refine_alignment_geometry` clears stale point geometry/inlier labels, re-triangulates all live native associations at 2 px / 1 degree, then performs fixed-calibration joint BA at Huber 1 px for up to 100 iterations. It preserves registrations and keeps duplicate-image poses fixed. Unusable/nonfinite/cost-increasing solves, or loss of the previous minimum support (up to 30 valid observations per camera), restore the prior poses, tracks, colors and track index. It does not certify every individual pose.

The mapping cache version and source fingerprint change. New cache entries contain the finished quality result; cache hits reuse this geometry instead of rerunning recovery/refinement. Older entries cannot bypass the new stage. Existing source-fingerprint policy can also invalidate earlier frontend caches. Saved projects are updated when the user runs alignment again.

### Actual product-entry validation

London and Alameda were rerun through the ordinary `reconstruct_from_frontend` entry point from existing native balanced frontend tracks. These are fresh mapping runs, not fresh feature-extraction benchmarks or evidence that balanced retrieval is now the default. A diagnostic observer saved the exact scene immediately before the new stage; it did not change any solver options. The final stage was accepted in both runs, with calibration bytes and registration flags unchanged. Evaluation uses the same preselected pixels on each side and excludes every test camera from its fresh depth fit.

| Dataset | Cameras | Pixels | Mean px before → after | RMS px before → after | P95 px before → after | Max px before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| London | 1869 | 2902 | 1.160300 → 1.131944 | 5.194169 → 5.101295 | 2.855846 → 2.771877 | 205.378827 → 210.824298 |
| Alameda | 1720 | 2697 | 1.555922 → 1.487565 | 3.123455 → 2.696784 | 4.237083 → 4.219761 | 55.309604 → 54.082183 |

These cohorts differ from the earlier offline-refinement cohorts. In particular, this London mapping has 1869 registrations versus the retained 1859-camera experiment; their absolute pixel statistics must not be compared as if they were the same observations. The new stage improves aggregate mean/RMS/P95 in the reported product runs, but local tails may regress (including London's maximum). No uniform improvement or downstream dense/rendering result is claimed.

Eight distinct BA/SfM/Studio alignment tests pass, including native-association recovery, fixed calibration, duplicate-pose preservation, rejected-solve rollback and the global product callback order. The strengthened rollback test was rebuilt and rerun separately after the seven-test BA/SfM suite. A 12-photo real CLI smoke test registers six cameras, accepts the quality stage on the first run, and hits the finished reconstruction cache on the second run; calibration and camera-pose bytes are identical, and the second run performs no new quality solve.

`photara.exe` and `photara_studio.exe` have both been rebuilt in `build/photara/Release/`. Studio resolves the CLI beside its own executable, so that build's Align command uses the integrated stage. Complete commands, hashes, pre/post models, pixel CSVs, cache checks and test logs are in `artifacts/sfm-production-quality-20261019/`.

## Missing-camera experiment (2026-10-20; disabled in production)

The experimental reconstruction entry point retried missing cameras after a successful native geometry refinement. This additional pass has now been removed from the default CLI/Studio path because its coverage gain did not justify its runtime and full-run precision risk. Earlier production recovery keeps its existing 16-anchor search. The offline final pass first tries those same anchors, then extends the failed search to 24 using verified overlap neighbors of the original anchors (hard cap 32). Multiple distinct anchor witnesses rank above a single large pair; unregistered, zero-baseline, weak and coincident-center proposals do not supply additional anchors. Existing local feature/match caches are reused between attempts.

Search proposals do not certify a pose. Depths are rebuilt from registered anchors without using the missing camera. Both fitting and independent held-out observations must reach at least 50% consensus, in addition to the existing minimum counts, independent-anchor, spatial, direction and conditioning checks. Accepted observations append new tracks and keypoints; existing registered camera poses, calibration, keypoint prefixes and old landmark geometry/observations stay fixed. No second joint BA moves the old map after admission. The experiment used cache version v40. Production now uses v41 to exclude finished experimental caches and ends with the previously accepted native track refresh and position audit.

### Frozen-map ablation

Both datasets start from the previous production result, using only native image data and pair/track associations. Reference models enter evaluation only.

| Dataset | Previous registrations | Original 16-anchor retry | Expanded retry | Retained conservative pass |
| --- | ---: | --- | --- | --- |
| London | 1869 / 1874 | No additions | DSC05050 | DSC05050: 1870 / 1874 |
| Alameda | 1720 / 1734 | DSC06475, DSC07484 | Same two | DSC06475: 1721 / 1734 |

DSC07484 is deliberately rejected: its fit consensus is 33/103 and held-out consensus 14/51. In independent reference-pixel evaluation, the permissive candidate has mean 8.95 px and P95 22.64 px on 200 sampled tracks. More registrations alone would conceal this failure. London DSC05050 passes with 53/84 fit and 23/41 held-out inliers; Alameda DSC06475 passes with 35/64 and 16/32.

Evaluation triangulates fresh points from at least three originally registered cameras and excludes every gained camera from depth fitting. Observations are selected deterministically before computing target errors; there is no target-residual filtering. London uses the relocated reference `D:/Models/london/sparse_rr/0`; Alameda uses `D:/Models/alameda/sparse/0`.

| Accepted camera | Evaluation pixels | Mean px | P95 px | Reference mean / P95 px |
| --- | ---: | ---: | ---: | ---: |
| London DSC05050 | All 121 eligible tracks | 2.33 | 5.68 | 1.44 / 3.66 |
| Alameda DSC06475 | 200 sampled tracks | 1.67 | 3.40 | 1.20 / 2.89 |
| Alameda DSC06475 | Other 244 tracks | 1.79 | 4.00 | 1.37 / 3.86 |

The late pass's frozen-map outputs preserve old camera calibration/pose bytes, keypoint prefixes and landmark geometry/observations in both datasets. These are modest coverage improvements, not proof of reference-level pose accuracy or improved dense reconstruction. The other four London and thirteen Alameda cameras remain missing, including Alameda DSC08181/DSC08182. The stricter graph-only position audit can still mark a newly registered camera unverified; registration is not a position-accuracy certificate.

The retained pass takes about 37 s for London and 153 s for Alameda on the frozen checkpoints, excluding checkpoint load, coloring and export. This extra cost is paid on a new reconstruction, not a finished-cache hit. The 12-photo CLI smoke test succeeds and reuses byte-identical calibration/camera geometry from cache on its second run (22.54 s first run, 0.56 s cached). Eight BA/SfM/Studio tests pass, covering anchor selection, consensus rejection and existing alignment safeguards. Both CLI and Studio binaries were rebuilt; Python binding fields are updated in source, but the optional Python module was not built.

### Full product-entry replay and limits

Fresh mapping from the same native frontend checkpoints through `reconstruct_from_frontend` confirms 1870 London and 1721 Alameda registrations, with no losses versus the historical product result. Both gained cameras pass the existing alignment-observability audit. Neither passes the stricter position audit; their position accuracy remains qualified. The final pass takes 38.94 / 149.65 s; whole mapping/export runs take 193.75 / 409.40 s. These timings exclude fresh frontend extraction. Full-run added-camera evaluation covers all 121 / 444 eligible observations: London mean/P95 2.332/5.681 px; Alameda 1.735/3.826 px.

The full rerun does **not** establish uniformly better reconstruction precision than the historical best. On identical shared-camera observations:

| Dataset | Pixels | Historical → current mean px | RMS px | P95 px |
| --- | ---: | ---: | ---: | ---: |
| London | 2902 | 1.131944 → 1.154676 | 5.101295 → 5.847612 | 2.771877 → 2.804763 |
| Alameda | 2697 | 1.487565 → 1.559670 | 2.696784 → 3.794633 | 4.219761 → 4.226405 |

Both new runs already differ from the historical result in every common registered pose before native quality refinement or this new recovery stage; calibration also differs slightly (recorded in `formal/pre-quality-run-comparison.json`). This establishes an earlier run difference, without establishing its cause. Within the current London run, native quality refinement plus the frozen-camera recovery improves mean/RMS/P95 from 1.183602/5.915430/2.863381 to 1.154676/5.847612/2.804763 px, while maximum error grows from 252.14 to 256.12 px. In the current Alameda run, mean/RMS/P95 improve from 1.625327/4.151840/4.253034 to 1.559670/3.794633/4.226405 px. The new recovery itself preserves old geometry in the controlled ablation, but the complete remapping path still has weak-camera tail variation. Accordingly, this change is a verified coverage improvement with conservative admission, not a claim that full reruns surpass the previous best precision. Existing validated reconstruction artifacts are retained.

Complete ablation commands, hashes, frozen-map proofs, pixel observations, product-entry replays and test logs are under `artifacts/sfm-support-recovery-20261020/`. The offline `photara_sfm_realign --post-refinement` mode retains the former experimental product helper, now disabled in the default path; `--audit-only` reports registration support without modifying the map.

## Registered weak-camera repair experiments (2026-10-20)

Production no longer calls `recover_post_refinement_views`. CLI and Studio were rebuilt with this rollback. A fresh 12-photo CLI run confirms that native alignment quality still runs, the additional recovery stage does not run, and the second run uses the finished reconstruction cache with identical calibration and camera geometry. This removes the added missing-camera search cost; it does not promise that a fresh nonlinear reconstruction reproduces a historical result bit for bit.

The next experiment repairs registered weak poses instead of adding registrations. The current full-run sample's squared error is dominated by London DSC04895/DSC04900 (about 90% together) and Alameda DSC07265 (about 41%). These contribution estimates each involve just one sampled pixel for those cameras. A denser check is therefore necessary before judging a repair. Native support independently explains the concern: the two London views have no three-view inlier tracks, and Alameda DSC07265 has just three, despite small internal reprojection errors.

Nomination is entirely native: fewer than 30 three-view tracks and fewer than 400 total inlier observations, excluding duplicate-image ties, at most eight candidates. This selects four London and six Alameda cameras from the retained production checkpoints. The experiment re-extracts local features, rebuilds depths from anchors, and uses the same independent fit/holdout certificates with a 50% consensus floor. Calibration and all non-target poses stay fixed. On success, stale target observations are removed from old tracks; points with fewer than two remaining inliers are omitted from export. Surviving old landmark coordinates and keypoint prefixes are preserved. Reference reconstructions are evaluation inputs only.

Two explicitly offline modes retain the admission tradeoff:

- `--weak-views-proposal`: batch proposals satisfying the absolute independent certificate. This corrects London DSC04900/DSC04910/DSC07280 and Alameda DSC07402/DSC07482.
- `--weak-views`: additionally requires at least a 10% and 0.01 px²/observation reduction in held-out clipped squared error (4 px cap), without losing held-out inliers, relative to the original pose. Trials restore better-supported cameras first; failed trials immediately restore their old poses. This keeps London DSC04910 unchanged, but loses the DSC04900 repair because its native certificate depends on the available anchor geometry. It retains DSC07280 and the same two Alameda repairs. This coupling is unresolved; the rule remains off by default. Its subsequent optional product integration is described below.

### Fixed-cohort reconstruction precision

The baseline is the retained 1869-camera London / 1720-camera Alameda production reconstruction, not the worse fresh rerun. Registration counts remain identical in both variants, with no gains or losses. Evaluation uses precisely the previous 2902 / 2697 shared pixels, excludes each target camera from its fresh depth fit, and performs no target-error filtering.

| Dataset / variant | Mean px | RMS px | P95 px | Maximum px |
| --- | ---: | ---: | ---: | ---: |
| London baseline | 1.131944 | 5.101295 | 2.771877 | 210.824298 |
| London proposal | 1.063559 | 4.224278 | 2.756827 | 210.824298 |
| London with relative improvement gate | 1.116816 | 5.031847 | 2.763291 | 210.824298 |
| Alameda baseline | 1.487565 | 2.696784 | 4.219761 | 54.082183 |
| Alameda proposal / relative improvement gate | 1.432252 | 2.333954 | 4.132456 | 30.634472 |

The proposal lowers RMS by about 17.2% / 13.5%, but a stricter admission rule reduces London's RMS benefit to about 1.4%. This is not grounds to silently select only favorable metrics or to integrate the proposal as a default.

### Denser per-camera checks and limitations

For these checks, all nominated cameras are excluded from anchor-depth fitting. Points use at least three originally registered cameras. The first cohort uses all eligible tracks up to 200 deterministic samples per view; target errors do not affect selection.

| Camera | Pixels | Baseline → proposal mean px | Baseline → proposal P95 px |
| --- | ---: | ---: | ---: |
| London DSC04900 | 102 (all eligible) | 189.717 → 1.990 | 292.878 → 5.398 |
| London DSC07280 | 40 (all eligible) | 6.929 → 1.645 | 45.207 → 2.588 |
| London DSC04910 | 26 (all eligible) | 1.859 → 2.037 | 3.676 → 3.874 |
| Alameda DSC07402 | 201 | 5.235 → 3.732 | 10.619 → 10.287 |
| Alameda DSC07482 | 201 | 41.924 → 1.771 | 74.958 → 4.576 |

DSC04910's local regression is retained, and the conservative variant keeps its original pose. A second disjoint cohort for the two retained Alameda repairs confirms the improvement: DSC07402 (201 pixels) mean 5.338 → 3.775, P95 10.940 → 8.932 px; DSC07482 (202 pixels) mean 41.110 → 1.614, P95 76.750 → 3.608 px. No solver options were changed after this second check. London DSC04895 and Alameda DSC07265 remain unrepaired and inaccurate in denser reference checks; their poses are retained rather than forced through weaker gates. Reference models are comparison reconstructions, not absolute ground truth.

Three current mapping/submap/Studio tests pass, including target-list isolation and acceptance/rejection of the relative-improvement gate. The latest tool reproduces the fully evaluated proposal poses and calibration byte for byte. Full evaluation outputs, disjoint pixel cohorts, native-support audits, frozen-geometry proofs, binary/input hashes and logs are under `artifacts/sfm-tail-quality-20261020/`. `proposal/` contains the repeatable proposal, `staged/` the conservative result; earlier trial folders document the anchor-order tradeoff. Timing runs overlapped some evaluation/smoke workloads and are not a clean comparative performance benchmark.

The incremental frozen-map evaluator recomputes only shared jobs containing a changed target or fit camera, reusing previously computed baseline/reference and exactly unchanged candidate values. It requires byte-identical calibration, registration and non-target poses. Its results are checked against the full evaluator on the initial London proposal; final proposal calibration and all poses also match that fully evaluated model. These checks accelerate repeated diagnostics without changing the test cohort or filtering errors. The product flow does not run this evaluator or use reference data.

## Optional production weak-camera refinement

The guarded repair is now implemented by `refine_weak_alignment_views` in the SfM library. `ReconstructionConfig::refine_weak_cameras` enables it after native alignment quality and before final position audit/summary in `reconstruct_from_frontend`, for both global and hierarchical modes. It defaults to false. The offline `--weak-views` tool delegates to this same helper; the unguarded proposal remains an offline experiment.

Enable it with CLI `--refine-weak-cameras`, or Studio's **Refine weak cameras / 精修弱约束相机** checkbox in the alignment settings. Studio forwards it for ordinary and training-triggered alignment. The option is persisted in project settings v14; older projects load with it disabled. The Python `SfmOptions` binding exposes the same field in source; the optional Python module was not built for this validation.

The helper operates on a candidate scene, commits only after independent admission and final support/invariance checks, and otherwise retains the complete input map. It preserves registration flags and non-target poses, rejects nonfinite registered poses, and requires repaired cameras to pass alignment observability without removing existing reliable-camera support. It removes stale observations only for moved targets, preserves surviving old landmark coordinates, and never runs an additional global BA. It does not require or consume external reference data. This stage does not re-enable post-refinement missing-camera registration.

Reconstruction cache v42 includes the option in its key. A real 12-photo CLI check performs four successful runs: off, on, cached on, cached off. Enabling the flag produces a distinct finished reconstruction entry, while both cache hits reuse byte-identical calibration and camera geometry and skip the refinement stage. Cached frontend work remains reusable. CLI and Studio binaries are rebuilt in `build/photara/Release/`.

Four current mapping/submap/Studio/project-IO tests pass. They cover native nomination bounds and exclusions, transactional failure, independent pose-improvement acceptance/rejection, product callback order, default-off/forwarded Studio flags, project roundtrip and v13 compatibility. The legacy test fixture's truncation was updated for the appended v14 byte; production migration behavior was not loosened to make the test pass.

Product-entry replay evidence is under `artifacts/sfm-weak-production-20261020/`. London and Alameda start from their existing native frontend tracks, run full mapping through the actual product library entry point with the option enabled, and save `before.asfm` at the `before_weak_camera_refinement` callback. This compares the optional stage against its exact same-run input rather than attributing differences between separate nonlinear mapping runs to refinement. These are full mapping replays, not cold frontend-extraction benchmarks; Studio validation covers command generation and project persistence rather than a manually clicked large GUI run.

### Product-entry results

| Dataset | Registered | Repaired cameras | Shared pixels | Mean px before → after | RMS px before → after | P95 px before → after | Extra stage time |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| London | 1869 / 1874 | DSC07280 | 2902 | 1.154882 → 1.140010 | 5.838462 → 5.780667 | 2.803121 → 2.790825 | 33.750 s |
| Alameda | 1720 / 1734 | DSC07402, DSC07482 | 2697 | 1.560085 → 1.524763 | 3.794369 → 3.618662 | 4.225343 → 4.158039 | 94.568 s |

The stage reduces RMS by about 1.0% / 4.6% in these product runs. Maxima remain 255.332143 / 125.527514 px: major uncorrected tails persist. These absolute results do not beat the historical retained best; the new full mappings already have different geometry before the optional stage. The same-run comparison isolates the measured benefit without claiming to fix mapping-run variability or to guarantee an improvement on every external pixel. The modest benefit and extra runtime support keeping this feature opt-in.

Saved-output proofs confirm identical calibration bytes, registration flags, keypoint prefixes, all non-target camera pose bytes and surviving old landmark coordinates. London drops 253 old landmarks and Alameda 143 because removing stale moved-camera observations leaves fewer than two inliers; other old coordinates are unchanged. The final native checkpoints reload successfully and all three repaired cameras pass alignment observability. London DSC07280 and Alameda DSC07482 still carry conservative position-unverified status; the repair does not promote these warnings to precision certificates.

Denser reference checks confirm the actual product-stage repairs. London DSC07280, on all 40 eligible pixels, improves mean 6.782 → 1.771 px and P95 44.156 → 2.784 px. A second disjoint Alameda cohort improves DSC07402 (201 pixels) mean 5.360 → 3.771 px and P95 10.956 → 8.897 px; DSC07482 (202 pixels) mean 41.097 → 1.618 px and P95 76.728 → 3.612 px. No options were retuned after these evaluations. All nominated cameras are excluded from depth fitting in these denser checks, and there is no target-error filtering.

Whole mapping/export runs take 195.90 / 354.22 s from native frontend checkpoints. CLI smoke checks and some CPU evaluation overlapped other work, so these are observed runtimes rather than clean comparative speed benchmarks. Complete metrics, commands, binary/input hashes, pixel CSVs, invariant proofs, checkpoint-reload audits and logs are summarized in `artifacts/sfm-weak-production-20261020/validation-report.json`. Frontend-cache reuse across flag states was checked within the same build; source fingerprint changes may independently invalidate older builds' caches.
