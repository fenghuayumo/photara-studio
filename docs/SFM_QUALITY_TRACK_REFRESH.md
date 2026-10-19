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
