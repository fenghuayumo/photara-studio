# Frozen-camera extension using fresh track associations

The subsequent joint refinement actually improves existing camera precision;
its final models and same-pixel results are documented in
[`SFM_RECONSTRUCTION_QUALITY.md`](SFM_RECONSTRUCTION_QUALITY.md).

The optional `photara_sfm_extend --fresh-tracks` path addresses a limitation of
boundary-only extension: a new correspondence can identify a landmark absent
from the retained map's selected tracks. Boundary-only matching cannot use
that landmark even when three retained cameras independently constrain it.

```
photara_sfm_extend baseline-full.asfm reconstruction-HEX.bin output.asfm --fresh-tracks
```

Both inputs must retain the original feature indices unless explicit merging
is requested. The tool checks image paths, extraction grid and every keypoint
in the common prefix before importing data; a longer retained suffix is kept.
It also accepts `tracks-HEX.bin`, so native verified frontend associations can
be tested without waiting for global reconstruction.

For divergent missing-search suffixes, `--merge-features` remaps candidate
associations by the exact float-bit tuple `(x, y, scale, orientation, response)`.
Every retained keypoint index remains intact. Distinct occurrences of duplicate
keypoints consume distinct retained indices; unmatched candidate occurrences
are appended. Both pair matches and track observations use the resulting map.
There is no nearest-pixel association or descriptor/pose-based identity guess.

Fresh tracks supply only observation associations. Their positions, inlier
labels and colors are cleared. All image poses, calibration and registration
state come from the retained model. The ordinary resection gates remain:
three frozen cameras per fitted depth, at least 30 fitting inliers, at least
12 held-out inliers, spatial spread, and pair rotation/direction checks. Newly
registered cameras never become anchors during the pass.

This option replaces track associations and retriangulates the sparse points;
it freezes cameras, not landmark bytes. It can append keypoints to retained
images when the compatible frontend supplies more features. Point colors are
invalidated and resampled from the source photos after triangulation. New poses retain
the product's `unverified` position-audit label. The default reconstruction and
the existing boundary-only tool behavior are unchanged.

## London validation, 2026-10-19

Evaluation artifacts: `artifacts/sfm-continued-20261019/london-fresh/`.
The final model with resampled point colors is in `london-release/model.asfm`
under the same artifact root. This is a retained-map
extension using a previously completed original-image correspondence run, not
a new cold reconstruction. The parent is the 1858-camera frozen extension from
`artifacts/sfm-balanced-frozen-extension-20261019/london/model.asfm`.

The new track path registers **1859/1874**, adding `indoor_DSC04915.JPG`.
Its fit has 31/47 inliers and its holdout 14/23, covering five image cells.
All 1858 old registered image records and all camera records are byte-identical
in the saved ASFM; no old camera is lost.

Evaluation uses the reference model only after reconstruction. Each tested
camera is excluded from the fresh depth fit, and target residuals do not select
or filter the evaluation pixels. DSC04915 has 83 pixels: mean **1.790050 px**,
P95 **4.661266 px**, versus reference mean **1.407398 px**, P95 **4.506613 px**.
On 2890 shared-camera pixels, both models have exactly the same mean
**0.981848 px** and P95 **2.815204 px**. The old maximum error of 69.906072 px
remains. This result improves coverage; it does not repair the existing pose
error tails or establish reference-level accuracy.

The native extension build and `photara.sfm.two_view`, `.mapping`, `.vocabulary`
and `.submap_recovery` checks pass (4/4). Run manifests record executable/input
hashes, resource sampling and subprocess exits. `frozen-proof.json` verifies
saved model records independently of the tool's in-memory pose assertions.

## Alameda continuation and comparison

The correct retained parent is
`artifacts/sfm-frontend-vocabulary-control-guarded-20261019/alameda/model.asfm`
with **1710** registrations. The 1709-camera whole-scene-vocabulary candidate
was rejected in the earlier investigation. Initial extension trials accidentally
used that experimental parent; they are excluded in `trial-exclusions.json`.

The complete balanced-retrieval run resumes saved features and primary matches;
it is **not a new cold run**. The earlier foreground interruption and detached
exit-1 attempt remain recorded separately. The sequential continuation completes
all 36589 rescue matches and reconstruction with exit 0, taking 557.30 s wall
time and peaking at 9.05 GB working set. Its model is
`artifacts/sfm-continued-20261019/alameda-sequential/model.asfm`.

| Same 2681 shared-camera evaluation pixels | Retained parent | Balanced global candidate |
| --- | ---: | ---: |
| Registered cameras | 1710 | 1720 |
| Mean error / px | 1.913158 | 1.599400 |
| P95 / px | 4.330761 | 4.172044 |
| RMS / px | 8.245253 | 4.750148 |
| Maximum / px | 219.990168 | 128.656554 |

Mean improves by 16.4% on the fixed shared cohort (1708 cameras). Twelve cameras
are gained: DSC07326–07327 and DSC07333–07342; DSC06474 and DSC06475 are lost.
On the lost cameras' separate fixed pixels, the old model has means 7.274406
and 224.776538 px, versus reference 2.613051 and 1.292350 px. Excluding those
lost cameras does not explain the shared-cohort improvement: neither model is
evaluated on their pixels in that comparison.

The twelve gained cameras' aligned center errors range 0.017089–0.037000 median
reference-camera steps and their rotation errors 0.084580–0.198876 degrees.
Their target-excluded pixel means range 1.605579–5.137991 px, versus reference
1.316435–4.093396 px. DSC07326 and DSC07327 still have mean/P95 errors
2.905804/8.895932 and 2.560356/7.969799 px. Late branch cameras have only 13–31
reference evaluation pixels. These are explicit limits: the complete candidate
improves aggregate alignment and coverage, but every new camera is not thereby
certified, and substantial old errors remain. Reference data is evaluation-only.

`independent-anchor-histogram.json` counts old registered anchors while excluding
all twelve gained cameras together. Most branch observations have no old-map
depth witness; some have multiple old anchors. This histogram is not a
triangulation-quality certificate or a new admission rule.

The independently evaluated 1720-camera candidate is retained for review as the
improved Alameda result. The balanced retrieval switch remains opt-in, since
the earlier complete London global candidate did not preserve the stronger
retained London precision. This is a per-dataset result, not a change to default
reconstruction policy.

### Frozen alternatives and exact feature merging

Alameda's independent missing-view searches produce different feature suffixes.
Strict prefix admission rejects them, as intended. Exact `--merge-features`
mapping preserves all **55086581** retained keypoints byte-for-byte in place,
appending ten new occurrences. Saved-model proofs confirm camera calibration
records and all 1710 old poses are byte-identical in both frozen alternatives.

Boundary-only extension registers DSC07327, giving 1711 without losing old
cameras. Its fit is 30/30, holdout 12/13 in four image cells. However, its 1123
independent reference pixels have mean/P95 **2.951373/8.463763 px**, versus
reference **1.316147/3.588060 px**. This is less precise than DSC07327 in the
global candidate and is retained as a conservative pose-preserving alternative,
not presented as the best Alameda result. Shared-camera means remain
2.0420332495/2.0420332058 px; the tiny difference is decimal pose export and
evaluation numerics, not moved saved poses. The shared cohort differs from the
global comparison and its aggregates must not be directly subtracted.

Fresh-track extension adds no Alameda camera (1710 remains); it is not adopted.
Its camera-pose equality is verified without repeating the same pixel evaluation.
The redundant second frozen pixel pass was stopped after the boundary result
and complete per-pixel CSV were saved. London frontend-track-checkpoint extension
also adds no camera beyond 1859 and is not adopted. All four SfM regression
checks pass again after the final merge implementation (11.98 s).
