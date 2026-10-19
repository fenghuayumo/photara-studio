# Alameda / London: cause isolation and implementation plan

Date: 2026-10-18. Baseline commit: `18c372b`.

Implementation and measured follow-up (2026-10-19):
[SFM_RECOVERY_IMPLEMENTATION.md](SFM_RECOVERY_IMPLEMENTATION.md).
The proposal repair and generalized alignment/segment-refinement prototypes
are implemented as explicit developer-tool variants. No default integration
has passed the real-data promotion criteria yet.

## New evidence: local recovery loses proposals before its own verifier

The group probe first calls `run_frontend` with a single supplied focal length,
trusted intrinsics and the default relative-pose minimum of 30 inliers. Camera
initialization uses equal fx/fy and a centered principal point. The group then
restores native calibration and runs a 15-inlier local verifier, but only over
`local.pairs` and their already-filtered matches. Pairs rejected by the initial
frontend are absent. The existing raw-match frozen-geometry variant restores
stable-to-stable proposals only, leaving target boundary and internal pairs on
this lossy path.

This is a limitation of the current offline recovery experiment, not evidence
that every production frontend should accept 15-inlier pairs. It also means
the previous zero-boundary result does not prove there were no tentative matches.

The new read-only `--audit-local-boundary` probe loads the exact saved local raw
matches and track snapshots, resolves cameras by full image path, and compares
the following counterfactual verifications without changing the parent map:

| Missing run | Boundary pairs tested | Existing subset, native / 15 | All raw, native / 15 | All raw, approximate / 15 | All raw, native / 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alameda 07333–07342 | 320 | 6 | 24 | 25 | 6 |
| Alameda 08180–08184 | 160 | 28 | 44 | 44 | 28 |
| London 04895–04920 | 160 | 0 | 13 | 13 | 0 |
| London 04985–05010 | 192 | 0 | 10 | 9 | 0 |

Corrected on 2026-10-19: all columns now explicitly trust their supplied
intrinsics, keeping the calibrated relative-pose solver branch identical.
The older audit preserved the native camera's `trust_intrinsics` flag; for
untrusted shared OPENCV cameras it could choose shared-focal estimation instead.
Its native counts therefore mixed calibration and solver-branch effects and
must not be used as a clean calibration ablation. The actual native proposal
experiment already set this flag explicitly and is unaffected by this audit fix.

Counts are accepted pair hypotheses, not newly registered images, verified
shared depths, or necessarily correct correspondences. The latter need independent
multi-view validation; repetitive texture may still produce false hypotheses.

For London, neither native-calibration run at 30 inliers accepts a boundary pair.
The 30-to-15 proposal threshold mismatch is therefore directly relevant. Changing
calibration alone is not a demonstrated cure: at 15 inliers, approximate and native
calibration accept different pairs, with similar totals. For Alameda, restoring
raw proposals increases candidate connectivity but does not establish an accurate
placement of the segment.

Corrected evidence: `artifacts/sfm-calibrated-audit-20261019/summary.json`, per-pair
`*.asfm.rounds.csv` and completion logs. Older outputs are preserved for provenance.
The parent is the previous frozen
control (1711 Alameda / 1858 London). No reference geometry enters these tests.

## Alameda's 6/24 fit consensus: what is and is not established

The previous frozen-geometry trial produced 691 candidate tracks. Only 32 had
two stable observations; 31 became independent shared points. Therefore the
largest measured loss occurs before successful stable-depth construction.

Of 24 fit points, only six agree with a similarity transform under the current
threshold. The code uses one isotropic 3D threshold: 1% of the median stable-point
radius. Stable depths may come from two views at a minimum one-degree angle.
It does not weight those points by triangulation uncertainty. The six-point
consensus alone cannot distinguish:

- Incorrect cross-boundary associations or wrong track unions.
- Correct observations with inaccurate, poorly conditioned depth.
- A locally distorted submap that cannot be aligned by one similarity transform.
- A systematic camera calibration mismatch.

Do not label all 18 other fit points as mismatches or treat the 1% gate as proven
too strict. Export per-point anchor identities, inlier observations, parallax,
pixel residuals and 3D residuals first. Refit with one anchor omitted at a time
and compare disjoint local windows. Stable local pixel fits with unstable depth
suggest conditioning; coherent window-dependent transforms suggest local shape
drift; image/feature conflicts suggest correspondence errors. These are proposed
diagnostic tests, not completed findings.

## Implementation order

### 1. Correct the local proposal path and measure usable boundary support

Use the same serialized raw descriptor matches for every ablation. Apply each
image's native calibration before target-target and target-anchor verification.
Apply the intended local 15-inlier proposal policy consistently to those raw
matches, including pairs absent from the initial frontend output. Keep the
30-inlier seed requirement and final registration checks independent of this
proposal policy. The production global frontend threshold need not change.

For lower-support boundary proposals, require track/cycle consistency across
multiple local cameras, retain source-anchor identities and exclude ambiguous
feature associations. Count newly usable boundary tracks and their anchor
coverage, not just accepted pair totals. Re-run both pair/star initialization
from identical serialized proposals to determine whether internal graph loss
also explains the three-camera partial initializations.

Acceptance for this step: an auditable increase in independently supported
boundary observations, no false registration in synthetic repeated-pattern
and disconnected-chain cases, and no change to frozen original geometry.
More candidate pairs alone is not sufficient to promote it.

### 2. Add multi-camera pose-and-scale alignment for sparse stable depth

The current alignment requires a feature to have a local 3D point AND an
independently triangulated stable 3D point. That discards useful local-point
observations seen in only one stable anchor, even when many such observations
jointly connect to several distinct stable cameras.

Prototype alignment from local 3D points to image rays in the frozen stable
cameras, jointly estimating rotation, translation and scale. Generalized
pose-and-scale estimation is an established formulation; see
[Sweeney et al., gDLS, ECCV 2014](https://sites.cs.ucsb.edu/~holl/pubs/Sweeney-2014-ECCV.pdf).
Its applicability here is a design proposal, not a measured Photara result.

Use robust hypotheses and pixel/angular refinement with scale-observability
checks. Single-anchor support cannot certify scale; require distributed anchor
baselines, multiple target cameras, spatially spread features and a well-conditioned
fit. Hold out entire feature tracks and additional anchor observations before
optimization. Score those observations against fixed original cameras. Independent
stable depths, where available, remain a separate geometric check.

This needs a new validator designed for these constraints; it cannot be presented
as passing the old 30-shared-3D-point validator. Test single-anchor scale ambiguity,
negligible baselines, planar/repeated patterns and wrong boundary links explicitly.

### 3. Refine the whole segment with a frozen boundary

Only after a validated placement exists, run bounded local BA over the missing
segment's poses and points, with original anchor poses and native calibration
fixed. Exclude withheld boundary observations from this fit. Compare held-out
pixel-error tails and local shape before and after refinement.

If shared-point diagnostics show depth noise is dominant, use uncertainty-aware
3D residuals plus independent pixel validation. If they show coherent segment
deformation, use overlapping local windows and fixed-boundary joint optimization.
Neither response is justified solely by the current 6/24 fit count.

### 4. Integrate only after paired end-to-end validation

Run control and candidate on identical frontend snapshots and sorted inputs.
Repeat enough runs to resolve the previously observed one-camera coverage deltas
before attributing small gains. Then perform fresh full-frontend reconstructions
on both datasets. Record the exact recovered/lost image names, per-segment held-out
mean/median/P95 errors, pose tails, runtime and memory. Compare the same evaluation
observations for shared cameras; changing random samples is not a paired test.

Reference models are evaluation-only. Measure newly registered segments against
original-anchor-only point fits, as well as shared-camera quality. Any claim about
3DGS improvement needs a separate matched training/rendering comparison.

## Priorities

Update after implementing these prototypes: see
[SFM_UPSTREAM_COMPARISON.md](SFM_UPSTREAM_COMPARISON.md). The ray-group experiment
did not recover real images; direct structure-less single-image registration
and geometry-gated track completion now take priority for London, while Alameda
needs association/depth/local-shape diagnostics and boundary-aware local windows.

Start with step 1: it addresses a directly measured information-loss path and is
small enough for a controlled ablation. Step 2 addresses the structural requirement
for shared stable depth. Step 3 targets precision after valid placement. Increasing
the global 20k positioning budget, relaxing final acceptance, or adding global BA
iterations does not directly address the measured failures and has lower priority.

Current validated parent coverage remains 1711/1734 and 1858/1874. This analysis
adds diagnostic evidence and an implementation plan, not a registration gain.
