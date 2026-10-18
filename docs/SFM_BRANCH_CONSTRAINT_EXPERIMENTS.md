# Cross-branch constraints and joint missing-run initialization

Follow-up measurements of the bounded positioning quota, covisible anchors and
raw-match frozen-anchor verification are in
[SFM_BOUNDARY_RECOVERY_EXPERIMENTS.md](SFM_BOUNDARY_RECOVERY_EXPERIMENTS.md).

Experiment artifacts: `artifacts/sfm-branch-trace-20261018/`.

The control starts from the same cached verified tracks, with no reference
poses/depths supplied to reconstruction. Read-only callbacks record actual
rotation filtering, weight refresh, track rebuilding, positioning selection,
triangulation/BA, final support pruning and weak-branch cleanup. Binary and
source hashes, commands and resource logs are in `manifest.json`.

## Definitions that matter

The traced cohort is the **final missing list from the previous control**:
23 Alameda / 16 London images. A boundary in the raw CSV is therefore an edge
between that fixed cohort and its complement, **not necessarily a connection
to an independently trusted anchor**. Some complement cameras are themselves
withheld during cleanup and only recovered later. The analysis separately
labels these using this run's `weak branch pose withheld` events. Even the
remaining complement is not a formal frozen stable mask at every early stage.

Track totals are camera-track incidences, not counts of unique landmarks.
`shared_inliers` only requires an inlier in another registered camera outside
the cohort. It does not prove stable-only triangulation with two/three anchors.
`positioning_selection` reports incidence in the actual selected tracks; it is
a separate subset, not a destructive track-filtering stage. The additional selection-only pass separates minimum-track-length rejection
from eligible-but-unselected tracks. It does not inspect per-observation IRLS
quarantine. It starts after frontend verification and
cannot attribute absent pairs to retrieval versus descriptor/geometry rejection.

## Alameda constraint loss

For the fixed 23-image cohort:

- Rotation filtering deactivates 6 of 26 initially active boundary edges.
- The weight-3 track rebuild reduces cross-cohort incidences from 2,189 to
  1,378. Actual positioning selects 255 cross-cohort incidences.
- Immediately before weak-branch recovery there are 20 active boundary edges,
  1,340 cross-cohort track incidences, 1,143 shared inlier incidences, and 16
  still-registered cohort cameras. Two other cohort cameras were support-pruned.
- Weak-branch cleanup removes registration from those 16, deactivates all 20
  remaining boundary edges and rebuilds away their tracks. All 2,098 preserved
  boundary matches remain in inactive pairs. This is intentional cleanup after
  independent validation failure, not evidence that a valid constraint was
  incorrectly deleted.

**Thirteen of those 20 edges connect to other withheld cameras.** For example,
`DSC07333` connects strongly to `DSC07330`/`07331`/`07332`, and `DSC07334`'s
only active edge is to `DSC07332`. They cannot certify the missing segment's
placement in the original stable map. `07332` is recovered later in the control.

The two active `DSC07333` edges outside the withheld set lead to `DSC07379`
and `DSC07487`. They are classified planar, with composite weights **0.0157152
and 0.0211617**, far below global track-building threshold 3. Thus apparent
pre-cleanup cross-cohort depth was largely branch-internal support, rather
than an independently anchored boundary. Other surviving outside-withheld
edges have weights 0.187544 (`08183`), 1.32525 (`08182`), 2.36352 (`07484`),
3.00701 (`08180`) and 9.77131 (`07522`). A lone strong edge does not itself
provide independently triangulated stable depth.

Do not lower the global weight or weak-branch acceptance threshold on these
counts alone. Preserve uncertain matches as *proposals* for fresh, independent
local reconstruction and boundary validation.

## Joint initialization experiment

The offline group probe uses the same missing connected-run selection and
fresh dense local matching settings for the star control and pair alternative.
Native calibration is restored and all proposed matches are reverified. Target
submaps contain only their internal active pairs, keypoints and fixed intrinsics;
all inherited target poses, points and registration state are cleared.

The alternative tries at most eight strongest nonplanar two-view seeds with
at least 30 matches and at least 30 triangulated points at 2 px / 1 degree.
It grows the group through the existing incremental PnP/BA implementation,
keeping intrinsics fixed. It does not use image-order motion priors. It ranks
trials by reconstructed cameras and valid local landmarks. Local initialization
is repeated on a fresh internal submap by the independent recovery validator,
so the draft cannot supply inherited geometry to its own acceptance test.

All original Sim(3), independent stable-only depth, 80% consensus, withheld
geometry/pixel checks and minimum-30-observations-per-camera gates remain.
The initializer callback is empty in production: default star behavior is
unchanged. Star/pair trials rerun the same frontend settings rather than sharing
a serialized local frontend; exact proposal identity must not be assumed.

A synthetic four-camera chain with maximum degree two fails the star control
and succeeds with pair growth through the independent alignment validator.
A locally reconstructable chain without stable boundary depth stays rejected.
The test also verifies no parent poses/depths enter initialization and frozen
anchor poses remain exactly unchanged. Mapping, checkpoint, submap recovery
and Studio alignment CTests: **4/4 passed**.

## Positioning qualification versus selection

A separate read-only pass repeats rotations, weights and track rebuilding,
then stops immediately after actual positioning track selection, before any
position solve. Its executable/hash/commands are in `selection_manifest.json`.
Selected incidences agree with the complete traced runs. Counts below include
registered cohort cameras and registered outside-cohort observations only.

| Cross-cohort camera-track incidences | Alameda | London |
| --- | ---: | ---: |
| Before minimum-four-view qualification | 1378 | 442 |
| Two registered views | 413 | 322 |
| Three registered views | 201 | 116 |
| Eligible, at least four registered views | 764 | 4 |
| Actually selected | 255 | 4 |
| Eligible but unselected | 509 | 0 |

This distinguishes two different bottlenecks. Alameda loses both short tracks
and eligible boundary incidence during selection. London loses 438/442 at the
four-view qualification stage; **the 20k cap is not dropping its four remaining
eligible boundary incidences**. The total registered cohort incidence in London
is 2845 rather than the raw trace's 3928, because three cohort cameras are not
registered at selection time.

London's `DSC04985`–`DSC05010` segment has zero cross-cohort track incidence
already after the global track rebuild. Its selected tracks are internal to
the missing segment. `DSC06335` has 315 cross-cohort incidences, all two-view
(228) or three-view (87), and therefore none enter point-bearing positioning.
These are proposal/coverage observations, not proof that the rejected short
tracks are geometrically correct.

London has 11 active boundary edges before weak cleanup: two lead to other
withheld cameras, nine lead outside the withheld set, and only two of those
nine have weight >=3. Weak cleanup deactivates all 11 and clears the 11
remaining registered cohort cameras; two were previously support-pruned.

## Real missing-run results

All trials use the previous baseline final graph as the same frozen parent:
Alameda 1711 registered, London 1858. No trial supplies reference geometry.
The local frontend is recomputed with identical settings for star/pair, so
this is not a claim of byte-identical frontend proposals across the two runs.

| Missing run | Star local reconstruction | Pair growth | Independent shared points / outcome |
| --- | --- | --- | --- |
| Alameda 07333–07342 (10) | 10/10 | 10/10 | 14 in either trial; rejected below 30 |
| Alameda 08180–08184 (5) | initialization fails | best seed reaches 3/5 | below four-camera minimum; rejected |
| London 04895–04920 (5 input images) | initialization fails | best seed reaches 3/5 | below four-camera minimum; rejected |
| London 04985–05010 (6) | initialization fails | 5/6, independently rebuilt 5/5 subset | 0; rejected |

For the three-camera outcomes, the helper deliberately returns failure without
committing the partial scene, so the summary CSV reports `local_registered=0`;
the per-seed log retains the maximum of three. The London six-image experiment
does not constitute whole-segment reconstruction: its five-camera subset was
independently reconstructed and still failed at stable-map alignment.

**Validated additional registrations: zero for both datasets.** The offline
parent models remain at 1711/1734 and 1858/1874. No failed initialization or
locally reconstructed but unanchored camera is counted as registered. There
is therefore no newly accepted camera requiring external quality promotion;
no production initializer or acceptance threshold was changed.

## Complete traced reruns and reproducibility

The full trace reruns finish at **1710 Alameda / 1857 London**, one below the
previous controls in each case: `DSC07403` and `DSC05050` respectively. The
fixed traced cohort deliberately stays the old 23/16 list, so these additional
failures are outside its CSV rows. `rerun_coverage_delta.json` records them.
Do not present these runs as coverage gains or claim identical reproducibility.
The cause of this difference has not been isolated; this experiment does not
establish that diagnostics, runtime variability or the earlier resection fixes
are responsible. The resection variant in these trace reruns uses the current
`both` defaults, while the previous named `baseline` used both switches false.
The independent group comparisons above avoid this changed-parent confound by
using the same previous final graph for both initializers.

External reference pose evaluation is saved separately for both traced models.
Alameda center median/P95 is about 0.01338/0.03396 reference steps and rotation
median/P95 about 0.05163/0.16278 degrees. Similar average pose errors do not
justify ignoring the lost registrations or accepting unsupported segments.

## What the measurements justify next

1. Alameda: evaluate a bounded quota for geometrically credible cross-component
   tracks alongside the existing per-image/grid coverage; log independent anchor
   identities so branch-internal tracks cannot consume a stable-boundary quota.
   This is a hypothesis, not a measured coverage improvement.
2. London: first recover or verify two-/three-view boundary observations with
   independent parallax, cycle and stable-depth checks. Raising the global 20k
   cap cannot recover tracks that fail qualification. Do not globally lower the
   four-view requirement from this evidence alone.
3. For the reconstructed missing runs, prioritize multi-anchor correspondence
   and independent shared depth at both available boundaries. Changing the
   initializer alone cannot fix Alameda's 14 shared points or London's zero.
4. Before promoting any future recovery, repeat the full pipeline, resolve the
   two new missing-camera deltas, and validate whole-segment pose tails and
   paired external reprojection. Current local successes remain offline.
