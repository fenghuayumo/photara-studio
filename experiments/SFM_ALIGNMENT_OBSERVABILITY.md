# Alignment structural audit

The `alignment_reliable` diagnostic is a conservative structural screen, not a
proof of geometric accuracy or a full numerical rank test. Passing it does not
rule out parallel-rigidity degeneracy, weak parallax, or incorrect matches.
Failing it means this audit could not establish independent support; a camera
can still be constrained by other configurations such as resection.

The audit removes bridges from the verified camera-pair graph. Cross-block
landmarks count towards connecting blocks only when observed from distinct,
nonparallel camera rays within each block. At least three non-collinear shared
landmarks are required by this conservative similarity-support test. A point
triangulated jointly across a bridge must not certify its own depth in both
blocks. Duplicate observations and coincident camera centers do not supply
independent depth. Numerical tolerances are relative to floating-point scale,
not dataset-specific settings. Current positions provide the ray directions;
this is not independent local reconstruction or a photometric verification.

## Regression evidence

Reports: `artifacts/sfm_acceptance_20260905/*_independent_depth.run.json`.
All three runs reused tracks and reconstruction checkpoints, so their timings
are audit/cache-load timings, not fresh reconstruction benchmarks.

| Dataset | Registered | Pass structural screen | Result |
| --- | ---: | ---: | --- |
| chuan | 109 | 109 | Acceptance passes |
| lego1 | 300 | 300 | Reference center/rotation gates still fail |
| Alameda | 1701 | 1671 | Structural gate fails on 30 cameras |

Alameda now includes `indoor_DSC07344.JPG` through `indoor_DSC07358.JPG` among
the flagged cameras. The preceding shared-point implementation reported 1697
passing cameras and missed this drift branch. Geometry is unchanged: the
maximum reference center error remains about 66% of reference radius.
This change improves failure detection, not reconstruction accuracy.

Position-outlier recovery now builds an adjacency list once instead of scanning
all image pairs for every visited camera. Reachability changes from O(VE) to
O(V+E), preserving active-edge and registration conditions. End-to-end speedup
has not been measured; checkpoint runs bypass this recovery path.

Tests cover a bridge branch, circular shared-point evidence, independent depth,
and collinear shared landmarks. Existing mapping and acceptance tests also run.
Further geometry work should use independently reconstructed submaps and
validated registration against the stable map, retaining uncertain poses as
uncertain when the images do not provide sufficient evidence.
