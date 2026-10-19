# Cross-anchor precision experiment

Date: 2026-10-19. Follow-up to [the boundary sequence](SFM_BOUNDARY_SEQUENCE_EXPERIMENTS.md).
Default reconstruction and validated coverage remain unchanged: London 1858/1874,
Alameda 1711/1734. This experiment audits support and validation; it does not add
descriptor matches or promote candidate poses.

## Reference-free validation

`experiments/sfm_cross_anchor_precision.hpp` adds the developer-only
`--cross-anchor-precision` probe. It uses the same cached raw matches, native
calibration and original registered poses. No reference files are read.

The complete verified feature graph is partitioned into connected components,
including conflicting edges that track construction may decline to union. Every
fifth boundary component, in deterministic observation order, is reserved. All its
observations are removed from raw matches before independently re-estimating pair
geometry and rebuilding the pose-fit tracks. This excludes reserved pixels from
both target resection and its pair rotation/direction certificates. Ambiguous
components are withheld together but cannot supply evaluation depth.

Evaluation points require three or more original stable anchors. The last anchor
in deterministic image order is excluded from the depth fit; all remaining anchors
must triangulate at 2 px and one-degree parallax, and every stable observation,
including the excluded anchor, must project within 2 px. Target poses/pixels never
participate in this depth selection. Every eligible reserved target pixel is then
scored, without target residual/depth filtering; nonfinite results fail validation.

The experimental gate requires at least 30 evaluation features, six frame cells,
finite errors and both mean and nearest-rank P95 at most 2 px. These conservative
limits are an uncalibrated experiment, not a production accuracy guarantee.

The `control_overlapping` row reruns the existing resection with all components.
Its evaluation pixels can overlap pose fitting and its metrics are descriptive.
Only `component_disjoint` rows can pass the additional precision gate. Zero
evaluated counts mean no candidate pose reached evaluation, not zero pixel error;
`eligible_reserved_features` separately reports available evaluation depth.

Independence is conditional on the cached proposals and initial verified graph:
the reserve partition is selected after initial pair verification. This is not an
end-to-end unseen-descriptor experiment. Original parent poses/calibration and
cached feature extraction are shared between variants. Removing observations also
changes relative-pose consensus and available fit support, so rejection cannot be
attributed solely to a pixel threshold.

## Reproduction and validation

Final evidence is recorded in
`artifacts/sfm-cross-anchor-complete-20261019/manifest.json`, per-scene
`audit.asfm.rounds.csv`, process logs and resource records. Earlier development
replays are marked superseded and are not the final source/binary pair.

Every examined target already has raw pair attempts to all 32 stable cameras in
its local cache. This counts attempted pairs, not successful correspondences.
Simply retrying absent pairs within those same caches therefore provides no new
pair coverage. Additional matching must investigate different anchors or improve
feature correspondences in the existing attempts; this experiment does neither.

London's first group supplies no eligible reserved evaluation depths. In the
second group, DSC04990, DSC04995 and DSC05000 each have one; DSC05005 has three;
the other targets have none. These are well below the experimental 30-feature
minimum. Both complete-data control and disjoint fitting register no London
candidate.

For Alameda DSC08181, the complete-data control again passes the existing
resection. Its eight selected stable-depth evaluations include only four within
2 px; mean is 30.40 px, median 1.50 px and nearest-rank P95 221.78 px, over two
frame cells. This very small, overlapping diagnostic set differs from the earlier
603 reference-selected observations and must not replace that comparison. It does
not distinguish pose error from wrong associations or depth uncertainty. The
component-disjoint trial rejects DSC08181 before this additional precision gate;
it consequently provides no independent target-pose error estimate.

DSC08182 also has eight eligible reserved features; the other three targets in
that group have none. The support shortage prevents this experiment from
certifying precision, even if a candidate can pass the earlier resection checks.
No accepted precision result is counted as a coverage gain.

The other ten-camera Alameda group supplies only 0–6 eligible reserved features
per target. Across all four groups, the disjoint branch registers no candidate
and no row passes the additional precision gate. Both final processes exit zero;
the manifest's source hashes and copied executable hash match the final files.

```powershell
cmake --build build --config Release --target photara_sfm_stable_resection_probe photara_submap_recovery_test
python scripts/sfm_cross_anchor_trial.py NEW_OUTPUT
ctest --test-dir build -C Release -R 'photara.sfm.(two_view|equirect|mapping|checkpoint|submap_recovery|hierarchical|export_mvs)$' --output-on-failure
```

Synthetic coverage includes pinhole/fisheye positive cases, biased target poses,
negative depth, reserved-ancestry leakage, conflicting components, insufficient
anchors and corrupted excluded stable anchors. Seven relevant CTests pass.

The benchmark executable retains SHA256
`b6571747df100c99ef7e11063788f2988fd03fd84a11944dfd796221cce59a69`.
No default SfM implementation or acceptance threshold is changed.
