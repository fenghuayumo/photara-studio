# Complete alignment regression and Alameda diagnosis

## Scope and reproduction

The 2026-10-14 baseline starts from fixed **tracks checkpoints** and invokes
`reconstruct_from_frontend`: metadata restoration, mapping, BA, final recovery,
support pruning and position auditing are included. Feature extraction/matching,
checkpoint loading and output serialization are outside `alignment_seconds` and
must not be interpreted as complete cold-start pipeline time. Reconstruction
cache hits are bypassed. No reference model enters reconstruction or recovery;
reference models are used afterward for evaluation and are not absolute truth.

Build these developer targets with `PHOTARA_BUILD_BENCHMARKS=ON`:

```powershell
cmake --build build --config Release --target photara_sfm_reconstruction_benchmark photara_sfm_reference_eval photara_sfm_position_diagnostics photara_sfm_position_recovery_probe photara_sfm_stable_resection_probe
python scripts/sfm_full_baseline.py scenarios.json NEW_OUTPUT_DIRECTORY
```

The script requires `psutil` and expects Windows Release binaries in
`build/photara/Release-fixed-deps`. Scenario input is a JSON array; for example:

```json
[
  {
    "name": "alameda_auto",
    "tracks": "artifacts/sfm_alameda_fisheye_20261014/cache/tracks-175c0c3ca81c46b3.bin",
    "images": "D:/ScanVideo/alameda/images",
    "reference": "D:/ScanVideo/alameda/sparse",
    "positioning": "automatic",
    "ba": "automatic"
  }
]
```

Each case runs sequentially and stores logs, stage timings, actual backend paths,
per-camera reference errors, process OS peak working set and sampled GPU usage.
GPU memory is device-wide, including other applications; it is not the process's
exclusive allocation. The manifest records source, binary and input SHA-256
hashes alongside HEAD because a working-tree build may differ from HEAD.
Outputs use a new directory to preserve previous trials.

ASFM omits pair geometry. The benchmark therefore also saves the **final** graph
checkpoint in `<output-stem>.graph/reconstruction-KEY.bin`. Position diagnostics
must consume that checkpoint, not ASFM or the pre-solve tracks checkpoint: final
pair active flags affect observability and recovery. Diagnostics recompute graph
observability; they do not read the ASFM persisted risk array.

```powershell
photara_sfm_position_diagnostics reconstruction-KEY.bin NEW_CSV_PREFIX
photara_sfm_position_recovery_probe reconstruction-KEY.bin NEW_INDEPENDENT.asfm
photara_sfm_stable_resection_probe reconstruction-KEY.bin NEW_LAYERED.asfm
photara_sfm_reference_eval REFERENCE_DIRECTORY IMAGE_DIRECTORY RESULT.asfm NEW_PER_CAMERA.csv
```

The diagnostics CSV includes stable-only depth support, positive projection,
2-pixel support, image grid coverage, depth-point conditioning, graph boundary
support and pair translation/parallax measurements. Its two-view depth counts
use surviving inlier observations; recovery separately assembles raw pair/track
correspondences and requires at least three frozen source views. These counts
are distinct and must not be substituted for one another.

## Complete default baseline

Final working-tree results (512-camera threshold retained). Center errors are
normalized by the reference's median camera step; angles are degrees. Peak RAM
is the reconstruction process's OS peak working set, in GiB.

| Scene / positioning | Registered | Alignment s | Center P95 | Rotation P95 | Peak RAM GiB | Worst center camera / steps |
|---|---:|---:|---:|---:|---:|---|
| Alameda automatic | 1742/1742 | 103.391 | 0.126978 | 0.324315 | 3.676 | DSC07017.JPG / 3.191614 |
| Alameda CPU | 1742/1742 | 135.198 | 0.100066 | 0.297916 | 3.707 | DSC07020.JPG / 1.341761 |
| Berlin automatic | 1508/1511 | 93.775 | 0.072662 | 0.350611 | 1.470 | indoor_DSC00700.JPG / 0.183077 |
| Bonsai automatic | 292/292 | 72.811 | 0.033840 | 0.243660 | 0.946 | DSCF5644.JPG / 0.044707 |
| Flowers automatic | 173/173 | 6.905 | 0.027285 | 0.218024 | 0.472 | _DSC9142.JPG / 0.063782 |

Alameda/ Berlin automatic positioning actually used CUDA bearing refinement;
Alameda CPU used CPU bearing refinement. Bonsai and Flowers retain CPU primary
positioning because they are below the 512-camera threshold. BA used CUDA in
these runs. Berlin's
three unregistered images are `indoor_DSC00709.JPG`, `indoor_DSC00710.JPG`, and
`indoor_DSC00721.JPG`; they are not omitted from the registration denominator.
It marks 1238 of its registered views position-unverified under the conservative
graph audit; that flag is uncertainty, not a measured error count.

The preceding strict local-veto build took 152.634 s on Alameda automatic,
registered only 1741 cameras, had center P95 0.370315 steps, and a worst center
error of 5.068572 steps. Local discrepancies before final triangulation/BA
triggered an entire CPU primary retry. Deferring those local discrepancies while
retaining whole-scene rejection restores the existing healthy warm path; it does
not repair the approximately 3.19-step tail. These are individual local runs,
not repeated hardware-independent speed measurements. Berlin timing varied
between about 84 and 94 seconds while its pose metrics remained similar.

The final large-scene results are in
`artifacts/sfm_full_baseline_20261014_final/manifest.json`. Its Bonsai run was
interrupted and is excluded. Bonsai/Flowers reruns have a separate manifest in
`artifacts/sfm_full_baseline_20261014_final_remaining/`. Earlier complete
comparisons remain under `_v2` (strict local veto) and `_v3` (local deferral before
the resection rotation guard).

Alameda automatic spent about 25.5 s in positioning versus 58.0 s for CPU;
initial triangulation took about 14 s, and two representative CUDA BA scopes
took about 13.5 and 11.7 s. Stage scopes can nest and must not be blindly summed.
Final recovery and position auditing remain inside the reported alignment time.

## Alameda failure investigation

The automatic path retains all 1742 cameras and marks 89 as position-unverified.
The worst camera remains `DSC07017.JPG`, approximately 3.19 reference steps from
the reference after robust similarity alignment. Low reprojection RMS and full
registration coexist with this drift.

Frames `DSC07012` through `DSC07020` include an indoor segment. Image inspection
shows repeated wallpaper patterns; this is a plausible source of correspondence
ambiguity, not a verified per-feature attribution. Independent frozen-map depth
support sharply deteriorates at `DSC07015`–`DSC07020`: the preceding three frames
have 1545–2236 triangulated two-view depths and roughly 1.1–1.3 px P90 projection
error; later frames have 64–296 depths and roughly 5.7–11.6 px P90 error. For
`DSC07017`, only 21 of 100 such depths support the existing pose within 2 px.

Two reference-free offline recovery experiments were completed:

1. **Independent submap anchored to frozen stable depths.** Grouping by any
   active edge mixed weak capture sections. Using the existing strong,
   nonplanar baseline rule isolates the nine-frame segment and reconstructs all
   nine frames with 8662 local landmarks. Of 904 shared stable points, the fit
   split gives only 216/723 similarity inliers, below the existing 80% consensus
   requirement. All 49 candidate groups are rejected; no pose replacement is
   accepted and the consensus requirement is retained.
2. **Three bounded layers of stable-only resection.** Original stable poses and
   intrinsics are frozen and checked exactly; accepted views become anchors
   only on the next layer. The initial probe accepted 83 views but worsened
   `DSC07018` rotation error from about 0.85 to 2.73 degrees, despite lower pixel
   errors. An independent pair-rotation non-regression guard rejects that first
   harmful proposal. The guarded experiment accepts 81 views, still leaves
   `DSC07017` near 3.19 steps, and does not prove safe default propagation.

`DSC07017` has low fitting and validation consensus and insufficient independent
spatial/depth support. `DSC07020` has a roughly 98-degree disagreement with its
strongest verified translation constraint and fails pose consensus. Logging now
distinguishes insufficient validation image cells from ill-conditioned depth
points. Neither gate is weakened to force a repair.

## Default policy resulting from this work

- Keep the 512-camera warm-start threshold and existing automatic backend,
  parallel splitting, triangulation refinement and CPU budget policy.
- Whole-scene independent geometry regressions still reject warm refinement.
  Local intermediate discrepancies are logged and proceed through final
  triangulation, BA and position auditing instead of forcing an entire scene
  into a random CPU primary solve.
- Stable-only resection now rejects new rotation disagreement supported by at
  least two frozen pair witnesses, in addition to its existing geometric gates.
- The two recovery probes remain developer executables; no new CLI/Studio/Python
  product option or default layered propagation is introduced.

The baseline and probes support these conservative changes. They do not establish
that Alameda's drifting segment is repaired or that smaller scenes should adopt
warm-start by default. Further work needs genuinely independent cross-segment
correspondences and broader scene/hardware coverage.

## Verification

CLI, Studio, Python and the developer executables build successfully. The six
CTest suites for Python import, mapping, submap recovery, hierarchical mapping,
Studio alignment and project IO pass. Mapping tests retain whole-scene
drift/collapse rejection and add a healthy bulk/local-discrepancy case. Recovery
tests demonstrate that a pixel-consistent aliased rotation with at least 150/160
inliers is rejected by two frozen pair witnesses, while a correction of an
inaccurate inherited rotation is still accepted. The diagnostics tool rejects
an ASFM input instead of silently reporting a nonexistent pair graph.
