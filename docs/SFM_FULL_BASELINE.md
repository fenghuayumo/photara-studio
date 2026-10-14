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

Scenarios may also specify `"mode": "hierarchical"` and
`"cluster_capacity": 512`. Omitting `mode` retains Global. The developer
benchmark accepts these after its optional BA backend:
`tracks-HEX.bin automatic NEW_OUTPUT.asfm automatic hierarchical 512`.
Both modes invoke the complete reconstruction and final auditing entry point.

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

## Baseline before localized Global recovery

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

## Berlin historical-best reproduction

Current reconstruction sources at `16b1313` were rebuilt and evaluated through
the complete entry point on 2026-10-14. The old and current tracks files have
different checkpoint keys/file hashes, but their payloads after the 40-byte
header are byte-identical: SHA-256
`3981a5ede52cbf139074869ec7b9bc5bbe46f7aee01ee8335070e3a70ffd897b`.
Thus the two 512 runs also check repeatability on identical input geometry.
All four runs bypass reconstruction caches and include final recovery, BA,
support pruning and position auditing. Reference evaluation happens afterward.

| Path | Registered | Center P95 / steps | Rotation P95 / degrees | Worst center / steps | Worst rotation / degrees | Alignment s | Peak RAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical Hierarchical 512 + rotation recovery | 1511/1511 | 0.032770 | 0.165872 | 0.067147 | 1.104738 | 313.531* | — |
| Current Hierarchical 512, old checkpoint | 1511/1511 | 0.035335 | 0.168239 | 0.067311 | 1.098003 | 301.030 | 6.078 |
| Current Hierarchical 512, current checkpoint | 1511/1511 | 0.035168 | 0.168199 | 0.067312 | 1.088987 | 293.066 | 6.078 |
| Current Global | 1508/1511 | 0.072838 | 0.351078 | 0.183231 | 2.172189 | 81.937 | 1.464 |
| Current Hierarchical 2048 | 1508/1511 | 0.072687 | 0.350615 | 0.183155 | 2.171697 | 82.329 | 1.519 |

*The historical timing was a directory reconstruction call including cached
frontend loading and is not the same timing scope. Current timings are individual
local runs and do not establish hardware-independent performance claims.

Full coverage and its image set reproduce twice. Quality remains close to the
historical best, with center P95 approximately 7%–8% higher and rotation P95
approximately 1.4% higher; exact historical quality is not reproduced. Current
512 reprojection RMS is 0.62349–0.62350 pixels, versus historical 0.62647 and
current Global/2048 approximately 0.62460.

Both 512 runs create 11 subscenes, merge all 11 at 1507/1511 coverage, then
register `indoor_DSC00709.JPG`, `indoor_DSC00710.JPG`, `indoor_DSC00721.JPG`, and
`indoor_DSC00723.JPG` through post-merge resection. The first three remain absent
in Global/2048. In the current-checkpoint 512 run their center errors are
0.02455/0.02662/0.02996 steps and rotation errors 0.48758/0.44617/0.69316 degrees.
Both rotation-recovery checks run without logged candidate nominations; these
trials therefore trace the restored coverage to post-merge resection, not a
rotation-repair action. Hierarchical 2048 logs the single-cluster Global fast
path: its intermediate 1511 registration count becomes 1508 after final audits.

Read-only audits of the saved **final** graphs find zero severe verified
rotation conflicts in all four runs (8138 strong edges for 512, 8266 for
Global/2048), using the production thresholds. Conservative position support
still marks 1266 views unverified for 512 versus 1238 for Global/2048. Low
reference errors do not resolve that lack of independent graph support.

The result identifies local submap geometry and post-merge resection as a
candidate for selective recovery in the default Global path. It does not
isolate the contributions of star initialization, merging, shared intrinsics,
or subsequent BA. Investigate independent local submaps for the missing views,
with stable-map alignment, held-out support and rotation validation, before
changing product defaults. Whole-scene 512 reconstruction has materially higher
time and RAM cost in these trials; no default capacity change is made here.

Reproduce with:

```powershell
python scripts/sfm_full_baseline.py artifacts/sfm_hierarchical_berlin_20261014_scenarios.json NEW_OUTPUT_DIRECTORY
```

Evidence is retained in `artifacts/sfm_hierarchical_berlin_20261014/`: the
manifest contains input/source/binary hashes and commands, `input_equivalence.json`
records payload hashes, and `final_audit.json` records camera sets, final-graph
conflicts and the recovered cameras' reference errors. Logs, ASFM models,
final graph checkpoints, per-camera evaluations and diagnostics are preserved.

## Current Global recovery and timing

The later 2026-10-14 Global improvement keeps the existing reconstruction mode,
2048 cluster capacity and 512-camera positioning threshold. It addresses a
specific recovery rejection: Berlin's three missing views had independently
triangulated PnP/held-out support, but the successful pixels occupied only one
or two cells of the fixed full-frame 4x4 grid. A conditional patch-relative
coverage check now requires independent pair rotation/direction witnesses,
pixel extent/conditioning and all existing held-out/3D checks. It applies only
to unregistered views. `00710` initially fails the extra rotation evidence;
after `00709` and `00721` recover, the next frozen-anchor pass accepts it.
No reference poses enter this process.

Performance changes reuse failed triangulations only while their observations,
poses and registration inputs are unchanged, and restrict local-BA
retriangulation to tracks incident to free or recovered cameras. Distant frozen
observations on those tracks remain in the solve. Berlin's two local scans now
select about 132500/132800 of 1080000 tracks instead of retriangulating the whole
scene; late full-scene BA and final audits remain enabled.

| Complete fixed-track path | Registered | Alignment s | Center P95 / steps | Rotation P95 / degrees | Peak RAM GiB |
|---|---:|---:|---:|---:|---:|
| Earlier Global control | 1508/1511 | 81.937 | 0.072838 | 0.351078 | 1.464 |
| Localized recovery, before performance changes, run 1 | 1511/1511 | 93.636 | 0.072721 | 0.352625 | — |
| Localized recovery, before performance changes, run 2 | 1511/1511 | 86.163 | 0.072387 | 0.352481 | — |
| Current Global, run 1 | 1511/1511 | 74.398 | 0.072039 | 0.351473 | 1.603 |
| Current Global, run 2 | 1511/1511 | 73.572 | 0.072148 | 0.351711 | 1.605 |

Both final Global runs retain the three recovered views. Their center errors
are approximately 0.0534/0.0456/0.0459 steps and rotation errors
0.3576/0.3738/0.5560 degrees for `00709`/`00710`/`00721`. Full registration no
longer requires Berlin's whole-scene multi-cluster reconstruction. Global's
overall pose quality remains near its previous level, with worst rotation
approximately 2.17 degrees; it does not reproduce the lower center/rotation P95
of Hierarchical 512. Conservative position auditing marks 1241 views unverified,
including the recovered views, versus 1238 in the earlier 1508-view result.

The measured recovery scope decreased from 43.646 s in the earlier Global
control to 35.784/35.583 s, while mapping remained about 37.3 s. Against the
localized-recovery trials without performance changes, complete alignment was
about 15%–21% shorter. These are two local repeats on the same machine and
fixed tracks, not a general speed guarantee or cold frontend measurement.
Geometry optimization and track peeling can change landmark membership;
current Berlin has approximately 1034425 landmarks and 4457790 observations,
versus about 1036810/4463160 before the performance changes. Reprojection RMS
is about 0.62497 pixels versus 0.62451, a small increase that accompanies the
measured pose/coverage checks rather than an assertion of identical point clouds.

The initial recovery-only comparison is retained in
`artifacts/sfm_global_recovery_20261014_patch/`. The final compiled algorithm
and repeated Berlin runs are in
`artifacts/sfm_global_recovery_20261014_final/manifest.json`; all inputs,
commands, source/binary hashes, stage timings, final graphs, diagnostics and
per-camera evaluation files are retained. Its `final_audit.json` also compares
the original 1508-camera set separately from the newly recovered cameras.

```powershell
python scripts/sfm_full_baseline.py artifacts/sfm_global_recovery_20261014_final_scenarios.json NEW_OUTPUT_DIRECTORY
python artifacts/sfm_global_recovery_20261014/audit.py NEW_OUTPUT_DIRECTORY
ctest --test-dir build -C Release -R '^photara.sfm.(submap_recovery|hierarchical|mapping)$' --output-on-failure
```

Synthetic regression coverage includes localized valid support, rejected
insufficient extent/rotation evidence, unchanged rejected poses, cached failed
depths, newly registered anchors invalidating cached failures, and numerical
comparison with full rebuild/triangulation. Global mapping and Hierarchical
tests also pass. Product CLI and Studio require no new tuning switches.

Final cross-scene controls retain Alameda 1742/1742, Bonsai 292/292 and Flowers
173/173. Center/rotation P95 are respectively 0.126906/0.324254,
0.033840/0.243662 and 0.027285/0.218024 (steps/degrees), close to their earlier
controls. Five final graph audits, including both Berlin repeats, report zero
severe verified rotation conflicts. All original 1508 Berlin cameras remain;
their separate center P95 is 0.072118/0.072243 and rotation P95
0.348731/0.348838, versus 0.072838/0.351078 before the change. The overall
1511-view metrics in the table also include the three additional cameras.

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
