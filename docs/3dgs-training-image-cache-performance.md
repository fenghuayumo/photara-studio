# High-resolution training image reuse

## Problem and implementation

The indoor iPhone dataset has 738 loaded views, with 2560 x 3840 source
images. Packed RGBA supervision for all views is 27.03 GiB, larger than the
available adaptive host cache. A miss repeats JPEG decoding, CPU lens
undistortion/bilinear sampling, masking and packing. Background prefetch alone
does not hide this work on this dataset.

The shared CUDA/Vulkan training loader now uses lossless Zstd level 1 to retain
the *processed* RGBA, depth and normal buffers when the dataset exceeds the host
budget. Camera metadata and mask semantics are retained too. Restoration reuses
the exact channel bytes; it does not recompress JPEG, change training resolution
or approximate camera projection.

- Datasets fitting the host budget keep the ordinary decoded cache.
- Large datasets reserve at most 512 MiB (or a quarter of a smaller budget)
  for decoded lookahead, with room for at least the largest individual view.
  The remaining existing host budget bounds compressed entries.
- Compression and restoration run in the existing background prefetch tasks.
  The synchronous fallback still works when prefetch is disabled or late.
  All GPU calls remain on the training thread.
- Both tiers use the existing epoch-aware eviction order. A resolution change
  clears both tiers and joins outstanding work before using the new scale.
- If a sampled image cannot save at least 25% of its resident bytes, the
  loader conservatively returns to the ordinary decoded cache for that scale.
- Zero host-cache budget retains the existing single-view fallback. In-flight
  task buffers remain bounded separately by the existing prefetch limit;
  resident-byte statistics do not include those temporary buffers.

The decoded hot set stays bounded even if much of the compressed allowance is
unused. An experiment retaining decoded frames in all remaining host RAM used
12.68 GiB of resident cache and took 49.70 s for 1600 updates, eliminating the
overall gain. That policy was rejected. Increasing retained RAM is not itself
evidence of improved throughput.

## Real dataset protocol

Windows Release, CUDA 12.8, RTX 5090 D v2, ADC-IGS, same existing 994,506-row
PLY warm start, fresh optimizer state, iterations 16001--17600. Preview,
progressive resolution and geometry supervision are off. Holdout interval 21
leaves 702 training views among the 738 loaded views. The 4096 cap preserves
native 2560 x 3840 images; the old harness used uncapped native resolution,
which is the same image size here.

The benchmark includes the first pass building the compressed cache. GPU
telemetry is sampled every 200 ms. Later-epoch figures use iterations
16801--17600: eight 100-step stage-profile windows and GPU samples between
their boundary timestamps. This isolates reuse after the first training epoch.
The CUDA stage timeline includes host gaps; `data_load_ms` is not GPU kernel
execution time alone. Adaptive host budgets vary with available system RAM.

Baseline executables were preserved before editing. The runner records the
command, executable SHA256, exit status, optimization wall time, GPU telemetry,
and single training-view evaluation. The latter is only a smoke check; these
short runs do not establish held-out convergence or equal training trajectories.
Even baseline repeats produce different final image metrics.

## Measurements

| Metric | Baseline run 1 | Baseline run 2 | Final compressed cache |
| --- | ---: | ---: | ---: |
| 1600 updates, optimization wall time (s) | 49.3825 | 50.0795 | 41.3987 |
| Whole-run data-get wall time (s) | 35.8686 | 36.5795 | 23.6444 |
| Later-epoch CUDA timeline per update (ms) | 27.988 | 28.660 | 13.998 |
| Later-epoch data-load stage (ms) | 15.115 | 15.800 | 1.013 |
| Later-epoch GPU utilization (%) | 48.43 | 50.55 | 92.95 |

The final implementation cuts total optimization time by 16.75% against the
mean baseline (including first-pass compression), and later-epoch time by
about 50.6%. The compressed resident tier is 6.82 GiB; both tiers together are
7.30 GiB. There were 896 completed restores from the compressed tier, zero
async-upload failures and zero async-upload event waits. These restore counts
include background prefetch completions, not only foreground cache requests.

Artifacts under `build/`: `image-pipeline-before-1600`,
`image-pipeline-before-repeat-1600`, `image-pipeline-verified-4096-1600`, plus
their `.log`, `.gpu.csv` and `.summary.json` siblings. The baseline executable
SHA256 is `c0f873248f8d0bf5c526c1bc9090171fe513f7e6c75e920a588163609657a7f0`;
the executable used for that compressed-cache measurement is
`e86ec0bb536041f35b16b5e713d7cd5c238d4663ab4c4509dbfc645394e6dd94`.

The first pass has additional compression work and can be slower than the
baseline's first pass. The large gain starts when processed images are reused;
these measurements do not imply faster initialization or universally identical
speedups for short runs, incompressible images or different cache budgets.

A 2048-cap control run completed the same 1600 updates in 15.1494 s. It
retained 7.31 GiB of decoded supervision with zero compressed resident bytes
and zero compressed restores, confirming that the fitting-dataset path stays
uncompressed. This is a control/smoke run, not an old/new throughput A/B at 2048.
Its artifacts use the `image-pipeline-verified-2048-1600` prefix.

## Validation

The focused cache test compares RGB, gray, masks, depth and normals by float
bit pattern. It covers CPU undistortion, decoded-cache eviction, synchronous
and prefetched compressed restores after moving the original image away,
resolution invalidation, host-budget accounting, incompressible input fallback,
and fitting datasets that should stay uncompressed.

CUDA and Vulkan focused cache tests passed, as did the complete rasterizer
regression (including both cache backends). The Release editor, test and
throughput targets built successfully. Vulkan fixtures finish discarded uploads
before releasing their result tensors, so deferred work does not reuse their
storage during a subsequent fixture request.

## Checking a normal editor run

Normal runs now emit `splat_data_pipeline` at progress boundaries roughly every
two seconds, without enabling CUDA stage profiling or adding GPU fences. Each
line identifies the iteration, Gaussian count, actual image dimensions,
compression state, retained compressed views, completed restores in that
window, mean data-get wall time, resident cache size and step time. Cache
configuration is logged again on a resolution transition; an insufficient
compression saving logs an explicit fallback reason. The final cache summary
is also available without `profile_cuda`.

These counters distinguish a cold full-resolution pass, active compressed
reuse, a fitting decoded dataset, and a disabled compressed cache. The isolated
benchmark uses a 1M-row warm start with preview/geometry disabled; its 93%
later-epoch GPU utilization is not a promise for an editor run with a different
model, phase, preview or loss configuration. Verify the running worker path:
the bundled `dist/Photara-Studio-0.8.1-win64` executables dated October 27 do
not contain this change; the rebuilt worker is `build/photara/Release/photara.exe`.

```powershell
cmake --build build --config Release --target photara_splat_training_throughput photara_splat_test photara_studio --parallel 6
& build/photara/Release/photara_splat_test.exe --data-cache-only
& build/photara/Release/photara_splat_test.exe --data-cache-vulkan-only
ctest --test-dir build -C Release -R '^photara.splat.rasterizer$' --output-on-failure
```

The throughput harness accepts optional `START_ITERATION` and `MAX_RESOLUTION`
positional arguments. `run_splat_throughput.py` exposes `--start-iteration` and
`--max-resolution`; output directories must be new. Use the same dataset,
existing PLY, 1600 steps, prefetch 4 and profiling in both arms. Original data
and models are read only; results go under `build/`.

## Profiling-disabled verification

After a report of low utilization in the editor, both benchmark arms were
also run with stage profiling disabled, still using the same 1M-row warm start,
native resolution, 1600 updates and preview/geometry disabled:

| Metric | Old loader | Compressed loader |
| --- | ---: | ---: |
| Optimization wall time (s) | 65.0213 | 50.9298 |
| Whole training-window GPU utilization (%) | 36.42 | 54.85 |

The measured wall-time reduction is 21.67%, but the whole-run utilization is
only 54.85%. In the new loader's cold windows, data-get time averaged 40--59 ms
per request with zero compressed restores. By iteration 16700 all 702 training
views were compressed. Later windows showed 198--204 restores per 200 requests
and data-get averages of 0.066--0.424 ms. GPU utilization from the iteration
16900 diagnostic timestamp to optimization completion averaged 89.12% across
50 samples. Neither 89% nor the profiled run's 93% describes the entire run.

Artifacts: `build/image-pipeline-before-no-profile-1600` and
`build/image-pipeline-after-no-profile-1600`, with `.log`, `.gpu.csv` and
`.summary.json` siblings. The latter executable hash was
`edd106490a63fb406890e3930a026fcd45b56326e54fba5b0b7710f6da02907d`.
This does not establish why a particular editor screenshot was low: that
requires the actual worker path, iteration/model count and concurrent logs.

With the default progressive schedule, iteration 6001 switches from half to
full linear resolution and clears both host cache tiers. For this dataset,
rebuilding requires approximately one 738-view editor epoch (702 views in the
holdout benchmark). Utilization can therefore fall during the first hundreds
of steps after 6000 even with compression enabled. Sustained low utilization
after the full-resolution cache has been populated requires a separate check.

## Editor SfM reproduction and resolution-epoch fix

The editor uses `images.cache/sfm.bin`, not the earlier COLMAP warm-start
fixture. An isolated reproduction copies that exact working SfM (739 registered
views, 664,583 sparse points; SHA256
`316147042d06c85b59279ad1185a93ea8b9e1b2c8bf486500c901874e20e82b7`)
and follows the recorded editor settings: ADC-IGS, standard, MIP2D, 4096 cap,
progressive 3000-step levels, 3M capacity, no geometry/masks. Profiling and
preview are disabled. Both runs keep the 30K iteration schedule and 15K growth
stop, and deliberately terminate only their own worker after reporting step
10000. They are partial diagnostic runs, not completed 30K convergence tests.
Source images and the original working SfM are not modified.

The initial compressed-cache implementation left the shuffled epoch cursor
unchanged when clearing the image cache at a resolution transition. After
finishing the partial old epoch, its new shuffle repeats already-prepared views
while some old-epoch views are still cold. At step 7000 only 703 of 739 views
were compressed; the last views were not ready until about step 7400. Thus the
low-utilization period really can extend well beyond step 7000.

The trainer now starts a complete shuffled epoch whenever the image scale
changes. Every training view is prepared before it can be repeated at that
scale. The measured cache was complete by the step-6800 diagnostic instead of
7400. This changes view order and subsequent RNG trajectories at resolution
transitions; it preserves all images, per-epoch coverage and pixel data, but
does not claim identical learning trajectories or held-out quality.

| Iteration window | Before: GPU % | After: GPU % | Before: ms/update | After: ms/update |
| --- | ---: | ---: | ---: | ---: |
| 5000--6000 | 96.53 | 96.03 | 6.057 | 6.117 |
| 6000--7000 | 43.51 | 45.63 | 41.868 | 33.625 |
| 7000--8000 | 64.72 | 96.10 | 24.037 | 16.071 |
| 8000--10000 | 97.70 | 96.87 | 16.336 | 16.366 |

First-pass decode/undistortion/compression is still expensive: this fix shortens
the cold phase, rather than making the entire 6000--7000 window GPU-bound.
The full-resolution 6000--10000 segment takes 98.576 s before and 82.427 s after
(16.4% reduction). Later decoded-cache restore windows remain below 0.4 ms per
data request. The complete rasterizer regression, including CUDA/Vulkan cache
checks, passed after the epoch change; the Release editor was rebuilt.

Reproduction artifacts are `build/iphone-editor-repro-4096` and
`build/iphone-editor-epoch-fix-4096`. Each contains exact `command.json`, copied
`sfm.bin`, `train.log`, `gpu.csv`, `run.json` (marks the controlled termination)
and `phase-summary.json`. GPU percentages are mean 200-ms samples bounded by
the matching iteration timestamps; per-update time uses the wall interval.
Preview is disabled here, so these results do not rule out additional costs in
an editor instance that remains slow after all its views have been prepared.


## Stage diagnostics and cold CPU pipeline (2026-11-04)

`CacheStats` now carries completed-job CPU wall-time totals for file read/decode,
mask loading, projection/bilinear sampling/packing, priors, Zstd compression and
restoration. Background jobs return their counters with their futures; the
training thread alone aggregates them. `host_wait_ms` measures waiting for a
prefetched result; synchronous work remains visible in the individual stages
and `get_wall_ms`. `upload_submit_ms` is host API/submission/wait wall time,
not GPU kernel execution time. The counters require no new GPU event fences.
Parallel job durations overlap, so their sum is not training wall time.
Discarded old-scale futures contribute completed-job counters during a scale
change; unused failed old-scale jobs are logged and discarded.

Normal logs add `splat_data_stages` and `splat_data_memory` at the existing
roughly two-second cadence, before/after each resolution transition, and at
training completion. Stage deltas belong to the reporting window, not the
exact wall-time lifetime of each job. The analysis script reports these as
`completed_job_stage_sums`; a diagnostic window can straddle a phase boundary.

The memory accounting separates:

- `cache_payload_bytes`: image-cache payload only.
- `pending_payload_bytes`: payload currently assigned to ring uploads, already
  inside ring capacity; **do not add it to ring bytes**.
- `tensor_live_bytes`: unique logical allocations created by the loader,
  including decoded outputs still held by callers after cache eviction or a
  scale change. Shared depth/normal storage is counted once. Non-owning weak
  storage references do not extend GPU lifetime.
- `ring_bytes`: actual retained CUDA upload-slot capacity. Growing a slot
  allocates its replacement before freeing the old block; both allocations
  contribute to the logical peak and `ring_growth_peak_bytes`.
- `logical_peak_bytes`: high-water mark of tracked tensor payload plus ring
  capacity at loader allocation/return boundaries. Allocator padding,
  freed-but-retained buffers, driver internals and unrelated allocations are
  outside this logical measure.
- `pinned_host_bytes`: synchronous, free and in-flight pinned upload staging;
  this is host RAM, not device VRAM.
- CUDA allocator requested bytes, bucket padding and retained bucket cache,
  plus driver pool used/reserved/high-water values. These are process-wide,
  overlapping views, not extra loader allocations to add together. Driver
  pool counters can be unreliable under WDDM; `pool_valid` describes API
  success, not proof of physical residency.
- Vulkan buffer-pool reserved/live/free bytes on that backend, without querying
  CUDA. `device_used_bytes` is the existing device/heap sample and must not be
  mistaken for loader-only memory.

### Progressive benchmark and memory observations

The throughput harness now accepts optional `PROGRESSIVE` after
`MAX_RESOLUTION`; the Python runner exposes `--progressive`. A preserved
`ed674aa` loader executable and the instrumented/optimized executables use the
same 994,506-row PLY, 702 training views, 4096 cap, prefetch 4 with unchanged
adaptive policy, 30K schedule and model-only warm start at iteration 2800.
They run 5000 updates through iteration 7800, crossing both 3001 and 6001.
Preview, geometry supervision and stage CUDA profiling remain disabled.
This measures a resumed training window with reinitialized optimizer state,
not full training from scratch or a convergence comparison.

| Window | Original repeat (s) | Instrumented (s) | CPU optimized (s) | Final build repeat (s) |
| --- | ---: | ---: | ---: | ---: |
| 3000--3800 | 7.081 | 7.143 | 6.507 | 7.657 |
| 3800--6000 | 14.653 | 14.676 | 15.089 | 15.061 |
| 6000--6800 | 32.710 | 29.585 | 27.298 | 29.129 |
| 6800--7800 | 20.024 | 19.345 | 19.731 | 19.707 |
| Whole 2800--7800 window | 75.671 | 72.002 | 69.794 | 72.943 |

The first optimized full-resolution cold window is 7.7% shorter than the
instrumented control, and the whole window is 3.1% shorter. The final-build
repeat takes 72.943 s overall, slightly longer than the instrumented control;
its cold full-resolution window takes 29.129 s. This variability does not
establish a repeatable end-to-end percentage gain, despite the CPU byte-level
microbenchmark improvement. These are individual diagnostic runs, not confidence
intervals. Random refinement and system load vary. An
initial original run took 105.062 s and had much higher sampled device memory;
it is retained as an outlier/context check rather than used as evidence of
optimization benefit. The original repeat was run after the optimized arm.
No source images, input PLY or user worker processes were modified.

The instrumented full-resolution cold phase reports about 25.8 s accumulated
read/decode, 254.4 s packing and 60.4 s compression across 701 completed jobs.
Packing accounts for roughly 75% of those three stages; GPU JPEG decode alone
would leave the dominant CPU work. Later compressed restores have little
foreground waiting. The optimization therefore keeps CPU decoding and changes
only the row-major pixel loop and byte quantization: it removes per-pixel
integer coordinate division and replaces CRT `lround` calls with nonnegative
truncation after adding 0.5 in **double** precision. Widening preserves values
immediately below half-integers. RGB interpolation arithmetic, masks, camera
projection and cache/prefetch budgets are unchanged.

At both transition points the image-cache payload, pending uploads and tracked
image tensors drop to zero in this fixture. The ring remains at 4.6875 MiB
following 3001 and 18.75 MiB following 6001; after warming full resolution it
reaches 75 MiB, with an 84.375 MiB ring-only growth overlap. The loader logical
peak is 480.469 MiB in both instrumented and optimized arms, reached in the
quarter-resolution cache. Maximum sampled device-wide memory was 8865 MiB in
the original repeat, 8761 MiB instrumented and 8796 MiB CPU optimized. The
half-resolution window's sampled peak rose from 6148 to 6243 MiB between the
original repeat and optimized run. These observations show no added logical
GPU footprint from the CPU optimization; they do **not** prove identical
physical peaks. The final-build repeat samples 8809 MiB overall, including
8809 MiB in 6000--6800; its logical loader peak and ring capacities remain the
same. Sampling is 200 ms and model/allocator trajectories differ.

Artifacts: `build/loader-stages-baseline`, `build/loader-stages-baseline-repeat`,
`build/loader-stages-instrumented`, `build/loader-stages-cpu-optimized`,
`build/loader-stages-final`, their
`.log`, `.gpu.csv` and `.summary.json` siblings, and
`build/loader-stages-comparison.json`. Each summary records the executable hash
and exact command. Recreate the comparison with:

```powershell
python experiments/analyze_splat_pipeline.py build/loader-stages-baseline build/loader-stages-baseline-repeat build/loader-stages-instrumented build/loader-stages-cpu-optimized build/loader-stages-final --output build/loader-stages-comparison.json
```

### CPU byte-equivalence check

The harness has a CPU-only `--host-images DATASET IMAGES OUTPUT_CSV` mode,
preparing the first twelve views at quarter, half and full scale. Run the
reference and optimized binaries with `OMP_NUM_THREADS=1` to match a background
prefetch job. All 36 output RGB/mask hashes match. Reference versus optimized
preparation totals (excluding hash computation) are 547.6/520.0 ms at quarter,
1412.7/1310.6 ms at half and 4403.7/3964.1 ms at full resolution. This first
single-thread microbenchmark shows a 10.0% full-resolution reduction, separate
from the end-to-end training result. Artifacts are
`build/loader-host-reference.csv` and `build/loader-host-optimized.csv`.

A scalar interpolation oracle additionally checks randomized PNG up/downsampling
against `std::lround`, including border behavior and half-integer cases. Cache
tests check retained-output and upload-ring accounting across scale changes.
CUDA/Vulkan focused cache tests and the complete rasterizer regression pass.
The final Release test, throughput and Studio targets build successfully.

The next measured optimization target remains CPU projection/sampling/packing.
GPU decoding should only be reconsidered together with that stage, including
its extra device scratch and any download needed for the host compressed cache.
No cache-budget or prefetch-count increase is part of this change.


### Implementation cleanup

The final runtime has one image preparation implementation. The unused
`upload_packed_training_pixels` wrapper was removed; upload and buffer tracking
now remain in the loader, which calls the shared CUDA pixel expansion kernel.
Historical benchmark executables (`before_image_cache`, `ed674aa`,
`host_reference`, `telemetry`) were deleted after verification. Their logs,
commands, hashes and measurements remain as evidence; rerunning those historical
arms requires rebuilding the corresponding revision. The scalar test oracle
is validation code, not an alternative runtime implementation. CUDA/Vulkan
backends and decoded-cache fallback remain supported paths of the same loader.


## Further cold-pass projection preparation

The loader already overlaps background host jobs with current-view GPU work,
using the same adaptive lookahead and two device upload slots. A lookahead
queue hides latency only while the producer sustains the GPU's consumption
rate. At a new resolution every training view is cold; host jobs still perform
read/decode, resampling/packing and lossless compression before publishing the
future. Compression is therefore currently part of the ready-to-consume
critical path. Warm compressed-cache restores avoid the expensive first-pass
projection and have almost no foreground future waits in this fixture.

`ImageSourceCoordinates` now prepares the separable x/y axes once per image
column/row and evaluates only projected x/y, without the generic BA projection
Jacobian. The previous per-pixel coordinate helper is replaced, not retained
as a selectable runtime path. Brown and fisheye projection expression order,
float pixel conversion and JPEG metadata-space scaling are preserved. No
image-sized remap cache is introduced: temporary axes are O(width + height),
about 50 KiB for a 2560x3840 image, and are released after host preparation.
The direct-copy path does not allocate these axes. GPU allocations, cache
budgets, compression level and prefetch policy remain unchanged.

The same 36 real-image RGB/mask hashes match the previous scalar implementation
at quarter, half and full resolution. Single-thread preparation totals are
428.3, 946.9 and 2710.4 ms, versus 520.0, 1310.6 and 3964.1 ms in the previous
row-loop microbenchmark: a 31.6% full-resolution reduction. These timings are
individual runs; output equality is the stronger invariant. The independent
scalar image oracle now also checks Brown-distorted and fisheye-to-pinhole
projection against `project_camera_plane`, in addition to linear resizing.

The matched 2800--7800 training diagnostic was repeated on the same dataset,
model and settings used above, with no other training arm running concurrently:

| Metric | Before axis preparation | After axis preparation |
| --- | ---: | ---: |
| 6000--7000 wall time (s) | 31.373 | 29.403 |
| 6000--7000 mean sampled GPU utilization (%) | 58.16 | 67.88 |
| 6000--6800 foreground future wait (s) | 13.68 | 10.75 |
| 6000--6800 accumulated packing time (s) | 224.63 | 167.82 |
| Whole 5000-update window (s) | 69.782 | 68.837 |
| 6000--7000 sampled device peak (MiB) | 8700 | 8496 |

The cold interval improves by 6.3% and GPU utilization rises by about 9.7
percentage points in this pair. It still contains waits; this does not imply
that prefetching can sustain the GPU indefinitely when CPU preparation is
slower. Accumulated stage times overlap and are reported by completion window.
Model refinement is nondeterministic and device memory is sampled every 200 ms,
so the peak difference is not evidence of a guaranteed VRAM saving. The loader
logical peak remains 480.469 MiB and full-resolution upload-ring capacity
remains 75 MiB.

Artifacts: `build/loader-cold-pack-before`, `build/loader-cold-pack-after`, their
`.log`, `.gpu.csv` and `.summary.json` siblings; `build/loader-host-coordinates.csv`
and `build/loader-cold-pack-comparison.json`. The analyzer additionally reports
6000--7000 and 7000--7800 windows. The initially tried packet-based RGB sampler
was slower in the microbenchmark and is not part of the runtime.

Release test/throughput/Studio builds, CUDA/Vulkan cache tests and the complete
rasterizer regression pass with the coordinate preparation change. The final
same-window repeat (`build/loader-cold-pack-after-repeat`) reports 62.165 s
for all 5000 updates, 24.809 s and 77.37% sampled GPU utilization for
6000--7000, and 6.72 s foreground future waits in the cold 6000--6800 window.
Accumulated packing is 136.85 s in that window, versus 224.63 s before.
The two optimized runs span 24.8--29.4 s and 67.9--77.4% utilization in
6000--7000; report that range rather than a guaranteed acceleration factor.
The final repeat's sampled peak is 8146 MiB for 6000--7000 (8307 MiB over
7000--7800), with unchanged logical loader memory/ring capacities.
