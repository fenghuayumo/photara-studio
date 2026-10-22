# CUDA / Vulkan SfM comparison on the indoor image set

Measured on 2026-10-21, after the user paused Blender. On this dataset, CUDA
completed alignment about 19% sooner and used substantially less memory.
Vulkan scored better on the shared held-out observations and reference camera
comparison. The outputs are not numerically equivalent.

These measurements precede the Vulkan memory optimization. See the
[subsequent optimization record](SFM_VULKAN_MEMORY_OPTIMIZATION.md) for the
changed implementation and its validation; this report preserves the baseline.

## Inputs and configuration

- Input: `D:/BaiduNetdiskDownload/室内iphone/images/`, all 739 JPEG images,
  1.77 GiB in total; sampled images are 2560 x 3840. The directory name does
  not establish the capture device; the run logs include Canon EXIF metadata.
- GPU: NVIDIA GeForce RTX 5090 D v2, 24,455 MiB, driver 610.88.
- CPU: AMD Ryzen 9 7950X, 16 cores / 32 logical processors; about 32 GB RAM.
- Release build from commit `7234874`, with the pending documentation/comment
  corrections. No SIFT, matcher or solver algorithm was modified for this test.
- Both runs: global mapping, automatic camera model, 27,000 features, sequential
  window 3, default retrieval augmentation and weak-view recovery, no masks,
  no feature checkpoint reuse, no MVS or splat training.
- Both runs use four frontend worker threads. An experiment-only copy of the
  CLI sets the existing `config.frontend.thread_count` through
  `PHOTARA_BENCHMARK_THREADS=4`; the product CLI and its defaults are unchanged.
- CUDA: `siftgpu`, `gpu_mutual_ratio`, CUDA positioning and CUDA BA preferences.
  Vulkan: `vulkan_sift`, `vulkan_mutual_ratio`, Vulkan positioning and Vulkan BA
  preferences. Default CPU geometry remains in use.
- Sequential order: Vulkan, then CUDA; one measured run per backend. OS file
  cache was not flushed, and there are no repeat-run confidence intervals.
- GPU utilization before the restarted experiment was about 2%, with about
  1.4 GiB device memory in use. No Blender process was present at the check.

The experiment records commands, image SHA-256 hashes, executable hashes,
timestamps, resource samples and source status in
`artifacts/sfm_cuda_vulkan_iphone_20261021/idle_threads4/manifest.json` and
`inputs.json`. Its CLI source copy and preparation/runner scripts are in the
parent experiment directory. Generated artifacts are ignored by Git.

## Performance

| Measurement | CUDA | Vulkan |
|---|---:|---:|
| Complete SfM reconstruction, excluding final export | 343.746 s | 425.810 s |
| Process wall time, including startup/export and sampling overhead | 351.274 s | 435.305 s |
| Primary extraction stage, including host work | 26.284 s | 29.359 s |
| Primary descriptor matching | 14.423 s | 29.44 s |
| Match and geometry-verification container | 143.717 s | 155.156 s |
| Entire frontend | 212.627 s | 222.183 s |
| Global positioning, including CPU initializer and host work | 11.159 s | 45.149 s |
| GPU positioning solve calls, 9 calls | 1.866 s | 35.709 s |
| GPU bundle solve calls, 4 calls | 55.123 s | 75.292 s |
| Process peak working set | 2.49 GiB | 11.60 GiB |
| Sampled device-wide GPU memory peak | 6.82 GiB | 20.11 GiB |

CUDA reduced complete reconstruction time by 19.3% (about 1.24x throughput),
primary extraction time by 10.5%, and primary matching time by about 51%.
The GPU positioning and BA timings compare the resulting pipeline problems;
their tracks, observation counts and optimization trajectories differ. They
are not fixed-problem kernel benchmarks.

Stage times are inclusive and can overlap; do not sum table rows. The primary
matching figure is the CUDA pipeline's `match/verify timing` wall measurement
and Vulkan's `match image pairs` progress duration, both before geometric
verification. The extraction stage includes decoding/preparation and differs
in scheduling. GPU memory samples include desktop/driver allocations and are
not process-exclusive VRAM; sampling interval is approximately two seconds.

Both controlled runs completed successfully. All four BA calls and the
point-bearing positioning solve selected the requested GPU backend, with no
runtime GPU-to-CPU failure messages. The camera-only positioning initializer
used CPU by design on both runs.

## Reconstruction and independent comparison

Reference: the existing `D:/BaiduNetdiskDownload/室内iphone/sparse/0` COLMAP
model, containing 738 registered images and 222,484 points. It was read only
after reconstruction and was never supplied as an initializer or constraint.
It is a comparison model, not ground truth.

The evaluator rescales reference intrinsics and observations to the native
image grid. Camera comparisons use all 738 shared cameras in a least-squares
Sim(3), without removing outliers. Position differences are normalized by the
reference RMS camera radius, rather than reported as metric distances.

For held-out reprojection, both evaluations use the same seed, 2,007 sampled
reference tracks and 10,772 held-out observations covering all 738 reference
views. Camera poses/intrinsics stay fixed. A fresh point is fit on alternating
camera groups and projected into the excluded groups; reference XYZ is not
used, same-view observations stay together, and no error/depth filtering is
applied. This compares cameras on common evidence rather than their separately
selected reconstruction tracks.

| Measurement | CUDA | Vulkan |
|---|---:|---:|
| Registered input images | 739 / 739 | 739 / 739 |
| Sparse points | 737,589 | 670,436 |
| Reconstructed observations | 2,912,260 | 3,157,194 |
| Mean track length | 3.948 | 4.709 |
| Own-track reprojection mean | 0.570 px | 0.581 px |
| Own-track reprojection RMS | 0.726 px | 0.741 px |
| Own-track reprojection P95 | 1.489 px | 1.546 px |
| Behind-camera reconstructed observations | 0 | 0 |
| Shared held-out reprojection mean | 2.694 px | 1.661 px |
| Shared held-out reprojection median | 1.286 px | 0.993 px |
| Shared held-out reprojection P95 | 7.599 px | 4.454 px |
| Shared held-out reprojection RMS | 8.414 px | 4.613 px |
| Shared held-out reprojection maximum | 211.974 px | 261.570 px |
| Reference camera-center median difference / RMS radius | 1.851% | 1.052% |
| Reference camera-center P95 difference / RMS radius | 8.574% | 8.041% |
| Reference rotation median difference | 0.456 degrees | 0.215 degrees |
| Reference rotation P95 difference | 2.096 degrees | 1.937 degrees |
| Cameras without independently verified translation support | 30 | 20 |

CUDA's slightly lower own-track reprojection error does not establish better
camera accuracy, because the tracks differ. Vulkan reduces the common held-out
mean error by about 38% and P95 by about 41%, and is closer to the reference
on the median camera metrics. Its worst held-out observation is nevertheless
larger. Neither full registration nor these averages certify every camera.

Both runs differ from the reference by about 46 degrees on `DCIM5783.jpg`,
with a center discrepancy about 185% of the reference RMS camera radius. The
shared discrepancy is retained in the statistics; this comparison alone cannot
determine whether that reference pose or the reconstructed poses are correct.

Direct CUDA-to-Vulkan comparison over all 739 cameras, after Sim(3), gives a
median center difference of 1.170% of the CUDA RMS camera radius and a P95 of
1.733%; rotation median is 0.266 degrees, P95 0.585 degrees and maximum
2.813 degrees. This full-pipeline test changes extraction, matching and both
GPU solvers together, so it cannot assign the quality difference specifically
to SIFT, matching or the BA preconditioner.

## Earlier attempts and artifacts

Earlier runs while Blender was in use are excluded from the timing and quality
tables. The default-concurrency Vulkan attempt failed in descriptor matching
with `vkAllocateMemory failed (-2)`; another four-thread attempt also did not
complete while resources were shared. Those records do not establish an
isolated default-concurrency performance or memory limit. The restarted idle
four-thread experiment completed both configurations without an allocation
failure. An idle default-concurrency Vulkan run has not been established here.

Raw artifacts under
`artifacts/sfm_cuda_vulkan_iphone_20261021/idle_threads4/` include:

- `cuda/` and `vulkan/`: alignment logs, stage profiles, resource samples,
  `.asfm` reconstructions, COLMAP text exports and camera diagnostics.
- `cuda_vs_reference/` and `vulkan_vs_reference/`: JSON metrics, per-camera
  CSVs, held-out track CSVs and trajectory/error figures.
- `comparison.json`: combined metrics and the direct backend comparison.
- `performance_quality.png`: timing and held-out-error overview.

Evaluation uses `scripts/sfm_quality_compare.py`; stage summaries use
`scripts/summarize_sfm_profile.py`. The experiment's `evaluate_comparison.py`
adds the direct backend comparison and combines the reports.
