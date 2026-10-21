# Vulkan memory budget follow-up — 2026-10-21

Target: keep the 739-image indoor dataset below 6 GiB of process dedicated GPU
memory, with 8 GiB as the upper target, using the production automatic frontend
thread setting. Device-wide use is reported separately because it includes
other applications. Image resolution, feature limits and shader math are
unchanged.

## Allocation and lifetime changes

The earlier pool rounded every allocation to the next power of two. A
2560 x 3840 image at SIFT's first 2x octave has a 150 MiB float plane, but each
plane reserved 256 MiB. The 600 MiB keypoint image reserved 1 GiB, and the
300 MiB gradient image reserved 512 MiB. Eight float planes plus keypoint,
gradient and histogram buffers amplified this rounding waste.

Allocations up to 16 MiB still use power-of-two classes; larger feature buffers
round to 16 MiB multiples. These examples now reserve 160, 608 and 304 MiB.
Distinct large classes have distinct pool buckets, so a cached 48 MiB buffer
cannot satisfy a 64 MiB request. The existing 256/128 MiB idle device/host pool
limits and two-free-buffers-per-class limits remain in place. SIFT's active
memory estimate uses the same rounding function.

`VulkanMutualRatioMatcher::clear_prepared()` now also releases the device
descriptor cache and host mirror, resetting capacity. Weak-view extraction and
BA no longer inherit that invalidated cache. Match scratch remains reusable
within a matching phase, retaining the previous performance optimization.
These changes do not clamp features, resize images or alter matching decisions.

## Validation

- Release CLI and Vulkan test target built successfully; normal
  `photara.features.vulkan` CTest passed.
- Size-class tests cover 17, 33, 49, 129, 157 and 261 MiB requests, including
  adjacent large classes that previously shared a power-of-two exponent.
- Six real-image snapshots match the pre-optimization keypoint, float
  descriptor and quantized descriptor hashes exactly.
- The 27,000-feature resident matcher benchmark still checks every self-match
  index. Three 240-pair samples took 1.16463 / 1.11542 / 1.10935 s, comparable
  to the preceding 1.27167 / 1.16517 / 1.15263 s. Its internal tracked device
  allocation peak fell from 1,217,396,736 to 697,303,040 bytes. This is a
  feature-allocation counter, not total process or solver VRAM.

## Complete production-default run

All 739 images completed successfully through Vulkan extraction, matching,
positioning and BA, using the production CLI's automatic frontend threads
(32 workers in the log), 27,000 requested features, global mapping and default
CPU geometry. Feature checkpoints were disabled. Vulkan BA executed four calls;
there was no CPU BA fallback or out-of-memory failure.

| Measurement | Result |
|---|---:|
| Sampled process dedicated GPU memory peak | **4.064 GiB** |
| Sampled device-wide memory peak, including other applications | **6,096 MiB / 5.953 GiB** |
| Sampled process shared GPU memory peak, backed by system RAM | 1.422 GiB |
| Process peak working set | 3.095 GiB |
| Reconstruction excluding final exports | 282.485 s |
| Process wall time including exports | 291.404 s |
| Registered images | 739 / 739 |
| Points / observations | 670,517 / 3,157,330 |
| Own reprojection mean / RMS | 0.581195 / 0.740956 px |
| Common held-out mean / P95 | 1.648868 / 4.409984 px |

This run meets the dataset's below-6-GiB target for both sampled process
dedicated memory and sampled total device use under the recorded background
conditions. Startup device-wide use was 1,764 MiB at 2% utilization. Dedicated
and shared process peaks can occur at different times; do not sum them as a
simultaneous peak.

The previous performance follow-up measured 5.655 GiB process dedicated memory
with four workers. The new peak is about 28% lower despite testing the automatic
worker setting. Extraction's sampled process peak fell from about 4.65 GiB to
3.18 GiB. The new whole-run peak occurs during BA, whose allocator was not
changed in this follow-up. The faster total time cannot be attributed solely
to the memory change because worker count and background conditions differ.

The held-out comparison uses the same fixed-seed 2,007 tracks / 10,772 test
observations covering 738 reference cameras, with zero behind-camera test
observations. The preceding follow-up scored 1.651206 px mean / 4.410812 px P95.
These measurements indicate stable quality; the existing COLMAP model is a
reference, not ground truth. Full metrics and timing profiles are preserved in
`artifacts/sfm_vulkan_memory_budget_20261021/comparison.json`.

## Records and scope

`artifacts/sfm_vulkan_memory_budget_20261021/` contains the build log,
`sift_snapshot.txt`, matcher benchmark and full production CLI run under
`default_threads/`. Commands, input hashes, binary hashes and GPU startup state
are preserved in the manifest; full logs, profiles, resource samples, ASFM and
COLMAP exports are alongside it. Generated artifacts are ignored by Git.

The original CUDA/Vulkan comparison and later memory/performance reports remain
historical measurements. The earlier successful full comparisons used four
frontend workers; this follow-up uses the production automatic setting. Their
whole-process timings must not be interpreted as an isolated allocator speedup.
GPU process counters sample every 500 ms and device-wide counters every two
seconds. Measurements on this 24 GiB GPU do not by themselves constitute a
physical 6/8 GiB graphics-card compatibility test or a universal memory cap.
