# Vulkan matcher allocation overhead — 2026-10-21

The preceding memory optimization and its benchmark records were committed as
`e7d6f52`. This follow-up keeps the bounded idle pools and changes the lifetime
of active matcher scratch buffers.
The subsequent [memory budget follow-up](SFM_VULKAN_MEMORY_BUDGET.md) reduces
allocation rounding and releases invalid descriptor mirrors; the measurements
below precede that additional change.

## Change

The top-two row/column partials can round up to 512 MiB each on large pairs.
They exceed the 256 MiB idle device-pool limit, so releasing them after every
batch forced fresh Vulkan allocations in the next batch. One workspace is now
kept by the shared matcher implementation, grown only when needed and reused
by all clones under the existing mutex. `clear_prepared()` releases the scratch
at the phase boundary. Shader math, pair order, ratio tests and match output
ordering are unchanged.

This retains active scratch between batches; an idle-pool limit is not a cap
on active work. The workspace is shared rather than multiplied by frontend
workers. End-of-batch live allocation is consequently higher than before;
compare peak memory and phase-boundary release, not end-of-batch live bytes.

Cache eviction now invalidates offsets and reuses the existing allocation.
Cache growth drops inactive partials before allocating the larger mirror, so
the old/new descriptor buffers do not overlap the retained partial workspace.
The frontend also clears prepared Vulkan matching state before weak-view
feature augmentation switches back to SIFT extraction. Keeping match partials
through that transition would overlap both large workspaces.

## Isolated allocation benchmark

Release build on the same RTX 5090 D v2. The optional test mode
`photara_vulkan_features_test --benchmark-matcher FEATURES 30` creates fixed-seed
quantized normalized descriptors, checks every self-match index/count and
measures three samples after warmup, each with 240 pairs in batches of eight.
Extraction, geometry and BA are excluded. Both executables use the same test
driver, descriptors, kernels and options; the matcher workspace lifetime is the
algorithm implementation difference. Other applications remain resident.

| Descriptor count per image | Before: three samples | After: three samples | Median change |
|---|---|---|---|
| 12,000 | 0.251771 / 0.236404 / 0.233823 s | 0.252987 / 0.239541 / 0.249246 s | +5.4%, overlapping sample ranges |
| 27,000 | 1.629420 / 1.637700 / 1.680700 s | 1.158990 / 1.121020 / 1.119320 s | -31.5% |

The large case benefits from avoiding repeated oversize allocations. The small
case already fits the bounded idle pool and shows no demonstrated benefit.
This is an allocation microbenchmark with repeated resident descriptors; it
does not predict the full dataset's cache misses, uploads or end-to-end speedup.
Normal `photara.features.vulkan` CTest passed after the implementation change.

A repeat on the final cache-growth implementation measured 1.271670 /
1.165170 / 1.152630 s for 240 large pairs: median 28.9% below the original
1.637700 s. Three samples are not confidence intervals.

The final `--benchmark-matcher-growth 27000 240` check rotates descriptor rows
and increments their generation each batch. It checks the shifted match index
of every output across cache growth and eviction. Three samples of 1,920 pairs
took 12.2923 / 12.8926 / 13.1334 s; peak tracked device allocations were
2,291,138,560 bytes (2.13 GiB), including retained idle device buffers, and did
not grow across samples. This is an internal feature-allocation counter, not
process-wide VRAM or solver memory. CTest also checks workspace release and
reconstruction after `clear_prepared()`, plus concurrent shared clones.

## Complete 739-image validation

The final phase-release implementation completed all 739 images with four
frontend workers, the same reconstruction options, CPU geometry, Vulkan SIFT,
matching, positioning and BA. No dense reconstruction or training was run.

| Measurement | Previous memory-optimized record | Final performance follow-up |
|---|---:|---:|
| Primary matching, 23,473 pairs | 67.25 s | 54.55 s |
| Reconstruction excluding final exports | 467.728 s | 469.879 s |
| Process wall time including exports | 475.810 s | 477.811 s |
| Process peak working set | 4.90 GiB | 4.38 GiB |
| Whole-run sampled process dedicated GPU memory | Not measured from startup | 5.655 GiB |
| Sampled device-wide GPU memory | 11.40 GiB | 9.26 GiB |
| Registered images | 739/739 | 739/739 |
| Points / observations | 670,362 / 3,157,222 | 670,410 / 3,157,404 |
| Own reprojection mean / RMS | 0.581167 / 0.740875 px | 0.581230 / 0.740973 px |

The observed primary matching time fell 18.9%, but **there is no demonstrated
end-to-end speedup**: total time was essentially unchanged. These were single
runs with different background conditions. The previous run started with about
5 GiB of other GPU memory and 93% GPU utilization; the final run started with
about 1.8 GiB and 2% utilization. Device-wide memory includes other applications.
The isolated matcher benchmark is the evidence for reduced allocation overhead;
the full runs establish successful execution and dataset-specific outcomes.

Intermediate scratch retention raised sampled process dedicated GPU memory to
6.656 GiB by overlapping the weak-view SIFT pass. The final phase-release run
reduced that peak to 5.655 GiB. Process counters sample every 500 ms, and
device-wide samples every two seconds; neither is a mathematical peak bound.
The final frontend retained 14,609 verified pairs and augmented the same 23 weak
images with 23,569 features. Vulkan BA ran four calls totaling 77.839 s; global
positioning took 44.124 s and alignment quality auditing took 57.840 s.

On the same fixed-seed 2,007 held-out reference tracks / 10,772 observations,
the final mean was 1.651206 px and P95 4.410812 px, versus 1.652723 / 4.407924 px
for the preceding memory-optimized run. All 738 reference views were covered,
with zero behind-camera test observations. Quality is stable by these measures;
the reference COLMAP reconstruction is not ground truth. The machine-readable
summary is `artifacts/sfm_vulkan_performance_20261021/comparison_phase_release_verified_threads4.json`.

## Records

Raw timings and executable copies are under
`artifacts/sfm_vulkan_performance_20261021/`. Run executable copies from the
Release binary directory so their runtime DLLs can be found. The full 739-image
four-worker intermediate run is under `threads4/`; it predates the cache
growth/eviction correction. `final_threads4/` tests that correction but still
overlaps match partials with weak-view extraction. The complete implementation,
including release at that transition, is under `phase_release_verified_threads4/`.
`phase_release_threads4/` is an excluded, intentionally stopped run whose
benchmark executable was still locked during relinking and retained the old
hash. The verified run uses SHA-256
`d767b89a389caab520f59d96692a0dd34e66a9775b1e9d150752317facffd80b`.
All runs preserve command/configuration and executable
hashes in `manifest.json`, stage timings in `vulkan/profile.json`, complete
`vulkan/alignment.log`, device/process memory sampling, ASFM and COLMAP exports.
Generated artifacts are ignored by Git.
