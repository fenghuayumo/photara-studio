# CUDA ADC-IGS parent selection (2026-11-04)

ADC-IGS CUDA replacement/growth sampling no longer downloads the full weight
array for host logarithms, random draws and `std::partial_sort`. The new kernel
filters finite positive weights, generates a per-row Gumbel score, and packs
its ordered float bits plus the row index into a uint64 key. CUB radix-sorts
keys in descending order and a device kernel extracts Int32 indices. Ties
prefer the larger row index, matching the old host pair comparison. Selection
is without replacement. Oversized/replacement priority, growth masks and final
union rules are unchanged. Vulkan keeps its existing device selector; CPU
inputs retain their host implementation.

Only one uint32 eligible count crosses to the host to size the returned tensor.
The sort and indices remain on CUDA. Temporary Tensor allocations use the
existing stream-aware memory pool. The implementation supports the current
stream, orders cross-stream input dependencies, and allocates no persistent
full-model selection cache. Optional IGS profiling waits for sort completion;
normal refinement returns after enqueueing sort/index extraction.

The CUDA generator is a stateless SplitMix64 counter keyed by two consecutive
mt19937 draws. It is repeatable for the same seed/weights and compatible with
the saved host RNG state for continued runs of this implementation. It does
not reproduce the former CPU per-eligible-row RNG trajectory, including later
random split offsets. Checkpoint format is unchanged, but cross-version
training trajectories will change. The weighted Gumbel/Plackett-Luce rule and
bounds on the uniform samples are preserved. Positive subnormal weights are
classified by bits and logged without fast-math flush-to-zero.

## Measurements

NVIDIA GeForce RTX 5090 D v2, CUDA 12.8, Windows Release/SM120. Three rounds
alternate CPU/GPU order. Weights are fixed positive floats; k is 1% or 25% of N.
The baseline implements the former full D2H / host RNG+partial_sort / H2D index
path. Both arms synchronize at measurement boundaries; GPU index bounds and
uniqueness checks happen outside timing. The table reports medians, not full
refinement or training time. Sampling sequences differ between implementations.

| Rows | Selected | Old CPU ms | CUDA ms | CUDA pool peak MiB |
| --- | --- | ---: | ---: | ---: |
| 1,000,000 | 10,000 | 18.865 | 0.218 | 28.0 |
| 1,000,000 | 250,000 | 66.174 | 0.296 | 29.0 |
| 10,000,000 | 100,000 | 193.754 | 3.067 | 288.5 |
| 10,000,000 | 2,500,000 | 975.202 | 3.388 | 298.5 |
| 20,000,000 | 200,000 | 392.646 | 8.300 | 561.0 |
| 20,000,000 | 5,000,000 | 2448.390 | 8.566 | 592.0 |

The 20M / 25% GPU samples were 8.567, 7.977, and 84.384 ms: the outlier is
retained, not discarded. The 10M / 25% samples ranged from 2.595 to 3.685 ms.
Peak bytes are the CUDA default pool used-byte high water for the isolated
benchmark, including the input/output and allocator buckets; they are not total
training VRAM. The algorithm still performs a full GPU radix sort, with O(N)
workspace. It is not a constant-memory or sublinear Top-K implementation.

## Validation and reproduction

`photara.splat.igs_selection` checks independent host Gumbel rankings, repeatable
seeds, different seeds, no duplicates, short/tail batches, k=0/k>N, all-invalid
weights, NaN/infinities/negative/zero/subnormal/extreme positive weights,
weighted single-draw frequencies, dominant weights, non-default streams, and
integrated replacement/oversized/growth budgets. The benchmark validates every
returned index for 1M, 10M and 20M rows. Existing IGS and rasterizer/training
regressions are also exercised. This is selection-level validation, not a
real-dataset quality/convergence comparison.

```powershell
cmake --build build --config Release --target photara_splat_igs_selection_test photara_splat_test photara_studio
ctest --test-dir build -C Release -R '^photara.splat.(igs_selection|rasterizer)$' --output-on-failure
& build/photara/Release/photara_splat_test.exe --igs-only
& build/photara/Release/photara_splat_igs_selection_test.exe --bench 1000000
& build/photara/Release/photara_splat_igs_selection_test.exe --bench 10000000
& build/photara/Release/photara_splat_igs_selection_test.exe --bench 20000000
```

## Real iPhone indoor dataset smoke test

Dataset: `D:/BaiduNetdiskDownload/ÊÒÄÚiphone/images`, existing COLMAP `sparse/0`
(738 views loaded). Both windows warm-start from the existing 994,506-row
`outputs/ÊÒÄÚiphone_20260911-124815/step-000030000.ckpt/splat.ply`, not from scratch
and not from its original optimizer state. The harness creates a new native
checkpoint with fresh optimizer state and the requested iteration counter.
Resolution is native, ADC-IGS/CUDA, capacity 1.5M, RGB L1/SSIM, preview off,
geometry supervision off. Both stage and IGS profiling were enabled. Output
paths are separate under `build/`; source data/models are not modified.

- `igs-real-iphone-800`: 800 steps, 36.450 s optimization, final 990,479 rows. Single training-view evaluation PSNR 17.7511, SSIM 0.925252.
- `igs-real-iphone-growth-800`: 800 steps, 35.212 s optimization, final 1,075,153 rows. Single training-view evaluation PSNR 17.4557, SSIM 0.921754.

The 16001¨C16800 window exercises post-growth replacement. The 6001¨C6800
window exercises both replacements and new growth, with four refinement events.
Profile timings for all events (ms):

| Window | Iteration | Replacement sort | Growth sort | Complete parent selection |
| --- | ---: | ---: | ---: | ---: |
| igs-real-iphone-800 | 16200 | 0.3701 | 0.0000 | 16.3423 |
| igs-real-iphone-800 | 16400 | 0.3050 | 0.0000 | 1.2963 |
| igs-real-iphone-800 | 16600 | 0.2212 | 0.0000 | 0.4360 |
| igs-real-iphone-800 | 16800 | 0.2319 | 0.0000 | 0.5556 |
| igs-real-iphone-growth-800 | 6200 | 0.2777 | 0.4598 | 22.3047 |
| igs-real-iphone-growth-800 | 6400 | 0.2542 | 0.2173 | 1.9267 |
| igs-real-iphone-growth-800 | 6600 | 0.1793 | 0.1653 | 1.5613 |
| igs-real-iphone-growth-800 | 6800 | 0.2093 | 0.2124 | 2.0542 |

The first growth-window replacement had a 20.035 ms preparation/count wait;
this is retained in the profile, not discarded as if sorting alone were the
whole refinement. Data-get wall totals were 26.187 s and 25.762 s respectively,
with zero asynchronous upload failures. This run is chiefly data-loading
limited, so selection microbenchmark ratios are not end-to-end training gains.
Both runs completed, wrote checkpoints/evaluation images, and returned exit
code 0. No old/new real-dataset convergence A/B or held-out quality claim is
made by these 800-step windows; the listed image metrics are one training view.
The harness accepts optional START_ITERATION to exercise the growth phase.
