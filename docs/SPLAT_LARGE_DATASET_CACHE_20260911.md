# Large-dataset splat cache optimization (2026-09-11)

This change keeps the packed-image cache useful on datasets whose decoded views
exceed the old 512 MiB CUDA budget, without allowing that cache to starve
Gaussian/optimizer state or rasterization scratch.

## What changed

- Added adaptive host and CUDA packed-view budgets.
- The loader now computes the exact packed size of the active-resolution dataset
  (RGBA plus optional depth/normal) instead of guessing from the view count.
- Host caching can cover the full packed dataset when system memory allows it.
- CUDA caching is bounded by:
  - total VRAM,
  - free VRAM,
  - projected Gaussian/optimizer state,
  - a smaller 1/8-of-VRAM share for large high-resolution datasets.
- Before a refinement that may change topology, the trainer checks free VRAM.
  If headroom is low, it drops reconstructable packed CUDA views, trims the
  CUDA memory pool, and disables the device cache rather than risking the
  non-reconstructable training state.
- Added `--splat-cache-auto` and `--splat-prefetch-views`.

Adaptive mode is enabled by default. Setting either explicit cache budget to
zero still disables that cache. Pass `--splat-cache-auto=false` when an exact
fixed budget is required for an experiment.

## Office dataset validation

Dataset:

```text
D:/Models/标准数据集-办公室/images
```

- 900 source JPEGs, 898 registered views
- Training resolution: 1080×1920
- Packed dataset: 7,448,371,200 bytes (~6.94 GiB)
- Strategy: `adc_igs`
- Gaussian cap: 500,000
- Prefetch views: 4

The 3,000-step comparison used the same cached working SfM and the
same seed/configuration. The warmup window starts at iteration 1,000.

| Configuration | Host cache | CUDA cache | Steps/s | GPU mean | Peak VRAM | PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Fixed old defaults | 6 GiB | 512 MiB | 45.70 | 25.84% | 4,824 MiB | 21.9810 dB |
| Adaptive safe budget | 6.94 GiB | 3.0 GiB | **180.07** | 73.72% | **13,873 MiB** | 21.9803 dB |

The safe adaptive run is **3.94× faster** on the same 3,000-step window while
using approximately 9.7 GiB more peak VRAM but remaining well below the 24 GiB
GPU limit. PSNR changed by only +0.001 dB on this fixed-view regression metric.

A longer 6,000-step adaptive run completed without an OOM:

| Configuration | Steps/s | GPU mean | Peak VRAM | PSNR |
|---|---:|---:|---:|---:|
| Adaptive safe budget | 177.14 | 82.54% | 13,963 MiB | 23.2496 dB |

After the operating-system file cache had seen the JPEG set once, a repeat
3,000-step adaptive run reached 208.90 steps/s, 91.55% mean GPU utilization,
and the same approximately 13.9 GiB peak VRAM. The conservative table above
uses the first adaptive run rather than this best repeat.

Profiler behavior changed as expected:

| Configuration | Typical `data_load_ms` | Typical CUDA timeline |
|---|---:|---:|
| Fixed 512 MiB | 10–14 ms/step | 15–18 ms/step |
| Adaptive safe | 0.45–0.48 ms/step | ~4.7 ms/step |

The first adaptive prototype allowed a 6.4 GiB CUDA cache and reached 23.7 GiB
peak VRAM. The live low-VRAM guard released it successfully, but the final
policy deliberately caps this high-resolution dataset at one eighth of VRAM
(~3 GiB), reducing peak VRAM to around 14 GiB with essentially the same
throughput.

Artifacts:

- [Fixed-default baseline summary](../artifacts/splat_perf_office_20260911/fixed_old_3k/summary.json)
- [Adaptive 3k summary](../artifacts/splat_perf_office_20260911/adaptive_safe_3k/summary.json)
- [Adaptive 6k summary](../artifacts/splat_perf_office_20260911/adaptive_safe_6k/summary.json)
- [Comparison plot](../artifacts/splat_perf_office_20260911/office_cache_comparison.png)

## Usage

Adaptive mode is already the default:

```powershell
--splat-cache-auto=true
--splat-prefetch-views 4
```

Use fixed budgets for A/B performance experiments:

```powershell
--splat-cache-auto=false `
--splat-view-cache-mb 6144 `
--splat-device-cache-mb 512
```

Disable only the CUDA cache:

```powershell
--splat-device-cache-mb 0
```

The adaptive host limit uses available system memory while retaining at least a
quarter of it, and has a 16 GiB hard cap. The CUDA budget is computed after
projecting Gaussian/optimizer storage and is reduced automatically for large,
high-resolution datasets.
