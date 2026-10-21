# Vulkan feature memory optimization — 2026-10-21

This change prioritizes peak memory. It does not establish a Vulkan speedup.
The [CUDA/Vulkan baseline](SFM_CUDA_VULKAN_IPHONE_COMPARISON.md) preserves the
earlier measurements on all 739 indoor images.
The [subsequent matcher allocation optimization](SFM_VULKAN_PERFORMANCE_OPTIMIZATION.md)
retains these pool limits and records its performance and memory checks separately.

## Implementation

- SIFT keeps three Gaussian and three DoG levels, one keypoint image and one
  gradient image. Compact oriented keypoints are stored on the host. Gaussian
  levels are rebuilt for descriptors, using the existing shaders and selection
  order.
- Extractor clones share a serialized GPU workspace and reuse allocations
  across images. CPU image loading/decoding remains parallel. Matching waits
  for extraction and releases its workspace before growing its own buffers.
- Matcher clones share one descriptor mirror/cache and mutex. The frontend
  uses the existing batch coordinator instead of separate single-pair calls.
- Idle feature-buffer pools retain at most 256 MiB device memory and 128 MiB
  host-visible memory, with at most two free buffers per allocation-size bucket.
  Large freed allocations return to the driver. Staging/readback allocations
  prefer non-device-local host-visible memory on discrete GPUs.
- These pool limits bound idle memory, not the entire reconstruction. Active
  images, match partials and solver buffers can exceed them. Oversized SIFT
  work is allowed alone; this is not a guarantee for arbitrary image sizes or
  low-memory GPUs. The defaults and CPU geometry selection remain unchanged.

## Validation and records

- Release build and `photara.features.vulkan` CTest passed, including concurrent
  extractor/matcher clones, exact CPU-oracle matching, workspace release/reuse
  and bounded buffer-pool retention.
- Six real images sampled across this dataset produce byte-identical keypoint,
  float-descriptor and quantized-descriptor hashes before/after the rolling
  workspace change. Snapshots are preserved under
  `artifacts/sfm_vulkan_memory_20261021/sift_*snapshot.txt`.
- The optimized four-worker full run registered 739/739 images, with 670,362
  points and 3,157,222 observations. Own reprojection mean/RMS are
  0.581167/0.740875 px, versus baseline Vulkan 0.581264/0.741000 px.
  Similar aggregate errors alone do not prove pose accuracy; the common
  held-out reference evaluation is recorded alongside the reconstruction.

Each run directory contains `manifest.json`, input hashes, `vulkan/alignment.log`,
`vulkan/profile.json`, `vulkan/resources.jsonl`, the ASFM reconstruction and
COLMAP text export. The manifest preserves commands, executable hashes, source
status, GPU state, process wall time and process peak working set. Stage timings
are inclusive and can overlap; do not sum them.

The generated artifacts are ignored by Git; this document preserves the
interpreted results. The experiment root is
`artifacts/sfm_vulkan_memory_20261021/`, with `rolling_threads4/` and
`default_threads/` as separate configurations. The `threads4/` directory contains
an intentionally stopped preliminary pool-only implementation and is excluded
from the final comparison.

## Recorded results

| Measurement | Vulkan baseline, 4 workers | Optimized Vulkan, 4 workers |
|---|---:|---:|
| Complete reconstruction, excluding exports | 425.810 s | 467.728 s |
| Process wall time including exports | 435.305 s | 475.810 s |
| Extraction | 29.359 s | 46.986 s |
| Primary matching | 29.44 s | 67.25 s |
| Process peak working set | 11.60 GiB | 4.90 GiB |
| Sampled device-wide VRAM peak | 20.11 GiB | 11.40 GiB |
| Registered images | 739/739 | 739/739 |
| Own reprojection mean | 0.581264 px | 0.581167 px |
| Shared held-out reprojection mean | 1.660667 px | 1.652723 px |
| Shared held-out reprojection P95 | 4.45445 px | 4.40792 px |

Memory fell substantially in these observations, while quality was stable on
the same 2,007 held-out reference tracks / 10,772 observations covering 738
reference views. No observations in that evaluation were behind cameras.
The held-out reference model itself scores 1.279982 px mean; it is not truth.
Six-image exact descriptor checks supplement this full-pipeline quality check.
The timing increase is recorded explicitly; this change does not resolve the
Vulkan performance gap. Backend comparisons remain single-run measurements.

The production automatic-thread run failed after extraction (34.086 s),
primary matching (72.91 s) and Vulkan global positioning (45.968 s). At bundle
preparation it logged `NVVM compilation failed: 3`, selected CPU with reason
`gpu_unavailable_or_not_built`, then terminated with `bad allocation` (exit 1,
232.866 s process wall time). No final model was exported, and it is excluded
from successful quality/performance comparisons. Its sampled device-wide peak
was 10.75 GiB and process peak working set 3.66 GiB, both only up to failure.
The failure is preserved under `default_threads/`; default-thread end-to-end
success has **not** been established by this optimization.

After this failure, Windows reported 61,573,791,744 committed bytes against a
62,432,591,872-byte limit (98%). A resident Blender process had about 17.9 GiB
private memory. This is evidence of system commit pressure, not proof of the
allocation failure's precise cause. Pausing rendering does not release scene
memory. Other applications were left running.

Machine-readable comparison and quality outputs are in
`artifacts/sfm_vulkan_memory_20261021/comparison.json` and
`rolling_threads4/vulkan_vs_reference/metrics.json` respectively.

## Measurement limitations

The four-worker optimized run started with 93% background GPU utilization and
about 5 GiB already resident, versus about 1.4 GiB and 2% utilization for the
baseline. Its reconstruction took 467.728 s, extraction 46.986 s and primary
matching 67.25 s. These are actual run records, not a controlled speedup test.
The default-worker run uses the production CLI, so its total time cannot be
attributed to the memory change when compared with a four-worker baseline.

Device-wide sampled memory includes other applications and is not exclusive
process VRAM. Windows process dedicated/shared GPU counters were started after
extraction; they cannot establish the whole-run process VRAM peak. The attempted
automatic monitor failed under Windows PowerShell script execution policy; its
failure log is preserved. No global execution policy was changed.

The existing COLMAP model is a reference, not ground truth. Both full alignment
and the fixed-seed held-out evaluation are needed to assess quality. No dense
reconstruction or splat training is included.
