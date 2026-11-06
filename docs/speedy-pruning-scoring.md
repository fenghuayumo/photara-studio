# Speedy-Splat scoring overhead

The scoring pass uses `TrainingDataLoader::get_camera()` instead of `get()`.
It applies the same training camera projection, maximum image size, pixel-center
resize convention, and progressive resolution scale without decoding RGB,
masks, or priors, uploading tensors, or disturbing the image cache/prefetch
request cadence. Scoring still visits the configured training cameras; the
default remains all training views, with pruning opt-in.

CUDA scoring disables `record_backward_state` and enables the separate
`record_contributor_counts` option. The 3DGS forward writes only the last
contributor number per pixel and retains sorted instance indices until scoring
finishes. It omits backward color snapshots, snapshot bucket scans and
compaction, and the extra accumulated-color buffer. Ordinary inference still
releases its instance buffer, and training still records backward state.
Vulkan already used a forward-only scoring path and remains unchanged.

The sensitivity formula, background handling, view count, SH degree, resolution,
prune ratios and schedule are unchanged. This removes setup/storage overhead;
it does not remove the per-view render or sensitivity-kernel cost.

## Validation

On the supplied indoor iPhone model (1,738,589 Gaussians, three registered
cameras, 640x960 renders), a serial run with two warmups and 50 repeats gave:

| View | Forward with backward state | Contributor-only forward | Pixel workspace before | After |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.220 ms | 3.115 ms | 232.94 MB | 2.46 MB |
| 1 | 3.623 ms | 3.447 ms | 271.18 MB | 2.46 MB |
| 2 | 0.651 ms | 0.603 ms | 84.40 MB | 2.46 MB |

MB is decimal; pixel workspace is live logical allocation, not total GPU
memory. Forward colors match exactly in all three views. Forward time improves
by about 3-7%; these measurements do not represent total pruning acceleration.
Removing RGB I/O is checked through cache counters, not assigned a universal
timing gain (cache residency and source image size vary).

### Full image-directory scoring pass

The real indoor iPhone image directory contains 739 images, all registered in
its existing `images.cache/sfm.bin`. An additional controlled pass scored all
739 views against the same 1,738,589-Gaussian model at maximum dimension 960:

| Camera acquisition | Total scoring pass | Image decodes | Uploaded image bytes |
| --- | ---: | ---: | ---: |
| Training image cache `get()` | 36.138 s | 739 | 1,816,166,400 |
| Calibration-only `get_camera()` | 2.748 s | 0 | 0 |

Both passes use the same new contributor-only scoring kernel, fixed model and
camera order. The comparison isolates image acquisition overhead, not the
combined before/after cost of the backward-state optimization. Host cache is
64 MiB, device image cache is disabled, async image upload is disabled, masks
and priors are disabled, and the second pass follows the first in one process.
End-of-pass score download synchronizes GPU work. Accumulated score relative
L1 difference is 4.49114e-7, consistent with floating-point atomic accumulation
variation. Camera projections match the actual decoded views for all images.
There is no model update or pruning in this benchmark; it measures one scoring
sweep, not full training or row-selection time. This configuration is about
13.15x faster; actual training-cache residency changes the benefit.

Regression coverage:

- Pinhole, fisheye, and equirectangular camera-only requests, missing image
  files, multiple progressive scales, and invalid view indices.
- Zero image requests, decodes, uploads and prefetches for camera-only lookup.
- Camera parity against decoded CUDA and Vulkan training-cache views.
- Independent reverse-compositing score oracle, CUDA/Vulkan per-row scores,
  prune-row parity, and small dense / ADC-IGS pruning training runs.

```powershell
ctest --test-dir build -C Release -R 'photara.splat.pruning_(camera|schedule)$' --output-on-failure
build/photara/Release/photara_splat_test.exe --speedy-only
build/photara/Release/photara_splat_test.exe --data-cache-only
build/photara/Release/photara_splat_test.exe --data-cache-vulkan-only
build/photara/Release/photara_splat_test.exe --speedy-native-bench model.ply sfm.bin 50
build/photara/Release/photara_splat_test.exe --speedy-image-pass model.ply sfm.bin images-directory
```
