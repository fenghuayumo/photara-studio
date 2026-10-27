# 3DGUT Vulkan performance

Measured on 2026-10-27 with an NVIDIA GeForce RTX 5090 D v2 (24 GB), using
`build/3dgut-vid007/masked/gut-5000_splat.ply`: 500,000 Gaussians, the first
registered panorama camera from `VID_20260911_145523_00_007_dataset`.
These are synchronized median wall times for resident parameters and a fixed
camera, after three warmups and 20 timed samples. The viewport uses five warmups.

| Image | Backend | Training forward | Backward | L1/SSIM + backward | Inference forward |
| --- | --- | ---: | ---: | ---: | ---: |
| 1000 x 500 | CUDA | 9.79 ms | 4.34 ms | 4.44 ms | 3.76 ms |
| 1000 x 500 | Vulkan | 3.66 ms | 4.57 ms | 4.77 ms | 3.62 ms |
| 1920 x 960 | CUDA | 19.71 ms | 10.60 ms | 10.96 ms | 3.91 ms |
| 1920 x 960 | Vulkan | 4.58 ms | 11.16 ms | 12.47 ms | 4.54 ms |

Training forward saves backward state; inference does not. Vulkan training
keeps images and visibility resident without materializing owned attachments.
Backward here returns full parameter gradients; these measurements exclude
Adam updates, densification, data loading, previews and CPU logging, so they
are not end-to-end training throughput. CUDA's existing forward also includes
its allocation/compaction work. Do not infer a complete training speedup from
the forward timings alone.

The initial portable Vulkan backward took 69.69 ms at 1000 x 500. A 32-lane
subgroup computes all bucket responses and compositing prefixes in parallel,
reducing it to 4.57 ms (about 15x faster). Hardware float atomics are used when
available; subgroup CAS and portable replay remain compatibility fallbacks.
Devices with a different subgroup width use portable replay, whose performance
is not established by the optimized RTX measurements.

The viewport and training both use `splat_drender::vulkan::SplatRasterizer` and
the same UT projection, radial ordering and 3D ray response. The viewport's
RGB path uses the specialized RGBA blender. Its measured times are 2.93 ms at
1000 x 500 and 3.70 ms at 1920 x 960. Both RGBA outputs exactly match the
float forward after quantization on this scene (maximum byte difference zero).
These times exclude the rest of the editor frame and frame-cache reuse.

The real-scene CUDA/Vulkan comparison passed at both resolutions. Worst RGB
relative L2 was 9.6e-5 and worst parameter-gradient relative L2 was 7.2e-5.
The four-camera synthetic loss, finite-difference, ADC and training tests also
pass for native float atomics, subgroup CAS and portable replay.

Reproduce with the `photara_splat_vulkan_parity` executable:

```text
--bench-gut-backends MODEL IMAGES OUTPUT SFM MAX_RESOLUTION ITERS
--bench-viewport MODEL IMAGES OUTPUT SFM WIDTH HEIGHT ITERS
```

The first mode writes `gut_backends.csv` and checks real-scene numerical
agreement. The second writes `viewport.csv` and checks RGBA parity.
`SPLAT_DRENDER_GUT_BACKWARD_REFERENCE=1` selects portable replay;
`SPLAT_DRENDER_DISABLE_ATOMIC_BACKWARD=1` selects subgroup CAS on capable devices.
