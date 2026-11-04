# CUDA 3DGS front-end audit (2026-11-04)

This change concerns CUDA 3DGS, not 3DGUT. Vulkan and CUDA point-depth queries
retain their existing paths.

## Retained optimization

`RasterizeOptions::fuse_activation` lets the CUDA rasterizer consume raw log
scales, scalar-first quaternions and opacity logits. Projection activates one
row in registers, including the Mip3D determinant correction, without three
full-model output arrays. Fused projection backward/structure Adam reconstructs
the row before updating parameters, reusing its unfiltered scales for the chain
rule. At one million rows the eliminated arrays total 32,000,000 bytes.

The trainer selects this path only when there is no normal-field or multi-view
gradient merge and structure is not frozen. Standalone callers default to the
materialized path. Explicit fused-forward callers that request standalone
backward materialize the existing activation chain locally in backward.

## Count-readback experiments

The exact visible/instance counts still size the CUB sorts and workspaces.
Two prototypes used pinned asynchronous readback and enqueued the independent
visibility scan before the host waited for count readiness. One used a separate
copy stream with a preprocessing event; the other used the default stream and
a completion event. Both preserved count/image/gradient/Adam parity but were
slower on the tested Windows driver. Neither prototype is retained or enabled.

In the initial 1M-row, 960x540 synthetic test (before the final benchmark adopted
the classic 0.3-pixel covariance dilation), three rounds of 120 measured steps
averaged 2.716 ms for the original path versus 4.629 ms for the default-stream
asynchronous-count prototype. The separate-stream prototype also took about
4.6 ms. These results reject those scheduling implementations on this machine;
they do not establish that removing the count fence is unhelpful in general.
A genuinely device-driven draw pipeline would also need a bounded-capacity sort,
overflow recovery, and consistent forward/backward workspace layouts.

## Reproduction and scope

Hardware: NVIDIA GeForce RTX 5090 D v2 (24 GB), Windows, CUDA 12.8, Release,
SM 120. The fixture uses seed 42, degree-3 SH, perspective projection, classic
0.3-pixel covariance dilation, mixed scales/quaternions/opacities and culled
rows. A/B arms start from identical models and optimizer states. The benchmark
alternates arm order across three rounds, excludes five warmups and synchronizes
only at measurement boundaries. Reported milliseconds cover forward plus fused
SH/structure backward/Adam, using fixed image gradients. They exclude image
loading, L1/SSIM, ADC refinement, preview and evaluation. They are not full
training-time speedups or quality measurements on a real dataset.

```powershell
cmake --build build --config Release --target photara_splat_cuda_frontend_test
ctest --test-dir build -C Release -R '^photara.splat.cuda_frontend$' --output-on-failure
& build/photara/Release/photara_splat_cuda_frontend_test.exe --bench 1000000 120
& build/photara/Release/photara_splat_cuda_frontend_test.exe --bench 262144 120
```

Variant 0 is materialized activation, variant 1 is fused activation. The test
compares RGB, alpha, depth, normals, visibility, radii, instance counts,
standalone gradients and three fused Adam steps, including nonzero moment
state, culled/zero-quaternion rows, filtered/unfiltered models, pinhole,
fisheye and equirectangular cameras, and empty visibility.

## Final measurements

Three-round means after compile-time specialization of raw/activated backward:

| Rows | Original ms | Fused ms | Time reduction | CUDA pool peak reduction |
| --- | ---: | ---: | ---: | ---: |
| 262,144 | 0.962 | 0.954 | 0.89% | 8 MiB |
| 1,000,000 | 2.839 | 2.783 | 1.97% | 32 MiB |
| 3,000,000 | 8.358 | 8.239 | 1.43% | 108 MiB |

The 262,144-row and 1M-row arms use 240 measured steps per round; the 3M-row
arms use 120. Peak memory is CUDA default memory-pool used-byte high water,
not total device VRAM; allocator size buckets make it differ from the exact
32 bytes/row eliminated. Individual 3M-row rounds show a smaller/noisier speed
gain, so the memory reduction is a stronger result than the timing improvement.

Validation: the dedicated CUDA frontend parity test and the complete existing
`photara.splat.rasterizer` regression passed. The editor Release build and
Gaussian transform preview smoke test also passed. No dataset PSNR/SSIM or
full training wall-clock claim is made by this synthetic benchmark.


## Review follow-up

When `PHOTARA_SPLAT_GRAD_DUMP` selects a backward call, raw activation is
materialized into a diagnostic-only snapshot before fused Adam runs. The
compute path still uses raw parameters; no snapshot is allocated on normal
non-diagnostic fused steps. Regression coverage verifies the three activation
files exactly match the pre-update values and that the model actually changes.

Parity checks use measured, quantity-specific absolute/relative tolerances:

| Quantity | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| RGB / normal | 2e-7 | 1e-6 |
| Alpha | 1e-7 | 1e-6 |
| Depth | 1e-8 | 5e-7 |
| SH gradient | 1e-13 | 1e-6 |
| Scale / quaternion gradient | 5e-11 | 1e-5 |
| Opacity gradient | 5e-10 | 1e-4 |
| Mean gradient, per component | 1e-9 | 5e-4 |
| Adam first moments | 2e-9 | 1e-4 |
| Adam second moments | 1e-10 | 1e-6 |

Every gradient tensor also has an independent relative L2 error limit of 2e-6.
The mean-gradient component tolerance accommodates cancellation in geometry
terms; tightening its absolute floor to 1e-10 failed the actual filtered
geometry cases. The norm bound prevents that local allowance from hiding a
systematic gradient bias. Measured RGB/alpha differences were zero, depth was
within 3 ULP, and SH gradient error was about 1e-12. These measurements guide
thresholds but do not require bit identity from unordered GPU reductions.

Parameter increments allow one ULP at the parameter magnitude plus 0.01% of
the update. Quaternion increments additionally allow 1e-9 absolute error; SH
increments allow 0.1% because quantized optimizer state can cross bins. The
old common 0.5% increment tolerance is removed. All eight structure Adam
moment arrays are checked. Each gradient comparison must reject an all-zero
result; each increment comparison must have a signal larger than the rounding
floor and reject a missing update. Explicit comparator guard cases reject
zero gradients, skipped updates, and injected 1% biases. Correctness cases
use larger observable learning rates; benchmark settings are unchanged.

The tightened frontend parity suite passed five consecutive runs. Use
`photara_splat_cuda_frontend_test --audit-errors` to inspect maximum absolute,
relative and ULP errors and the largest per-tensor relative L2 error. Large
relative/ULP errors near a cancellation zero should be interpreted alongside
absolute and norm errors, not in isolation.
