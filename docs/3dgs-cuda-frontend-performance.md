# CUDA 3DGS front-end audit (2026-11-04)

The initial audit concerned CUDA 3DGS. A subsequent cleanup makes fused
activation the only Photara CUDA render path for both 3DGS and 3DGUT, and uses
raw parameters for CUDA depth/occupancy queries. Vulkan retains its own path.

## Retained optimization

The CUDA rasterizer consumes raw log scales, scalar-first quaternions and
opacity logits directly. Projection activates one
row in registers, including the Mip3D determinant correction, without three
full-model output arrays. Fused projection backward/structure Adam reconstructs
the row before updating parameters, reusing its unfiltered scales for the chain
rule. At one million rows the eliminated arrays total 32,000,000 bytes.

There is no runtime activation switch or alternate preactivated forward in
Photara. Standalone gradient merges and diagnostics materialize activation
only for their chain rule or export, not for projection. The shared helper
also remains necessary for export/visualization. The lower-level renderer's
native activated/covariance input contract remains supported for its consumers.

3DGUT reuses the same row helper for unscented preprocessing, transform
backward, and fused Adam. Blending fetches opacity from the preprocessed
`conic_opacity.w`, including the reference geometry renderer. Its reference
ray-response path loads shape through the shared helper as needed. CUDA depth
and occupancy contexts no longer retain full-model activated arrays.

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

The A/B variants in the table below were measured before cleanup (commit
9bb6390). The current benchmark reports three runs of the single fused path.
The frontend test compares fused Adam with independent standalone gradient,
chain-rule and optimizer operations for both rendering methods, including
nonzero moment state, culled/zero-quaternion rows, filtered/unfiltered models,
pinhole, fisheye and equirectangular cameras, and empty visibility. Dedicated
rasterizer/GUT oracle and finite-difference tests validate rendering and gradients.

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

The current frontend regression compares fused backward/Adam against independent
standalone gradients plus the standalone structure and quantized-SH optimizers.
It covers both 3DGS and 3DGUT. The retired preactivated render path and its A/B
switch have been deleted from Photara; test-only baseline rendering is not
retained in production. Existing CPU-oracle/finite-difference/rasterizer/GUT
parity suites cover render outputs and the complete gradient chain.

Captured fused opacity gradients use per-component absolute/relative limits
5e-10 / 1e-4 plus an independent relative L2 limit of 2e-6. Adam first-moment
limits are 2e-9 / 1e-4; second-moment limits are 1e-10 / 1e-6, for all eight
structure-state arrays. Parameter increments allow one ULP at parameter
magnitude plus 0.01% of the actual change. Quaternion increments additionally
allow 1e-9 absolute error; quantized-SH increments allow 0.1%. No tolerance was
relaxed for the switch to the independent optimizer reference.

Explicit comparator guards reject zero gradients, skipped updates, and injected
1% biases. Every gradient check rejects an all-zero result, and update checks
require a signal above rounding noise. Correctness cases use larger observable
learning rates; the performance benchmark keeps its original settings.
`--audit-errors` reports maxima for the quantities currently compared.

## Single-path cleanup validation

`photara.splat.cuda_frontend`, `photara.splat.rasterizer` and
`photara.splat.gut` passed after unifying the paths. The GUT suite includes its
CUDA/Vulkan/reference parity and geometry tests. The frontend covers filtered
and unfiltered models, both render methods, RGB and geometry, culled rows,
zero quaternions, camera types, standalone optimizer references and diagnostics.

A 1M-row, 960x540 synthetic forward/backward/Adam benchmark (three rounds of
120 steps) measured 3DGS at 2.815/2.836/2.829 ms with a CUDA pool peak of
791,674,880 bytes. 3DGUT measured 3.433/3.439/3.439 ms and 925,892,608 bytes.
These are current-path samples, not a new matched before/after speedup claim.

```powershell
& build/photara/Release/photara_splat_cuda_frontend_test.exe --bench 1000000 120
& build/photara/Release/photara_splat_cuda_frontend_test.exe --bench 1000000 120 3dgut
```
