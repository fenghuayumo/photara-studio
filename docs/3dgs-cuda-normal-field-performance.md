# CUDA normal-field draw-list reuse (2026-11-05)

CUDA 3DGS training now consumes the primary render context after its backward
and uses `Rasterizer::forward_colors` for the learned normal-field render.
The auxiliary pass reuses projection, visible/instance counts, sorted draw
indices, tile ranges and contributor bucket metadata. It refreshes signed
feature colors and backward snapshots without a second projection, sort,
contributor prepass or host count readback.

Alpha, median depth, footprint normals, radii and contribution visibility are
color-independent and share the primary render's owned tensors. A specialized
geometry blend preserves normal snapshots while omitting the repeated
median-depth bisection. Auxiliary backward still differentiates geometry,
opacity and normal features, and its gradients merge before the optimizer.

The API consumes the source's exclusive context, so primary backward must
finish first. The camera and raster settings are inherited from that context.
Replacement geometry buffers and a prior fused structure-Adam update are
rejected. In-place geometry edits between these operations remain forbidden
by the caller contract. The trainer releases the consumed workspace before
densification and optimizer updates. Ordinary RGB steps retain their existing
workspace lifetime; normal-field training remains disabled for 3DGUT.

## Validation

The CUDA frontend regression compares full auxiliary rendering with reuse:

- Bit-exact signed RGB, alpha, median depth, footprint normals, visibility and
  radii for perspective, fisheye, equirectangular and orthographic cameras.
- Geometry, opacity, precomputed-color and normal-feature gradients, with
  component-wise tolerances and a relative L2 error limit of 1e-5. Both fixed
  image gradients and actual normal-field loss gradients are exercised.
- Filtered/unfiltered geometry, 2D compensation, half-pixel sampling, nonzero
  background, compact/uncompacted draw storage, forced compact contributor
  snapshots, RGB-only renders, culled rows and zero visibility.
- Context consumption and rejection before primary backward, with shared
  context ownership, replacement geometry buffers or fused geometry updates.

`photara.splat.cuda_frontend`, `photara.splat.rasterizer` and
`photara.splat.gut` all passed. The rasterizer suite includes normal-field
training with geometry supervision and Mip3D filtering.

## Synthetic measurements

Hardware: NVIDIA GeForce RTX 5090 D v2, driver 621.01, Windows, CUDA 12.8,
Release, SM 120. The fixture uses seed 42, degree-3 SH, a 960x540 perspective
camera, mixed raw parameters, culled rows and a 3D filter.

Each measured step includes primary forward/backward, feature conversion,
auxiliary forward, normal-field loss, auxiliary backward and the feature chain
rule. Primary image gradients are fixed, and the model stays fixed between
steps. It excludes loading, L1/SSIM, gradient merge, Adam, densification and
preview, so these results are not full training-time or dataset-quality claims.

Three rounds alternate A/B order, use five warmups per arm and 120 measured
steps, and synchronize at timing boundaries. The full-render arm releases the
primary context before auxiliary forward, matching the prior trainer lifetime.

| Gaussians | Full auxiliary forward, ms | Reuse, ms | Time reduction |
| ---: | ---: | ---: | ---: |
| 262,144 | 2.367 | 1.955 | 17.4% |
| 1,000,000 | 8.000 | 6.822 | 14.7% |

CUDA default-pool peak usage decreased by 12 MiB at 262,144 rows. At one million
rows it stayed around 1.06 GB, with reuse occasionally 4-6 MiB higher. Allocation
bucket sizes and retaining projection state through the auxiliary pass mean
shared attachments do not establish a universal peak-memory reduction.

```powershell
cmake --build build --config Release --target photara_splat_cuda_frontend_test photara_splat_test photara_splat_gut_test
ctest --test-dir build -C Release -R '^photara.splat.(cuda_frontend|rasterizer|gut)$' --output-on-failure
& build/photara/Release/photara_splat_cuda_frontend_test.exe --normal-reuse
& build/photara/Release/photara_splat_cuda_frontend_test.exe --bench-normal-reuse 262144 120
& build/photara/Release/photara_splat_cuda_frontend_test.exe --bench-normal-reuse 1000000 120
```
