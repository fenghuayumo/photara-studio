# USDC Gaussian export

Studio's **Export Splat** dialog includes **USDC (OpenUSD 3DGS)**. The public
`save_gaussians` API accepts `GaussianFormat::usdc`, and the CLI detects the
format from `--output scene.usdc`. This is a real binary USD crate file,
starting with `PXR-USDC`.

The single default prim `/Gaussians` is an OpenUSD
`ParticleField3DGaussianSplat`. Export keeps model coordinates (Y up, one model
unit authored as one meter), with no PlayCanvas import rotation. Photogrammetry
scale must be calibrated separately if physical units matter.

| Photara data | USD attribute | Conversion |
|---|---|---|
| Means | `positions` (`point3f[]`) | Full float positions |
| Log scales | `scales` (`float3[]`) | Exponential to linear standard deviations |
| Scalar-first quaternion | `orientations` (`quatf[]`) | Normalize; scalar-first crate tuple |
| Opacity logits | `opacities` (`float[]`) | Stable sigmoid |
| SH degree | `radiance:sphericalHarmonicsDegree` | Integer 0–3 |
| SH `[N,K,3]` | `radiance:sphericalHarmonicsCoefficients` | All coefficients, with `elementSize=K` and vertex interpolation |

`extent` bounds the rotated three-sigma ellipsoids. `primvars:displayColor` and
`primvars:displayOpacity` provide basic colour preview data. A viewer must support
OpenUSD's Gaussian particle-field schema to render the Gaussians; a generic USD
viewer may inspect the data without displaying splats.

Writing and reading use the pinned LightUSD C++ crate implementation linked into
Photara. There is no external converter, Python runtime, or USD CLI invocation.
`PHOTARA_ENABLE_USDC` defaults to `ON` for splat builds; turn it off to skip the
dependency. The dependency fetch includes build sources and license files, but
excludes test fixtures containing symlinks that cannot be extracted normally on
Windows.

Native import handles Photara's single, untransformed `/Gaussians` layout.
General USD composition, other prim paths, transforms, animation, and arbitrary
Gaussian hierarchies are not imported. Import converts linear scales and opacity
back into Photara's parameterization; exact opacity endpoints are clamped to
finite logits. Normal-field training features and derived 3D filters are not
part of the standard 3DGS schema.

Build and run `photara_splat_usdc_test` to check binary headers, SH degrees 0–3,
array values, normalized quaternions, and invalid-model handling. The optional
`photara/tests/verify_usdc_openusd.py` independently opens an exported test scene
using Pixar OpenUSD and validates standard schema attributes and metadata.

Schema reference: [OpenUSD ParticleField3DGaussianSplat](https://openusd.org/release/user_guides/schemas/usdVol/ParticleField3DGaussianSplat.html).
