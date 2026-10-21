# USDC Gaussian export

Studio's **Export Splat** dialog includes **USDC (OpenUSD 3DGS)**. The public
`save_gaussians` API accepts `GaussianFormat::usdc`, and the CLI detects the
format from `--output scene.usdc`. This is a real binary USD crate file,
starting with `PXR-USDC`.

The single default prim `/Gaussians` is an OpenUSD
`ParticleField3DGaussianSplat`. Export converts Photara's RDF frame (-Y up) to
right-handed Y-up coordinates with `(x, -y, -z)`. The rotation, SH coefficients,
and three-sigma bounds use the same frame. This is a 180-degree rotation about
X, distinct from the PlayCanvas collision-export rotation about Z. One model
unit is authored as one meter; photogrammetry scale must be calibrated separately
if physical units matter.

| Photara data | USD attribute | Conversion |
|---|---|---|
| Means | `positions` (`point3f[]`) | Float positions, `(x,-y,-z)` |
| Log scales | `scales` (`float3[]`) | Exponential to linear standard deviations |
| Scalar-first quaternion | `orientations` (`quatf[]`) | Normalize; `(w,x,-y,-z)` scalar-first crate tuple |
| Opacity logits | `opacities` (`float[]`) | Stable sigmoid |
| SH degree | `radiance:sphericalHarmonicsDegree` | Integer 0–3 |
| SH `[N,K,3]` | `radiance:sphericalHarmonicsCoefficients` | Rotate all bands; `elementSize=K` and vertex interpolation |

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

Native import handles the single, untransformed, Y-up `/Gaussians` layout and
converts positions, orientations, and SH back to RDF. New exports carry the
`coordinates=RUB-v1` marker in the layer comment. Old Photara exports with the
exact original `Photara native C++ 3D Gaussian export` comment are recognized
as legacy RDF data and imported without a second rotation. Z-up files are
rejected rather than silently assigned the wrong frame.
General USD composition, other prim paths, transforms, animation, and arbitrary
Gaussian hierarchies are not imported. Import converts linear scales and opacity
back into Photara's parameterization; exact opacity endpoints are clamped to
finite logits. Normal-field training features and derived 3D filters are not
part of the standard 3DGS schema.

Build and run `photara_splat_usdc_test` to check binary headers, SH degrees 0–3,
array values, normalized quaternions, and invalid-model handling. The optional
`photara/tests/verify_usdc_openusd.py` independently opens an exported test scene
using Pixar OpenUSD and validates standard schema attributes, coordinate signs,
rotated ellipsoid bounds, external Y-up import, and legacy Photara import.

Schema reference: [OpenUSD ParticleField3DGaussianSplat](https://openusd.org/release/user_guides/schemas/usdVol/ParticleField3DGaussianSplat.html).
