# Native Gaussian deployment processing

Studio's **Export Splat** dialog includes **LOD Streaming (Vulkan)** and
**Collision Mesh (Vulkan)**. Both run native C++ code in a background job. There
is no Node.js, subprocess, or splat-transform runtime dependency.

Choose a new output folder. Browse selects its parent. Existing destinations
are rejected. Work is staged in a sibling temporary directory and published
only after all outputs succeed. Stop processing cancels between compute
batches and export stages; shutdown requests cancellation too.

## Streamed SOG

`photara::splat::export_streamed_sog` writes the PlayCanvas version-1 streaming
index (`lod-meta.json`) and one unbundled SOG directory per leaf and LOD.
The native SOG encoder produces `meta.json` and lossless WebP textures.

The scene is partitioned by median splits on the longest bounding-box axis.
Splitting stops when both the Gaussian-count and spatial-extent limits are met;
sparse nodes below `chunk_min` are not split solely for extent. Every finest
level Gaussian belongs to exactly one leaf. The lower levels are generated
inside each leaf, preserving the spatial partition. Morton ordering is applied
within each chunk. Bounds enclose rotated three-sigma ellipsoids at every LOD.

LOD reduction uses the existing native kNN and moment-matching simplifier.
Vulkan evaluates the geometric KL and base-colour costs for candidate merges.
The host builds the graph, selects independent merges, matches Gaussian
moments, and encodes SOG. Level 0 is the source; the default four levels retain
approximately 100%, 50%, 25%, and 12.5% per region (minimum one Gaussian).
Rounding and small regions can give equal counts in adjacent levels.

The format intentionally omits `lodErrors`, so readers use their count-based
LOD heuristics. Loading this dataset in a PlayCanvas application enables runtime
streaming; Studio currently generates the dataset but does not render streamed
SOG inside its own viewport.

## Collision

`voxelize_gaussians` bins Gaussian ellipsoids into 8-voxel spatial blocks.
Vulkan evaluates voxel centers against only the candidate ellipsoids in that
block, including rotation, anisotropic scales, and opacity. Contributions use
`1 - product(1 - alpha)` with a three-sigma cutoff and minimum contribution of
1/255. This is a Gaussian density occupancy approximation, not a reconstruction
from rendered camera depth.
Candidate buffers remain resident on the GPU while voxel batches run.

The outputs are:

- `scene.voxel.json`: voxel bounds, resolution, and octree statistics;
- `scene.voxel.bin`: little-endian Laine–Karras nodes and 4×4×4 leaf masks;
- `scene.collision.glb`: by default, the binary voxel field's Marching Cubes
  surface at the 0.5 isovalue, matching splat-transform's default smooth mode.
  The optional voxel-faces mode emits the exact exposed voxel faces instead.
  Coplanar merging may produce a different triangle count from splat-transform,
  but does not change the extracted surface.

The default coordinate conversion rotates 180 degrees about Z (flips X and Y),
matching the PlayCanvas splat import transform. Both collision outputs use this
frame and the voxel header is version 1.1. Disable **PlayCanvas coordinates**
for Photara's model frame; the header then uses version 1.0. Ground fill assumes
Y up in the chosen output frame.

Optional host fills close enclosed solid-object cavities by boundary flood fill,
or fill below the first occupied voxel in each XZ column. Cavity fill is not the
reference tool's room-navigation exterior-fill mode. Capsule carving and
connected-cluster filtering are not implemented in this version. Smooth
collision meshes are extracted from the same occupancy grid used for the voxel
octree.

The current compute workspace is a bounded dense grid (default 16 million
voxels); the exported octree is sparse. Oversized grids or candidate lists fail
with a request to increase voxel size. This is not yet an out-of-core voxelizer
for city-scale scenes. WebP encoding and octree/mesh assembly run on the CPU.

## Build and validation

The existing `splat_drender::vulkan` / TinyTensor Vulkan configuration compiles
`deployment.hlsl` to embedded SPIR-V with DXC. Vulkan is the default for both
export APIs; a build without it reports an error, with no silent backend switch.
Set options' `vulkan = false` to use the CPU reference path.

Build `photara_splat_deployment_test`, then run CTest with
`-R photara.splat.deployment`. The Vulkan test exercises the real GPU and compares
occupancy against CPU and analytic rotated-Gaussian results, decodes the emitted
octree, reads every LOD chunk, and checks cancellation/output protection.

References: [Streamed SOG specification](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/),
[voxel format specification](https://developer.playcanvas.com/user-manual/splat-transform/voxel-format/),
and [collision generation](https://developer.playcanvas.com/user-manual/splat-transform/collision/).
