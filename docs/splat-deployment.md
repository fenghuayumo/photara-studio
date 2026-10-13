# Native Gaussian deployment processing

Studio's **Export Splat** dialog includes **LOD Streaming (Vulkan)**.
Processing runs native C++ code in a background job. There
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

## Build and validation

The existing `splat_drender::vulkan` / TinyTensor Vulkan configuration compiles
`deployment.hlsl` to embedded SPIR-V with DXC. Vulkan is the default for the
export API; a build without it reports an error, with no silent backend switch.
Set options' `vulkan = false` to use the CPU reference path.

Build `photara_splat_deployment_test`, then run CTest with
`-R photara.splat.deployment`. The Vulkan test exercises the real GPU and compares
merge costs against an Eigen reference, reads every LOD chunk, and checks
cancellation/output protection.

Reference: [Streamed SOG specification](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/).
