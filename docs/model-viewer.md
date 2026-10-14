# External model preview

Drop one model file onto the 3D viewport. Studio switches to the appropriate
view and frames the model automatically. No photos, reconstruction or training
are required; this also works in a new session without a project directory.

- Gaussian models: PLY, SOG, SPZ, GLB and USDC (requires USDC support in the build).
- Triangle meshes: PLY and OBJ. OBJ geometry, normals and UVs are read; a
  sibling `<name>_albedo.png` can be previewed. General MTL materials and mesh
  GLB/STL/FBX import are not supported yet.

PLY routing inspects the header: standard Gaussian parameters select 3DGS,
and a nonempty face element selects the mesh viewer. Plain point-cloud PLY
is not a mesh and is rejected by this model import entry point.

Imported Gaussians use the existing renderer and editing tools (switch to the
Gaussian/ring editing view), including edit history and export. Source model
files are not overwritten by dropping them. Mesh geometry can use existing
mesh export and can be stored in the project; imported OBJ materials are not
packaged when saving the project.

Drop a single model at a time. Import replaces the displayed scene; it does
not append or merge models. Wait for active jobs/loads/exports/simplification
to finish. If Gaussian edits are pending, export the result and start a new
project before importing another model. Multi-model scene management and
Gaussian merging are separate future features.

## Mesh load performance

Mesh PLY is parsed in one pass without an intermediate dense point cloud or
per-property temporary arrays. OBJ shares matching position/UV/normal tuples;
UV seams, hard normal edges and flat shading for faces without normals are
preserved. GPU vertex packing, optional texture decoding and mesh subject-box
fitting run on the loader thread. Switching from an existing Gaussian scene
keeps its cached subject box. Wireframe indices are built only when wireframe
is first enabled, and the mesh/GPU buffers remain cached across view switches.

`photara_mesh_load_test --benchmark 1000000` measures PLY parsing only, excluding
preview preparation and GPU upload. A same-machine Release comparison of a
synthetic binary PLY with 1,000,002 vertices and 1,000,000 triangles gave a
three-run median of 1,193 ms before these changes and 198 ms afterward (about
6 times faster). This is not an end-to-end viewport timing; OBJ files and large
textures have different costs.

Regression checks: `photara.mesh.load`, `photara.studio.mesh_preview`, and
`photara.studio.mesh_preview_vulkan`. The Vulkan check uses an invisible window
and exercises solid/textured rendering, lazy wireframe creation, mesh
replacement and switching back to solid rendering.
