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

Dropping several Gaussian files at once (or dropping more onto an existing
Gaussian session) adds every model to the Scene panel's **Gaussian Models**
list. Meshes stay single-file: one mesh per drop, never mixed with other
inputs. Wait for active jobs/loads/exports/simplification to finish. If
Gaussian edits are pending, export the result before changing the model set.

## Multi-model management and merging

Every imported Gaussian model gets a row in the Scene panel:

- The eye toggle hides a model in the composed preview without unloading it.
- `X` removes a model from the session. Removing every model falls back to the
  project's trained working copy, if one exists.
- Offset XYZ and a uniform scale place a model before merging. Scale composes
  in log space, so Gaussian anisotropy is preserved; placement is applied on
  top of the file's own coordinates. The preview reloads when a drag ends.

The viewport always renders the **composition** of all visible models: files
are loaded, transformed, concatenated, and uploaded as one Gaussian cloud
(this also drives the Gaussian edit tools and splat/ring rendering). The
source models stay cached on the CPU. Moving the controls does not reload
files or re-upload the composition; releasing a drag commits one background
rebuild using those cached sources.

Saving a project stores the ordered model list, names, visibility, placement,
and embedded snapshots of every source (including hidden sources). Reopening
does not depend on the original files still being present, and hidden models
can be shown and merged again. Source snapshots and the composed Gaussian
chunk increase project size. Screen-space edits operate on the composed cloud;
when saving those edits, the project restores them as one editable embedded
model named **Edited composition**, rather than reverting to the original
source list. Old project settings remain readable.

**Merge Models...** writes every visible model (with its placement) into one
new splat file, then loads that file as the session's single model. SH bands
are zero-padded to the widest input degree, and optional fields
(`normal_features`, Mip-Splatting `filter_3d`) are kept only when every input
carries them. Merging runs on a background thread with progress in the Scene
panel; the merged model loads automatically once the session is idle (if you
were mid-edit, finish or export the edits first). `merge_gaussians` and
`transform_gaussians` in `photara::splat` are the library primitives; the
`photara.splat.merge` test covers placement math, SH padding, ordering, and a
merged-file round trip.

Export always matches the composed preview: a multi-model session exports the
merged composition (or the edited composition when edits are pending), never
just the first file.

`photara.studio.gaussian_session` covers project reopening with deleted source
files, hidden models, source-cache reuse, merge reload, and removing an open
model row through actual ImGui mouse input. The Vulkan variant also checks
that live drags do not re-upload the cloud, committed changes do update it,
saved edits survive reopening, and exports ignore stale working files.

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
