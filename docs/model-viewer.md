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
