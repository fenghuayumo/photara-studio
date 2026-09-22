# Photara MVS architecture

This document describes `Photara::MVS` in `photara/include/mvs` and
`photara/src/mvs`: PatchMatch depth estimation, depth fusion, Delaunay and
TSDF mesh backends, cleanup, and CLI integration. See
[SfM architecture](SFM_ARCHITECTURE.md) for camera alignment and
[Splat C++ backend](SPLAT_CPP.md) for the default product reconstruction path.

## 1. Role in the product

MVS is not part of the default Splat path.

| Use | Trigger |
|---|---|
| Explicit dense cloud or mesh reconstruction | `--dense` |
| Fixed-pose dense reconstruction from an external dataset | `--splat-dataset --dense` |
| OpenMVS interoperability and algorithm comparison | MVS export and benchmark tools |

Default Splat training initializes Gaussians directly from SfM sparse points;
logs report `patchmatch=false`. MVS depth and meshes are not used as implicit
subject-mask sources.

MVS and Splat mesh extraction share the sparse-block TSDF and Marching Cubes
implementation.

## 2. Stage overview

```text
mvs::densify(scene, options)
├── select_neighbors                         stage: mvs.select_neighbors
├── estimate_depth_maps                      stage: mvs.estimate_depth
│   ├── load images / build pyramids         mvs.load_images / mvs.build_pyramids
│   ├── PatchMatch on CUDA or CPU            depth, normal, confidence
│   ├── geometric consistency                mvs.geometric_consistency
│   └── depth filtering                      mvs.filter_depth
├── fuse_depth_maps                          stage: mvs.fuse
└── reconstruct_mesh (optional)              stage: mvs.mesh
    ├── delaunay_cut                         mvs.mesh_global_cgal
    ├── tsdf                                 mvs.mesh.tsdf
    └── clean_mesh                           mvs.mesh_clean*
```

If CGAL is unavailable, requesting `delaunay_cut` fails explicitly instead of
silently changing mesh backend.

## 3. Scene construction

`mvs::build_mvs_scene` converts an `sfm::Scene` into an `MvsScene`:

1. Keep registered views; fewer than two is an error.
2. Compute working resolution from `resolution_level` and `min_resolution`,
   rescale intrinsics, and retain source calibration in `src_*` fields.
3. Convert tracks to sparse points with observations and sampled source color.
4. Load `--masks` by filename, binarize at `>=128`, and erode by
   `mask_border_px` to suppress grazing fragments near silhouettes. This
   differs from SfM, which currently treats every non-zero value as valid.
5. Undistort working images according to the source camera model.

`mvs::prepare_imported_scene` preserves imported poses and source calibration,
rescales only the working cameras, and clears stale depth, mask, and neighbor
state.

`SubjectBounds` estimates a conservative 3D region from SfM sparse points. In
object mode it constrains TSDF space and identifies intentional crop
boundaries during mesh cleanup. It does not delete sparse points or Gaussians.

## 4. Neighbor selection and depth estimation

`select_neighbors` ranks source views by covisibility and baseline angle using
`max_neighbors`, `min_shared_points`, and `optim_angle_deg`.

`estimate_depth_maps` provides:

- **CUDA/CPU backends.** CUDA processes one reference view at a time. CPU
  processes `patchmatch_concurrent_views` references concurrently and divides
  each view into `patchmatch_tile_rows` row tiles.
- **Coarse-to-fine estimation.** `sub_resolution_levels + 1` pyramid levels
  propagate estimates from coarse to fine.
- **Photometric cost.** NCC plus randomized refinement;
  `ncc_keep_threshold` controls retention.
- **Geometric consistency.** Each iteration snapshots source depth maps before
  propagation to avoid read/write races.
- **Depth filtering.** Reprojected neighbor depths reject insufficient support
  and free-space conflicts; consistent depths may be averaged.

Logs report backend selection and depth-filter considered/kept/rejected
counters.

## 5. Depth fusion

`mvs::fuse_depth_maps`:

1. uses worker-private grids and hash-partitioned merging;
2. chooses layers by weighted median consensus and suppresses surface
   thickness with a tight inlier mean;
3. applies depth, reprojection, normal, and minimum-view gates;
4. uses incidence angle as a soft weight instead of a hard silhouette reject;
5. removes small connected depth-consistent speckles.

`DenseCloud` stores position, normal, weight, contributing views, and
per-view weights. `save_dense_ply` preserves these fields so the Delaunay
graph-cut can be replayed with the same camera order and coordinate system.

## 6. Mesh backends

### 6.1 `delaunay_cut`

`src/mvs/mesh_cgal.cpp`, stage `mvs.mesh_global_cgal`:

1. transform points and cameras into numerically stable local coordinates;
2. filter samples by projected spacing and `mesh_max_points`;
3. construct a Delaunay tetrahedralization;
4. accumulate Jancosek-Pajdla-style visibility terms with adaptive local
   uncertainty;
5. add calibrated free-space support around weak surfaces;
6. graph-cut occupied/free cells and remove unsupported long-edge webbing.

### 6.2 `tsdf`

`src/mvs/mesh_tsdf.cpp`, stages `mvs.mesh.tsdf` and
`mvs.mesh.tsdf.marching_cubes`:

| Option | Default | Meaning |
|---|---|---|
| `mesh_tsdf_voxel_size` | inferred | world-space voxel size, normally `max_depth / 2048` |
| `mesh_tsdf_voxel_scale` | 1 | multiplier for inferred voxel size |
| `mesh_tsdf_truncation_voxels` | 4 | truncation half-width |
| `mesh_tsdf_pixel_step` | 4 | sparse-block allocation step |
| `mesh_tsdf_min_weight` | 0.25 | minimum accumulated extraction weight |
| `mesh_tsdf_support_closing_axes` | 2 | axes required for zero-weight support closure |
| smoothing iterations / lambda / mu | 2 / 0.5 / -0.53 | boundary-locked Taubin smoothing |
| `mesh_tsdf_bounds_padding` | 2 | point-cloud bounds padding |

### 6.3 Cleanup

The TSDF cleanup path removes degenerates and unused vertices, rejects small
components, and closes small holes. The Delaunay path additionally resolves
non-manifold edges and bow-tie vertices, orients components, removes tiny
islands and spikes, applies scale-aware spurious-geometry rejection, closes
small boundary loops, compacts the mesh, and recomputes normals.

Crop boundaries created by valid `SubjectBounds` are treated as intentional
and are not filled as reconstruction holes.

## 7. Quality presets

`mvs::apply_quality_preset` resets the complete pipeline configuration.

| Option | preview | default | high |
|---|---:|---:|---:|
| `resolution_level` | 2 | 1 | 0 |
| `sub_resolution_levels` | 1 | 1 | 1 |
| `estimation_iters` | 3 | 4 | 5 |
| `geometric_iters` | 1 | 2 | 3 |
| `random_iters` | 4 | 6 | 8 |
| `max_neighbors` | 8 | 12 | 16 |
| `min_patch_views` | 2 | 2 | 3 |
| `ncc_keep_threshold` | 0.50 | 0.45 | 0.40 |
| `min_views_fuse` | 2 | 3 | 3 |
| `min_views_filter` | 1 | 1 | 2 |
| `speckle_size` | 24 | 40 | 80 |
| `mesh_method` | delaunay_cut | delaunay_cut | delaunay_cut |

CLI options override the preset. `--dense-resolution-level` can override only
the working resolution after the preset is applied.

## 8. Common CLI options

| Option | Default | Description |
|---|---|---|
| `--dense` | off | Run MVS |
| `--dense-quality` | `default` | `preview`, `default`, or `high` |
| `--dense-resolution-level` | preset | 0 = source resolution, 1 ≈ half |
| `--mesh` | off | Build a mesh |
| `--mesh-method` | `auto` | `tsdf` selects TSDF; other values use Delaunay in MVS |
| `--mesh-max-points` | 2,000,000 | Delaunay sample cap; 0 means unlimited |
| `--mesh-dist-insert-px` | preset | projected-spacing filter |
| `--patchmatch-tile-rows` | 8 | CPU row tile size |
| `--patchmatch-concurrent-views` | 8 | concurrent CPU reference views |
| `--masks` | `auto` | shared mask directory; MVS uses a `>=128` threshold |
| `--mesh-obj` | off | also write ASCII OBJ |

## 9. Exports and diagnostics

- `save_dense_ply` / `load_dense_ply`: weighted dense cloud I/O.
- `save_mesh_ply` / `load_mesh_ply` / `save_mesh_obj`: mesh I/O.
- `encode_mesh` / `decode_mesh`: `.ascan` mesh chunk.
- `save_subject_bounds` / `load_subject_bounds`: bounds text format.
- `save_depth_map` / `load_depth_map`: `.admap` depth cache.
- `--mesh-tsdf-frame-export-dir`: exact depth/intrinsic/pose frames consumed by
  TSDF for backend comparisons.
- `mesh_tsdf_diagnostics_dir`: depth-consistency and observation-weight maps.

## 10. Tests

`photara.mvs.pipeline` covers scene construction, source-camera preservation,
parallel masked fusion, dense PLY round trips, sparse-block TSDF, support
closure, hole triangulation, CGAL max-flow conventions, Delaunay meshing,
topology cleanup, quality presets, `SubjectBounds`, and early failure without
CGAL. `photara.sfm.export_mvs` covers MVSI orientation and visibility data.

## 11. Known limitations

- Delaunay meshing requires CGAL; otherwise use TSDF.
- TSDF output is intended for preview and downstream repair and is not
  guaranteed to be watertight, especially around thin structures and
  occlusion boundaries.
- MVS quality depends strongly on camera poses. Inspect SfM diagnostics before
  tuning dense reconstruction.
- Triangle count and optimization loss are not acceptance criteria; inspect
  thin structures and occlusion regions visually.
