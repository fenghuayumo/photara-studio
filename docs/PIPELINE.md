# Photara pipeline overview: SfM, MVS, and 3DGS

This document maps the current Photara code paths from source images through
Structure from Motion, optional Multi-View Stereo, Gaussian Splatting, mesh
extraction, and texture baking. For algorithm details, see
[SfM architecture](SFM_ARCHITECTURE.md),
[MVS architecture](MVS_ARCHITECTURE.md),
[Dense reconstruction](DENSE_RECONSTRUCTION.md), and
[Splat C++ backend](SPLAT_CPP.md).

## 1. Main pipelines

The `photara` CLI selects a pipeline from the requested inputs and flags. It
does not silently run MVS before Splat training.

```text
1. Default product path without --dense
   images -> SfM -> sparse points/poses -> sparse Splat initialization
          -> 3DGS training -> optional TSDF or PAM mesh -> optional texture

2. Explicit MVS path: --dense without --splat
   images -> SfM -> PatchMatch -> depth fusion -> dense cloud
          -> optional Delaunay or TSDF mesh -> optional texture

3. External dataset path: --splat-dataset
   COLMAP / RealityCapture / OpenMVS -> cameras + sparse points -> 3DGS
   With --dense: fixed imported poses -> MVS -> dense cloud/mesh, no Splat
```

| Path | Trigger | Initialization | Splat training | PatchMatch |
|---|---|---|---|---|
| Default product | `--splat`, `--capture-mode`, or internal SfM + `--splat` | SfM sparse points | yes | no |
| Dense initialization | `--dense --splat` | fused MVS cloud | yes, dynamic densification disabled | yes |
| External point cloud | `--dense --dense-ply` without `--splat-dataset` | supplied PLY with internal SfM poses | yes | no |
| MVS only | `--dense` without `--splat` | n/a | no | yes |
| External dataset | `--splat-dataset` | dataset sparse model or `--dense-ply` | yes | no |
| External dataset + MVS | `--splat-dataset --dense` | imported poses | no | yes |

`--capture-mode object|scene` is a product preset. It enables Splat and mesh;
object mode also estimates `SubjectBounds` from SfM sparse points. Omitting
the option preserves low-level SfM-only behavior.

Two combinations deserve special attention:

- `--splat-dataset --dense` runs MVS with fixed imported poses unless
  `--splat` is also explicitly supplied.
- `--mvs-mesh-only` uses an existing dense cloud or `--mask-mesh`, creates or
  loads a mesh, and renders `<stem>_masks/` plus mesh previews for quality
  inspection.

## 2. Code map

```text
photara/include/   public sfm/mvs/splat/texture/ba/features/io/project APIs
photara/src/sfm/   frontend, mapping, BA orchestration, exports
photara/src/mvs/   PatchMatch, fusion, Delaunay/TSDF mesh, cleanup
photara/src/splat/ dataset loading, CUDA training, densification, mesh export
photara/src/ba/    shared CPU/CUDA LM + Schur + PCG solvers
photara/src/texture/ UVAtlas, projection baking, optional delight
photara/src/project/ .ascan project container
photara/src/tools/ reconstruct CLI and benchmark tools
apps/editor/       Photara Studio, which drives photara.exe
photara/tests/      correctness and regression tests
```

The main CMake targets are `Photara::SfM`, `Photara::MVS`, and
`Photara::Splat`. Splat requires CUDA. CUDA PatchMatch and CGAL meshing remain
optional MVS capabilities.

## 3. Stage map

| Stage | Log stage | Main implementation | Output |
|---|---|---|---|
| Video frame extraction | `extract video frames` | `src/io/video_frames.cpp` | `<video>/images/*.jpg` |
| Features, matching, verification, tracks | `sfm.frontend` | `src/sfm/frontend.cpp`, `src/features/*` | staged checkpoints |
| Mapping | `sfm.incremental_mapping`, `sfm.global_mapping`, `sfm.hierarchical_mapping` | `src/sfm/*` | `sfm::Scene` |
| Bundle adjustment | `ba.cpu`, `ba.cuda` | `src/ba/*`, `src/sfm/bundle.cpp` | optimized scene |
| SfM export | `sfm.export_openmvs` | `src/sfm/export_*` | MVS, PLY, ASFM, ASCAN, COLMAP |
| Subject bounds | `sfm.subject_bounds` | `src/mvs/subject_bounds.cpp` | `*_subject_bounds.txt` |
| MVS scene build | `mvs.load_images`, `mvs.build_scene` | `src/mvs/scene_build.cpp` | working views and sparse points |
| Neighbor selection | `mvs.select_neighbors` | `src/mvs/neighbors.cpp` | source-view lists |
| Depth estimation | `mvs.build_pyramids`, `mvs.estimate_depth` | `patchmatch.cpp`, `patchmatch_cuda.cu` | depth, normal, confidence |
| Depth fusion | `mvs.fuse` | `src/mvs/fuse.cpp` | dense cloud |
| MVS mesh | `mvs.mesh_global_cgal` or `mvs.mesh.tsdf` | `mesh_cgal.cpp`, `mesh_tsdf.cpp` | mesh |
| Mesh cleanup | `mvs.mesh_clean*` | `mesh_clean.cpp` | cleaned mesh |
| 3DGS training | `splat iteration=...` logs | `src/splat/trainer.cpp`, CUDA kernels | Splat model |
| Splat mesh | `mvs.mesh.tsdf` or `splat.pam` | `src/splat/mesh.cpp`, `pam_mesh.cpp` | TSDF/PAM mesh |
| Texture | `texture.*` | `src/texture/*`, `photara_drender` | OBJ/MTL/albedo |

## 4. CLI implication rules

`reconstruct.cpp::parse_cli` applies these rules:

```text
--capture-mode object|scene  -> --splat and, unless explicit, --mesh
--delight                    -> --texture
--texture                    -> --mesh; without --splat this also implies --dense
--mesh without --splat       -> --dense
--mesh-obj                   -> --mesh
--splat-dataset              -> Splat unless --dense requests fixed-pose MVS
--dense-ply                  -> Splat; with --dense, keep internal SfM poses
--mvs-mesh-only              -> requires external dataset plus dense PLY or mesh
--masks auto                 -> sibling masks/ directory; `-` disables masks
--sam-text PROMPT            -> generate/reuse masks before SfM and use them downstream
```

Current constraints:

- `--mesh-method pam` requires Splat training.
- Texture is not yet available in the direct external-Splat path.
- `--ba-backend auto|cpu|cuda` changes the BA solver, not reconstruction
  semantics.
- `--splat` requires CUDA and `PHOTARA_ENABLE_SPLAT=ON`.
- SAM preprocessing requires `PHOTARA_ENABLE_SAM=ON` and a local SAM 3 GGML
  checkpoint. The in-tree ggml runtime uses CUDA when the toolkit is available.

## 5. Artifacts

Artifacts use the stem and parent directory of `--output`.

| File | Condition | Contents |
|---|---|---|
| `*_sfm_diagnostics.csv` | successful internal SfM, non-GUI | per-image alignment diagnostics |
| `*_subject_bounds.txt` | valid object-mode bounds | world-space subject bounds |
| `*.mvs` | MVS export | OpenMVS cameras and sparse points |
| `*.ply` | sparse output | sparse XYZRGB and matching MVS file |
| `*.asfm` | ASFM output | native SfM scene |
| `*.ascan` | ASCAN output | project settings + SfM + optional Splat/mesh |
| `*_dense.ply` | `--dense` | fused cloud with view weights |
| `*_mvs_mesh.ply` | MVS mesh | Delaunay or TSDF mesh |
| `*_splat.ply`, `.sog`, `.spz`, `.glb` | `--splat` | trained Gaussian model |
| `*_splat_mesh.ply` | Splat TSDF mesh | cleaned surface mesh |
| `*_pam_pivot_mesh.ply`, `*_pam_candidates.ply` | PAM | PAM intermediate geometry |
| `*_textured.obj/.mtl/_albedo.png` | `--texture` | textured mesh and albedo |
| depth/normal/alpha PNGs | diagnostics enabled | intermediate visualizations |

GUI mode keeps working copies and logs but skips ordinary sidecar exports and
evaluation PNGs.

## 6. Invariants and boundaries

1. **MVS is not a prerequisite for Splat.** The default path consumes SfM
   cameras, sparse points, source images, and observations. Logs include
   `splat_input=sfm_sparse ... patchmatch=false`.
2. **MVS is an optional diagnostic and compatibility backend.** Its depth and
   mesh do not silently become Splat initialization or subject masks.
3. **External masks are shared with stage-specific semantics.** The
   composable SfM frontend removes zero-mask keypoints before retrieval and
   matching. MVS thresholds masks at 128 and erodes borders. Splat consumes
   soft coverage. SfM warns and continues when a view mask is missing;
   subject-only Splat requires a mask or source alpha for every training view.
   `lightglue_end2end` is not yet mask-aware for final pair-wise matches.
4. **Mesh backend follows the source.** Splat uses TSDF or PAM; MVS uses CGAL
   Delaunay or TSDF.
5. **Texture and Delight consume the final mesh.** They do not feed back into
   Splat or camera alignment.
6. **Splat training requires CUDA.** MVS PatchMatch can fall back to CPU.

## 7. Quick commands

```powershell
# SfM -> sparse-initialized 3DGS
build\photara\Release\photara.exe --images images --output scene.ply --splat

# Masked SfM -> MVS -> mesh
build\photara\Release\photara.exe --images images --masks masks `
  --output scene.ply --dense --mesh

# Prompted SAM 3 masks -> masked SfM -> 3DGS
build\photara\Release\photara.exe --images images --sam-text "the object" `
  --output object.ply --capture-mode object

# Object product preset with texture
build\photara\Release\photara.exe --images images --output object.ply `
  --capture-mode object --texture

# Direct external COLMAP training
build\photara\Release\photara.exe --images images `
  --splat-dataset D:\data\colmap --output scene.ply
```

Regression tests:

```powershell
cmake -S . -B build -DPHOTARA_ENABLE_CUDA=ON -DPHOTARA_BUILD_TESTS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```
