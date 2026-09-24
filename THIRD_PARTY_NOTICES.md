# Third-party notices

Photara itself is copyright Bingyang Hu and is licensed under Apache-2.0.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

This file lists third-party code, SDKs, and model weights that a Photara
build can pull in. Licenses below are the upstream terms as of this writing.
They are not legal advice. Before distribution, review the license text that
ships with each enabled dependency.

## Restricted and copyleft components

These are the terms that most often change what a binary may do. Everything
in this section is either optional, gated by a CMake option, or a model that
is never bundled.

| Component | License | How it enters a build | Notes |
|---|---|---|---|
| TinyTensor (`photara/third_party/tinytensor`) | GPL-3.0-or-later (LichtFeld Studio) | Linked when `PHOTARA_ENABLE_SPLAT=ON` | Extracted from LichtFeld Studio (`lfs::core`). `core/vram_profiler.*` still carry that SPDX header. OffsetAllocator inside TinyTensor is separately MIT (Sebastian Aaltonen, 2023). |
| SiftGPU | University of North Carolina non-commercial / non-profit license | Default `PHOTARA_ENABLE_SIFTGPU=ON` | GPU SIFT by Changchang Wu. Commercial use needs a separate arrangement with UNC. Disable the option and use CPU VLFeat SIFT (`--extractor sift`) for a non-SiftGPU build. |
| CGAL | GPL-3.0-or-later, or a CGAL commercial license | Delaunay / PAM meshing, and `PHOTARA_ENABLE_MESH_TOOLS` | The MIT `photara_drender` library does not link CGAL. Mesh repair/decimation tools do. |
| GNU MP (GMP) | LGPL-3.0-or-later or GPL-2.0-or-later | Copied beside Windows CGAL mesh tools | Only when those tools are enabled. |
| FreeImage | GPL-2.0, GPL-3.0, or FreeImage Public License (FIPL) | Feature I/O fallback (`find_package(freeimage)`) | JPEG/PNG go through libjpeg and libpng. FreeImage covers the remaining formats. Redistribution must ship the chosen FreeImage license text and the upstream acknowledgement. |
| SuperPoint weights and inference | Magic Leap restrictive license | Optional `--extractor superpoint` | LightGlue itself is Apache-2.0. SuperPoint weights and the SuperPoint inference path follow Magic Leap's terms. |
| Intrinsic / Delight ONNX weights | Academic / non-commercial (verify upstream) | Optional `--delight` | Weights are not in this repository. See [docs/LICENSE-Intrinsic.md](docs/LICENSE-Intrinsic.md). |
| SAM 3 / SAM 2 / EdgeTAM checkpoints | Meta SAM 3 license | Optional SAM masks; editor download | Weights are not bundled. Studio downloads only after the user accepts [Meta's SAM 3 license](https://github.com/facebookresearch/sam3/blob/main/LICENSE). |
| Instant Meshes | BSD-style (Wenzel Jakob et al.) | `PHOTARA_ENABLE_INSTANT_REMESH` via PyNanoInstantMeshes | Linked into mesh tools, not into the core `photara_drender` library. |
| oneTBB (vendored with Instant Meshes) | Apache-2.0 | Same Instant Meshes fetch | |
| FFmpeg | Typically LGPL-2.1+ (build-dependent; GPL if `--enable-gpl`) | External executable on `PATH` | Photara calls FFmpeg; it does not vendor or statically link it. |

A default CUDA Studio build currently turns on SiftGPU and, when Vulkan and
the submodule are present, texture/mesh tools. Splat training additionally
links TinyTensor. Turn those options off if the corresponding terms do not
fit the intended distribution.

## Photara-owned trees in this repository

| Path | License | Copyright |
|---|---|---|
| Photara core, CLI, Photara Studio (`photara/`, `apps/`, `python/` except vendored code) | Apache-2.0 | Bingyang Hu |
| `third_party/splat_drender` | Apache-2.0 (Photara rasterizer written for this project) | Bingyang Hu |
| `third_party/photara_drender` | MIT | photara_drender contributors; see that tree's [LICENSE](third_party/photara_drender/LICENSE) |

## Vendored source

| Component | Path | License | Copyright / origin |
|---|---|---|---|
| VLFeat SIFT | `photara/third_party/vlfeat` | BSD | Andrea Vedaldi and Brian Fulkerson, 2007–2012 |
| cxxopts | `photara/third_party/cxxopts` | MIT | Jarryd Beck, 2014–2022 |
| TinyTensor | `photara/third_party/tinytensor` | GPL-3.0-or-later (see above) | LichtFeld Studio Authors |
| OffsetAllocator | `photara/third_party/tinytensor/internal/offset_allocator.hpp` | MIT | Sebastian Aaltonen, 2023 |
| Open3D marching-cubes tables | `photara/src/mvs/marching_cubes_const.hpp` | MIT | Open3D, 2018–2024 |
| fused-ssim | `photara/src/splat/fused_ssim.cu` | MIT | Rahul Goel, 2024 (ported from fused-ssim) |
| sam3.cpp | `third_party/sam3` | MIT | Pierre-Antoine Bannier, 2025–2026 |
| ggml | `third_party/sam3/ggml` | MIT | The ggml authors, 2023–2026 |
| stb_image / stb_image_write | `third_party/sam3/stb` | Public domain / MIT | Sean Barrett |

## Submodule and fetched native libraries

| Component | License | Used for |
|---|---|---|
| photara_drender | MIT | UV unwrap, projective texture bake, optional mesh tools |
| Microsoft UVAtlas | MIT | Chart parameterization when texture is enabled |
| Instant Meshes (PyNanoInstantMeshes vendored core) | BSD-style | Optional field-aligned remeshing |
| Dear ImGui (docking branch) | MIT | Photara Studio |
| nanobind | BSD-3-Clause | Optional Python module (`PHOTARA_BUILD_PYTHON`) |
| pybind11 | BSD-3-Clause | photara_drender Python bindings (that submodule) |

## Packages resolved by CMake / vcpkg

| Component | Typical SPDX / terms | Used for |
|---|---|---|
| Eigen | MPL-2.0 | Linear algebra |
| Ceres Solver | BSD-3-Clause | Bundle adjustment |
| PoseLib | BSD-3-Clause | Minimal pose solvers |
| gflags | BSD-3-Clause | Pulled in by Ceres/glog on some vcpkg exports |
| Boost | BSL-1.0 | Utilities |
| hnswlib | Apache-2.0 | Vocabulary / retrieval |
| libjpeg / libjpeg-turbo | IJG and/or BSD-3-Clause / zlib | JPEG decode |
| libpng | libpng License | PNG decode |
| libwebp | BSD-3-Clause | WebP I/O on the splat path |
| zlib | Zlib | Compression |
| zstd | BSD-3-Clause or GPL-2.0 (dual) | Compression |
| GLFW | Zlib / libpng | Photara Studio windowing |
| OpenMP | Compiler/runtime license | Parallel CPU paths when available |

## SDKs and toolchains (not redistributed here)

| Component | Terms | Used for |
|---|---|---|
| CUDA Toolkit | NVIDIA CUDA EULA | CUDA BA, CUDA MVS, splat training, TinyTensor |
| Vulkan SDK, SPIR-V tools, DXC | Khronos / LunarG / LLVM (DXC) terms | Studio, texture baking, TinyTensor Vulkan compute |
| ONNX Runtime | MIT | Optional learned features, LightGlue, delight |
| CGAL | GPL-3.0-or-later or commercial | Meshing and optional mesh tools |

## Optional learned models (not shipped)

| Model | Upstream terms | Photara flag / path |
|---|---|---|
| LightGlue | Apache-2.0 | `--matcher lightglue` / `hybrid_lightglue` |
| DISK | Apache-2.0 | `--extractor disk` |
| ALIKED | BSD-3-Clause | `--extractor aliked` |
| SuperPoint | Magic Leap restrictive license | `--extractor superpoint` |
| Intrinsic `stage_0.onnx`–`stage_3.onnx` | Academic / non-commercial; verify [compphoto/Intrinsic](https://github.com/compphoto/Intrinsic) | `--delight` |
| SAM 3 (and related SAM 2 / EdgeTAM) checkpoints | Meta SAM 3 license | `--sam-*` / Studio download |

## Acknowledgements required by upstream

FreeImage requires an acknowledgement when it is used, for example:

> This software uses the FreeImage open source image library.
> See http://freeimage.sourceforge.net for details.

SiftGPU, CGAL, TinyTensor/LichtFeld Studio, SAM 3, SuperPoint, and Intrinsic
each keep their own notices in addition to this file.

Full texts for some nested components live next to those trees:

- [third_party/photara_drender/LICENSE](third_party/photara_drender/LICENSE)
- [third_party/photara_drender/THIRD_PARTY_NOTICES.md](third_party/photara_drender/THIRD_PARTY_NOTICES.md)
- [third_party/photara_drender/third_party/UVAtlas.LICENSE](third_party/photara_drender/third_party/UVAtlas.LICENSE)
- [third_party/sam3/LICENSE](third_party/sam3/LICENSE)
- [third_party/sam3/ggml/LICENSE](third_party/sam3/ggml/LICENSE)
- [docs/LICENSE-Intrinsic.md](docs/LICENSE-Intrinsic.md)
