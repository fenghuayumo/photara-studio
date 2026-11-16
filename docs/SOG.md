# SOG and SOG-XT

Gaussian export, container compatibility, optional compaction and encoding-aware
adaptation in Photara. For platform build commands and general reconstruction
usage, see [Build and usage](BUILD_AND_USAGE.md).

Plain SOG-XT export preserves the Gaussian count and does not run POPSpa or
encoding-aware adaptation. Both are separate opt-in operations.

- [Export and import](#export-and-import)
- [Encoding and progress](#encoding-and-progress)
- [Format and compatibility](#format-and-compatibility)
- [POPSpa compaction](#popspa-compaction)
- [Encoding-aware adaptation](#encoding-aware-adaptation)
- [SH clustering and fixed-model evaluation](#sh-clustering-and-fixed-model-evaluation)
- [SOG and SOG-XT comparison](#sog-and-sog-xt-comparison)
- [Validation](#validation)

## Export and import

Select **SOG-XT** in Studio's **Export Splat** panel to write a `.sog-xt`
file. The file can be reopened or dropped into Studio like `.sog` and `.spz`.
Studio also exports PLY, SOG, SPZ, GLB and USDC in a background task with a
progress display. PLY reports processed Gaussians, SOG reports encoded attribute
images, SPZ reports compressed streams, and disk writes report completed bytes.
Opaque preparation/encoding/copy stages show an activity animation rather than
a fixed percentage. Successful completion remains visible at 100%. Closing the dialog keeps
the export running. The C++ `save_gaussians_with_progress` API reports the same
stage completion events. Existing `save_gaussians` callers remain supported.

## Encoding and progress

SOG-XT export preserves the original quantization, layout search rounds and
lossless WebP settings (`quality=100`, `method=6`, `exact=1`). The C++
`save_sog_xt` API accepts an optional stage progress callback through
`SogXtExportOptions`.
CPU layout, box filtering and PRAS use OpenMP when available. At most two WebP
images encode concurrently. Export logs report stage and per-image times
(`sog_xt_export`); WebP work overlaps SH clustering, so those durations are not
additive. Progress reports stage completion rather than an estimated finish time.

Export runs in the background. The control API accepts `format: "sog-xt"`;
when it returns `completed: false`, poll `studio.job.status` and inspect
`splat_export.running` and `splat_export.error` before using the output.
The reconstruction CLI recognizes `.sog-xt` output paths. The C++ API supports
`GaussianFormat::sog_xt` and extension-based detection; Python exposes
`GaussianFormat.SOG_XT` when the Python module is built.

## Format and compatibility

This is Photara's native implementation of the image-coding ideas in
[KISS-GS](https://arxiv.org/abs/2608.26948), not the complete KISS-GS training
pipeline. It keeps the Gaussian count, performs coarse-to-fine local assignment
sorting of both position byte planes and SH codebooks, selects equivalent
rotation/scale representatives with PRAS, and quantizes the attributes before
lossless WebP encoding. Non-DC SH clustering uses the existing Vulkan backend;
non-Vulkan builds have a bounded-memory hierarchical two-means CPU path.
0-3 SH degrees, image-grid padding, and Mip2D metadata are supported. Bake
Mip3D before exporting, as for SOG. Loaded attributes use the existing renderer;
the smaller file does not imply smaller runtime GPU storage.

The `.sog-xt` container is a ZIP with `container_meta.yaml` and WebP attribute
images. Version 2 exports use the paper reference decoder's global position
range, scalar opacity range, and channel-major SH range ordering. Optional
native fields preserve SH degree, active count, and Mip2D mode; reference
decoders can ignore them. The reader accepts both JSON and ordinary block YAML,
old native v1 files, and unmarked reference-style WebP directories (including a
direct `container_meta.yaml` path). Missing degree defaults to 3, missing count
is inferred from the active mask, and missing quantization levels use the
reference's observed-image-range normalization. Positions and SH remain in
Photara's PLY coordinate frame. PNG/JPEG-XL containers are not supported.

The splat build requires `yaml-cpp` (`libyaml-cpp-dev` on Debian/Ubuntu, or the
`yaml-cpp` vcpkg port). An independent NumPy decoder using the paper's reference
layout cross-checks exported degree-3 files against the C++ decoder. This tests
the reference conventions, not the authors' unreleased encoder. The CPU sorter
and hierarchical clustering remain independent implementations. POPSpa pruning
and encoding-aware fine-tuning are not performed by this export.

## POPSpa compaction

POPSpa compaction is a separate opt-in operation. Studio exposes a `POPSpa`
checkbox in training settings and in the model simplification panel. The latter
requires the project's registered SfM reconstruction and its source images;
with POPSpa disabled the existing training-free NanoGS merge remains available.
CLI training accepts `--splat-popspa=true --splat-popspa-keep-ratio 0.2`.
The default retains 20% of the trained input count, rounded to a whole primitive.
Speedy-Splat remains an independent training-time option.

The native implementation builds on KISS-GS Section 3.2: GaussianPOP RGB deletion
scores across training cameras, first pruning, 5,000 optimize-sparsify steps
with GaussianSpa auxiliary opacity/dual variables and effective-rank shape
regularization, a new scoring/pruning stage, and 5,000 fine-tuning steps. Held-out
cameras are excluded from both scoring and optimization. Scoring loads camera
metadata only. The first pass keeps up to twice the final target; this budget
split is an implementation choice because the paper does not specify it.
The GaussianSpa penalty is 0.0005 and its projection/penalty cadence is 50 steps;
projection keeps the largest auxiliary magnitudes with deterministic tie breaks.
Each pruning stage now uses four partial removals with fresh scores between
them, addressing GaussianPOP's batch-removal approximation. Use
`--splat-popspa-prune-cycles 1` for a single removal per stage. The existing
normalized minimum-eigenvalue shape term is retained; it differs from the
paper's physical minimum-scale term and remains an independent choice.
Scale/rotation learning rates keep their original multiplier of 1. The optional
`--splat-popspa-geometry-lr-scale` can reduce them, but the indoor iPhone sweep
did not support changing the default to 0.2: some close-ups improved while
aggregate quality declined.
Final fine-tuning drops the shape prior,
continues the terminal position rate, and scales the other Gaussian learning
rates by 0.2 (`--splat-popspa-finetune-lr-scale`). Compaction honors the selected
bilateral-grid and PPISP settings; these remain training-only corrections and
are excluded from the exported canonical RGB and SOG-XT decoder.
This is an independent implementation, not a claim to reproduce the authors'
rate-distortion results. The standalone C++ entry point is
`Trainer::compact_popspa(model, scene, progress)`.

Compaction does not densify or initialize from the point cloud. It uses the
existing RGB/SSIM optimizer with all SH bands active, no progressive resolution,
and a reduced position learning rate. The reconstruction checkpoint remains
intact; interrupted compaction restarts from that checkpoint rather than
resuming partially saved ADMM state. CLI stage lengths can be overridden with
`--splat-popspa-refine-steps` and `--splat-popspa-finetune-steps` for diagnostics.

## Encoding-aware adaptation

SOG-XT encoding-aware adaptation is a separate CUDA/Vulkan opt-in stage. For a
`.sog-xt` output, pass `--splat-sog-xt-finetune=true`; select the device with
`--backend=cuda` or `--backend=vulkan`; the default is 4,000
optimizer steps with eight views accumulated per step. Override with
`--splat-sog-xt-finetune-steps` and `--splat-sog-xt-batch`. It preserves the
selected PPISP/bilateral-grid settings, excludes evaluation-split cameras,
freezes the image order and SH assignments, renders the quantized model, and
propagates STE gradients through the codec. PRAS runs at initialization and at
60%/80%, resetting covariance moments and updating only covariance ranges.
Export writes the trained assets directly without a second clustering pass.
`Trainer::finetune_sog_xt(model, scene, output, progress)` exposes the same
operation for an existing model; Python exposes `finetune_sog_xt` as well.

The native adaptation additionally uses a canonical input-render preservation
loss (weight 2) and conservative Gaussian learning-rate multiplier 0.02; these
are quality-oriented implementation choices, not the paper's exact training
recipe. CLI controls are `--splat-sog-xt-reference-weight` and
`--splat-sog-xt-lr-scale`. Twelve fixed **training** cameras select the best
canonical-MSE snapshot every 500 steps, including the unadapted baseline.
Held-out RGB does not participate in optimization or snapshot selection.
Cancellation returns the input model without writing a partially adapted file.
Plain export and Studio's existing export action remain training-free.

## SH clustering and fixed-model evaluation

SH clustering now spends the exact requested codebook budget, including
non-power-of-two budgets, then performs three global Lloyd updates across all
initialization batches. The bounded GPU point batches share the same final
centers; no N-by-K distance matrix is allocated. Empty codewords are filled
from non-singleton clusters and their means are recomputed. Constant/repeated
SH vectors retain deterministic tie membership. This corrects the former
power-of-two truncation and batch-local-only optimization.

On the indoor iPhone POPSpa output (347,718 degree-3 Gaussians), the former
encoder used 30,837 codewords instead of the requested 43,264. The new encoder
uses all 43,264. A fixed-model sweep over all 739 registered cameras, at a
960-pixel maximum dimension with Mip2D rendering, measured:

| Stage | Mean PSNR | Mean SSIM | Compaction-holdout PSNR | SOG-XT size |
| --- | ---: | ---: | ---: | ---: |
| Input PLY | 23.6808 dB | 0.957596 | 23.1082 dB | — |
| Former encoder | 23.2682 dB | 0.952129 | 22.7835 dB | 5.797 MB |
| Exact budget + global Lloyd | 23.4161 dB | 0.953797 | 22.9260 dB | 6.135 MB |
| Encoding-aware adaptation | 23.5948 dB | 0.956209 | 23.0532 dB | 6.120 MB |

The final row uses 4,000 steps, eight views per step, both PPISP and the
bilateral grid, and selects step 4,000. All 93 compaction-holdout cameras are
excluded from adaptation and snapshot choice. Average encoding loss relative
to the same input PLY falls from 0.413 dB to 0.086 dB; views losing more than
0.5 dB decrease from 293 to 2, and the worst decrease improves from 1.822 dB to
0.561 dB. The final file grows about 5.6%. These figures isolate encoding;
they do not undo earlier POPSpa errors or establish a multi-scene guarantee.
The first adaptation experiment with higher rates and no input-render anchor
improved SSIM but reduced PSNR; it is not the selected recipe.

Measured on an RTX 5090 D v2, the former/new encoding passes took about
25.2/32.8 seconds, and the quality-preserving adaptation including codec setup,
selection and export took about 315 seconds. Evaluation renders are additional.
Use `photara_splat_popspa_review ... OUTPUT codec` for a fixed-model encode and
`photara_splat_popspa_review ... OUTPUT adapt 4000` for adaptation. Compare the
four sweeps with `scripts/analyze_sog_xt_refinement.py`.

Fixed training ranges can lose an observed code endpoint during adaptation.
The frozen export algebraically rebases codes and metadata ranges, without
introducing a new quantization, to match the paper decoder's observed-range
normalization. Native and independent NumPy reference decoding were compared
across all 347,718 rows and 59 PLY fields; the external-style YAML reader was
also checked. The decoder accepts explicit precisions up to the format's field
limit, including smaller effective code ranges from this rebasing.

## SOG and SOG-XT comparison

A check on the supplied indoor iPhone scene, with 1,738,589 degree-3 Gaussians,
gave the following results (decimal MB, same count, no pruning):

| Format | Size | Encode | Decode | Mean PSNR vs original render |
| --- | ---: | ---: | ---: | ---: |
| PLY | 458.99 MB | 0.37 s | 0.41 s | Reference |
| SOG | 30.41 MB | 4.67 s | 0.67 s | 35.31 dB |
| SOG-XT | 26.17 MB | 74.13 s | 0.92 s | 35.22 dB |

Eight evenly spaced registered cameras from the neighboring `sfm.bin` were
rendered at an 800-pixel maximum dimension using the same CUDA raster settings.
PSNR measures deviation from the original PLY render, not fidelity to source
photographs. SOG-XT is 13.95% smaller, averaging 0.096 dB below SOG; its worst
view is 0.639 dB below SOG. This is one scene, not a universal rate-distortion
claim. Encoding is substantially slower and does not reduce GPU storage.
Do not apply the paper's full-pipeline compression factors to this export.

## Validation

From the repository root, enable tests in an existing configured build and
explicitly build the format test target (tests are excluded from the default build):

```sh
cmake -S . -B build -DPHOTARA_BUILD_TESTS=ON
cmake --build build --config Release --target photara_splat_formats_test --parallel
```

To repeat the measurement and retain files/screenshots on Windows:

```powershell
build/photara/Release/photara_splat_formats_test.exe --sog-xt-benchmark model.ply output-directory
python scripts/test_sog_xt_reference.py output-directory/benchmark.sog-xt output-directory/decoded-sog-xt.ply output-directory/reference-container
build/photara/Release/photara_splat_formats_test.exe --sog-xt-compare output-directory/reference-container output-directory/decoded-sog-xt.ply
```

For single-config Linux/macOS builds, use
`build/photara/photara_splat_formats_test` in place of the Windows executable.
The independent decoder script requires Python with NumPy, Pillow, and PyYAML.

The format tests also cover SH/position association after sorting, covariance
preservation before PRAS quantization, opacity, constant attributes, single-point
models, padding, ordinary reference-style YAML, and rejection of corrupt assets.
The CLI RPC fixture tests cover waiting for asynchronous exports, returning
their errors, and ignoring unrelated historical export failures when waiting
for a training operation.
