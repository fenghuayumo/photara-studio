# AI Inference Module

## Reading Order

1. `photara/include/inference/geometry_prior.hpp`: data consumed by reconstruction modules.
2. `photara/include/inference/geometry_predictor.hpp`: the image-to-geometry-prior task interface.
3. `photara/include/inference/options.hpp` and `runtime.hpp`: execution configuration and actual devices.
4. `photara/src/inference/runtime/ggml_runtime.*`: backend initialization, graph allocation, and execution.
5. `photara/src/sam/masks.cpp`: how existing models integrate with Photara workflows.

## Current Implementation

```text
CLI reconstruction ──→ Photara::SAM ──→ photara_sam3 ──→ ggml
                             │
Studio ──────────────→ Photara::SAMCache
                             │
                       Photara::Inference ──→ ggml Vulkan / CPU
```

`Photara::Inference` provides geometry validation, model-file resolution, device
enumeration, and internal ggml sessions for model adapters. Public headers contain
no ggml or Vulkan types. `Photara::SAM` has been extracted from the CLI `tools`
module, but the external `sam3` library still manages its computation graphs and
backend lifecycle rather than using the new session wrapper. Both share the same
ggml build.

MoGe provides weight conversion, GGUF inspection, a ViT-B encoder, the neck and
points/normal/mask/scale heads, RGB preprocessing, and a public Vulkan
`GeometryPredictor` factory. Predictions produce a `GeometryPrior` at the original
image resolution. MoGe prediction requires Vulkan and rejects CPU fallback;
existing low-level CPU utilities remain available for reference diagnostics.
Existing ONNX feature and delighting paths remain unchanged; their session
wrappers are not being migrated at this stage.

## Directories and Responsibilities

| Directory | Responsibility |
|---|---|
| `photara/include/inference` | Public data and task contracts |
| `photara/src/inference/runtime` | Backends, graph allocation, and execution; no knowledge of model architecture |
| `photara/src/inference` | Public geometry validation and model-cache paths |
| `photara/src/inference/models/moge` | MoGe weights and computation graphs |
| `photara/src/sam` | SAM mask workflows, model locations, and license records |
| `third_party/sam3` | External model implementation and current ggml sources |
| `photara/tests/inference` | Weight-free public-contract and real-backend tests |

MoGe weight loading, computation graphs, future preprocessing, and geometry
recovery belong in the model directory. Extract common implementations only when
multiple models actually share operators; do not create empty model directories
or a generic operator framework in advance.

MoGe-2 integration targets `Ruicheng/moge-2-vitb-normal`. The model
adapter owns the token grid, weight loading, computation graph, and geometry
recovery. Callers provide optional pinhole intrinsics through `GeometryInput`.
For production integration, pin the weight revision, compare reference outputs
stage by stage, and then validate Vulkan/CPU accuracy and reconstruction quality.

## MoGe Weight Pipeline

`python/models/moge/convert_weights.py` converts the official `model.pt` to GGUF,
preserving original tensor names and contiguous storage order. GGUF dimensions
are ordered in reverse relative to PyTorch, but conversion does not rearrange
tensor storage. Convolution and transposed-convolution graphs must therefore use
the correct layouts. The first supported model is MoGe-2 ViT-B normal. F32 is the
default for initial numerical validation. Optional F16 converts only matrix and
convolution weights; positional embeddings and one-dimensional affine parameters
remain F32.

The file records the format version, original configuration, checkpoint revision,
and source-file SHA-256. Validation used revision
`ca5f0e07ff01d3e5a364c1d954ed12ee1814b368` and source-file SHA-256
`16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`.
C++ `inspect_moge_model()` validates the format, metadata, major encoder/output
tensor shapes, and completeness of the data region without allocating GPU memory.
Internal `MogeWeights` handles backend allocation and upload and accepts a limit
on raw tensor bytes. That limit excludes allocation alignment and is not a global
VRAM budget. The SHA-256 identifies the source file; it is not an integrity digest
of the converted GGUF.

```powershell
python -m pip install -r python/models/moge/requirements.txt
python python/models/moge/convert_weights.py model.pt model-f32.gguf --revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368
cmake --build build --config Release --target photara_moge_weights_test
python python/models/moge/verify_weights.py model.pt model-f32.gguf build/photara/Release/photara_moge_weights_test.exe
python python/models/moge/verify_weights.py model.pt model-f32.gguf build/photara/Release/photara_moge_weights_test.exe --vulkan
```

The verification tool independently samples the beginning, interior, and end of
every source-checkpoint tensor and compares them with values read back by C++ from
the actual backend. F16 files require `--precision f16` for both conversion and
verification. Model files are not committed to the repository; default CTest runs
neither download nor load checkpoints. The converter also validates the fixed
ViT-B neck/head topology, rejecting different normalization, activation, residual
counts, or upsampling configurations. Preprocessing and the complete prediction
interface still need implementation. Successful weight readback does not establish
correct model inference.

## ViT Encoder and Layer-by-Layer Comparison

`models/moge/encoder.*` implements single-image ViT-B/14: patch projection, class
token, positional embeddings, 12 Transformer blocks, normalization at layers 5/11,
and feature projection. Outputs are token-major fused 768-dimensional features
and the final class token for the neck and scale head. The graph includes
multi-head attention, LayerScale, residual connections, `LayerNorm eps=1e-6`,
and erf GELU.

The current input is a prepared CHW floating-point RGB image of size
`rows*14 × columns*14`. Full image resampling and normalization are not yet
integrated. Patch ordering and native DINOv2 positional interpolation are
implemented in the encoder; interpolation retains the 0.1 offset and does not
enable antialiasing. Attention uses explicit matrix operations, with grids limited
to 1..4096 patches. Flash attention has not yet been optimized.

`compare_encoder.py` directly instantiates the encoder from local official MoGe
sources and strictly loads the same checkpoint. Hooks capture reference outputs;
there is no separately reimplemented reference Transformer. Validation covers 21
stages, including every block, both feature taps, final features, and the class
token. The native tool also repeats the normal path without retaining intermediate
results and checks that it matches the diagnostic path.

```powershell
cmake --build build --config Release --target photara_moge_encoder_test
python python/models/moge/compare_encoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_encoder_test.exe --moge-source path/to/MoGe --rows 30 --columns 40 --report build/encoder-cpu.json
python python/models/moge/compare_encoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_encoder_test.exe --moge-source path/to/MoGe --rows 30 --columns 40 --vulkan --report build/encoder-vulkan.json
```

The reference source revision is `74fbce054ebed49800de42d0ad0e83495065719a`.
F32 checkpoints have been validated on CPU/Vulkan with 8×12 and 30×40 grids,
and on Vulkan with 12×8 (different random input) and 37×37 (native positional
embeddings) grids. Inputs are synthetic RGB images with deterministic seeds;
complete preprocessing of real images is not yet covered. Relative L2 error of
the final 30×40 features is approximately `2.86e-6` on CPU and `3.33e-6` on strict
Vulkan.

Default thresholds are relative L2 ≤ `2e-4` and maximum absolute error ≤
`2e-3 + 2e-4 * max(abs(reference))`. Reports also retain raw absolute error and
reference magnitude. Later residual layers can have activations above 300, so one
fixed absolute-error threshold is inappropriate for all layers.

Vulkan comparison uses strict F32 mode by default, setting
`GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2=1` only for the subprocess. It does not modify
the editor's or library caller's environment. `GGML_PREC_F32` constrains only some
accumulation policies and does not guarantee that the backend avoids converting
inputs to F16. Use `--vulkan-precision default` to inspect the default fast path.
Its 8×12 result has not passed the thresholds above (fused-feature relative L2
approximately `6.1e-3`) and must not be described as equivalent to strict mode.
The library does not change these process-wide options itself; product integration
must define an explicit precision policy. Encoder forward accuracy with F16
weights has not yet been accepted.

## Neck/Head Decoder

`models/moge/decoder.*` takes the encoder's fused features, class token, and the
original input image's aspect ratio. That ratio may differ from the integer token
grid's ratio; the UV pyramid must not substitute the grid ratio. The implementation
has a five-level neck and three image heads, with channels 768/256/128/64/32.
The scale head is a separate MLP.

3×3 convolutions construct replicated-edge padding through gather and explicitly
use F32 im2col. 2×2/stride-2 transposed convolutions use four-phase projection and
pixel rearrangement to avoid confusing regular and transposed-convolution weight
layouts. The final level uses bilinear upsampling.

`DecoderResult` returns raw HWC points, raw normals, mask logits, and log metric
scale at 16 times the grid resolution. The public `MogeRawGeometry` contract passes
these outputs to postprocessing, which produces geometry priors for reconstruction.

```powershell
cmake --build build --config Release --target photara_moge_decoder_test
python python/models/moge/compare_decoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_decoder_test.exe --moge-source path/to/MoGe --vulkan --report build/decoder.json
python python/models/moge/compare_decoder.py model.pt model-f32.gguf build/photara/Release/photara_moge_decoder_test.exe --moge-source path/to/MoGe --rows 30 --columns 40 --vulkan --native-encoder --report build/network-raw.json
```

The comparison tool validates 104 stages using the official ConvStack/MLP and the
same weights. By default, reference encoder features are passed directly to the
native decoder to isolate decoder error. `--native-encoder` instead chains the
native encoder and decoder to validate their actual module boundary. Both modes
repeat decoder execution without retaining intermediate outputs and check that
normal and diagnostic paths agree. Vulkan validation uses the same strict F32
subprocess policy; the default fast path and F16 weights have not yet been accepted.

Validation passed for CPU grids 4×6 and 1×1 and Vulkan grids 4×6, 6×4 (original
image aspect ratio 1.7), and 30×40. Chained validation passed on CPU 4×6 and Vulkan
30×40. Relative L2 errors for the chained 30×40 outputs are approximately `5.24e-7`
for raw points, `4.33e-7` for normals, `7.27e-6` for mask logits, and `2.51e-6` for
log metric scale. The normal decoder graph buffer for this grid is approximately
901 MiB. This excludes model weights, diagnostic data, upload staging, and backend
internal buffers and does not represent peak VRAM usage for the entire process.

## Output Postprocessing and Geometry Recovery

The public entry point is `recover_moge_geometry()` in `inference/models/moge.hpp`,
implemented in `models/moge/postprocess.cpp`. Inputs include raw head outputs,
original image dimensions, optional pinhole intrinsics, and model/device metadata.
The output is `MogeGeometryResult`, containing a `GeometryPrior` plus recovery
status, z-shift, normalized focal length, reprojection RMS, and sample count.
Postprocessing runs on CPU and does not require the caller to hold a ggml session.

Processing follows official MoGe-2 ordering: first bilinearly restore the original
image size (`align_corners=false`), then apply `z=exp(raw_z)`, `x/y=raw_x/y*z`,
predicted-normal normalization, mask sigmoid, and scale exp. The default mask uses
strict `probability > 0.5`. Infinities, NaNs, zero normals, and nonpositive depths
are excluded from recovery. The scale converts final camera-Z depth to meters.

Recovery uses the official nearest-neighbor 64×64 sampling grid and requires at
least four distinct valid pixels. With unknown intrinsics, it assumes `fx=fy` and
a principal point at `(width-1)/2, (height-1)/2`, jointly recovering focal length
and z-shift. With known intrinsics, it preserves the full K, supports different
fx/fy and off-center principal points, and recovers only z-shift. Output points
are reprojected using recovered depth and K to maintain pinhole consistency.
Normals retain their model-predicted provenance. Confidence remains empty; mask
probabilities are not treated as geometric confidence.

The native solver does not depend on SciPy. Focal length has a closed-form solution
for a fixed shift; the solver performs logarithmic grid search and local
golden-section search under positive-depth constraints. Its implementation differs
from the official optimizer and is validated through numerical output comparisons.
Constant-Z point maps with unknown focal length are considered degenerate.
Insufficient support returns `no_support`; degeneracy or a search-boundary solution
returns `degenerate`. Failed results have an all-zero valid mask and do not invent
intrinsics. Successful recovery does not establish multi-view consistency; callers
must still evaluate reprojection RMS and reconstruction constraints.

```powershell
cmake --build build --config Release --target photara_moge_geometry_test photara_moge_postprocess_compare
ctest --test-dir build -C Release -R '^photara\.inference\.moge_geometry$' --output-on-failure
python python/models/moge/compare_geometry.py build/photara/Release/photara_moge_postprocess_compare.exe --moge-source path/to/MoGe --report build/geometry-reference.json
```

Contract tests cover positive/negative shifts, known/unknown intrinsics, off-center
principal points and different fx/fy, metric scale, point-map projection
consistency, empty masks, sparse support, degenerate planes, and invalid inputs.
The reference script directly invokes the official forward/infer postprocessing
path; eight synthetic cases passed. The reference environment also requires the
official `utils3d_moge` package and SciPy for Python validation only, not as C++
runtime dependencies. `--raw-dir` reads native decoder diagnostic outputs, with
dimensions specified through `--raw-width/--raw-height`. Outputs from real F32
weights, synthetic input, and the strict Vulkan encoder/decoder also passed
known/unknown-focal comparisons: depth relative L2 is approximately `1.50e-7` /
`2.80e-6`, normal error approximately `5.86e-8`, and valid masks agree.
There is no automatic integration into SfM/MVS/Splat or scene-camera updates.

## RGB Prediction and Real-Image Validation

`create_moge_predictor()` loads a reusable MoGe-2 ViT-B normal session. The caller
supplies tightly packed 8-bit RGB in `GeometryInput`; decoding and camera
undistortion stay with the caller. Optional intrinsics describe those exact input
pixels. The adapter never updates scene cameras. Example:

```cpp
#include "inference/models/moge.hpp"

photara::inference::MogeOptions options;
options.num_tokens = 1200;
// Vulkan device 0; CPU fallback is disabled and is not supported by this adapter.
auto predictor = photara::inference::create_moge_predictor("model-f32.gguf", options);
photara::inference::GeometryInput input{rgb_image, pinhole_intrinsics};
auto prior = predictor->predict(input);
```

Preprocessing selects `rows=round(sqrt(tokens/aspect))` and
`columns=round(sqrt(tokens*aspect))` using Python's ties-to-even rounding. The
rounded grid must contain 1..4096 patches. The default budget is 1200, the low end
of the checkpoint's intended 1200..3600 range. The image is resized to
`columns*14` by `rows*14` with separable bilinear antialias filtering,
`align_corners=false`, then ImageNet-normalized in F32 CHW order. Antialias filter
coordinates, edge normalization and float arithmetic follow ATen's F32 path.
Original aspect ratio is passed separately to the decoder. Outputs are recovered
at the original RGB resolution with camera-Z depth, predicted camera-space unit
normals, projected points, a binary valid mask, model/revision/device metadata and
`preprocessing_revision="moge2-rgb-aa-imagenet-geometry-v1"`. Confidence stays empty.

Progress is reported at 0, 0.1, 0.5, 0.9 and 1. Stop requests are checked before
and after each callback, between preprocessing, encoder, decoder and recovery.
Cancellation and failed geometry recovery throw `std::runtime_error`; malformed
inputs/options throw `std::invalid_argument`. Cancellation cannot interrupt an
in-flight GPU dispatch. A cancelled session can be reused; concurrent calls are
unsupported.

`python/models/moge/compare_prediction.py` runs the complete native predictor
against the **unmodified official v2 `infer()`** with F32 weights/arithmetic.
Pillow decodes each real photograph once and shares identical RGB bytes with
both implementations; codec differences are excluded from this test. The script
also isolates resize math on downsample, upsample, singleton and mixed-axis
fixtures. It exports depth/normal/mask previews and raw maps, and records input,
checkpoint and GGUF hashes, source commit, grid, settings and errors in JSON.
Known-focal cases supply a centered square-pixel test K (`fx=fy=0.75*width`);
this is an API comparison, **not calibrated ground truth** for those photographs.

The normal/general relative-L2 limit is `5e-4`, mask mismatch limit `1e-4`, and
normal angular p99 limit 0.1 degrees. Known-focal depth/points additionally allow
`1e-3` against official LM, whose stopping tolerance is `ftol=1e-3`. An independent
float64 SciPy solve with `ftol/xtol/gtol=1e-13` must also agree with native depth
within `5e-4`; the official source and inference results are never modified.
This distinguishes graph errors from optimizer termination: on the mountain
reference image the original LM shift is 0.2022033, native is 0.2112307 and tightly
converged SciPy is 0.2112301. Native depth agrees with that control to relative L2
`9.4e-7`; the original-infer depth difference is approximately 0.084%.

**Precision scope:** acceptance uses F32 GGUF and Vulkan with
`GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2=1` set **only in the diagnostic child** before
backend initialization. Library code never changes process environment or shared
device precision. The ordinary fast Vulkan path still implicitly converts some
F32 operands to F16 and did not pass these strict normal gates; it is not accepted
by this validation. A production caller requiring this baseline must configure
the same startup policy. F16-weight prediction remains unaccepted.

```powershell
cmake --build build --config Release --target photara_moge_predict_test
ctest --test-dir build -C Release -R '^photara\.inference\.' --output-on-failure
python python/models/moge/compare_prediction.py model.pt model-f32.gguf build/photara/Release/photara_moge_predict_test.exe --moge-source path/to/MoGe --images room.jpg portrait.jpg outdoor.jpg --output build/moge-prediction
```

## Experimental 3DGS Prior Supervision

`splat::TrainingOptions::lambda_d` and `lambda_n` independently weight external
AI depth and normal supervision. Both default to zero. `prior_directory` selects
an offline cache; CLI equivalents are `--splat-lambda-d`, `--splat-lambda-n`, and
`--splat-prior-dir`. Python `TrainingOptions` exposes the same three fields.
This initial experiment is available through the CLI/API, not the Studio panel.
The weights do not enable DN, normal-field training, multi-view objectives, the
geometry-training mode, or its automatic 3D filter. Fast mode remains RGB-only
and clears these weights. Legacy MVS supervision and AI priors cannot share the
same target slots in one training run.

`photara_splat_prior_prepare` supports a separate native Vulkan preparation process:

```powershell
photara_splat_prior_prepare dataset/sparse/0 images model-f32.gguf prior-cache 640 1200 8 0.01 0.01
photara --images images --splat-dataset dataset/sparse/0 --splat --mesh=false --splat-max-resolution 640 --splat-eval-split-every 8 --splat-prior-dir prior-cache --splat-lambda-d 0.01 --splat-lambda-n 0.01
```

The preparation arguments after the output directory are maximum training image
dimension, token budget, evaluation split interval, lambda_d and lambda_n. Both
weights default to zero: preparation returns before opening the dataset/model or
creating output directories when neither is positive. A depth-only request saves
only depth; a normal-only request saves only normals and skips sparse depth-scale
alignment. Pass the same weights to preparation and training. The current MoGe
adapter still computes its joint depth/normal output internally; selective caches
do not yet imply selective model heads or proportional inference-time savings.
Use the accepted strict
F32 Vulkan startup policy documented above. Preparation uses the exact training
RGB decode, undistortion, pixel-center resizing and known camera intrinsics. It
always requests pinhole views; fisheye training must likewise use
`--splat-undistort=true`. Panorama priors are not supported. Preparation skips
held-out views entirely. A median/MAD log-ratio fit aligns predicted camera-Z
depth to **observed SfM sparse points**, requiring at least 16 inliers per view.
There is no MVS depth estimation, dense fusion, or mesh reconstruction.

Each local `PHPRIOR2` `.prior` file stores the selected scene-unit camera-Z depth
and/or CHW camera-space unit normals, plus camera intrinsics/pose and source-image
identity/size/mtime. Map-presence flags distinguish absent channels; the loader
rejects a cache missing an enabled channel. Existing `PHPRIOR1` caches remain
readable. Training reads, resizes and uploads only requested channels, even when
an older cache contains both. No cache is read when both weights are zero. Zero depth
and zero normals independently mark invalid supervision. Cache loading checks
camera, input identity, payload size and map values; changing the image, camera,
or full training resolution requires regeneration. Progressive training
resolution uses pixel-center nearest sampling, retaining depth units and avoiding
interpolated surfaces at depth discontinuities. The training image's RGB mask is
never replaced by prior validity. Evaluation explicitly disables prior loading,
so held-out images need no prediction/cache. Positive weights and the cache file
identities are included in checkpoint compatibility; a disabled prior directory
does not affect the ordinary RGB-only fingerprint.

The losses on CUDA and Vulkan are:

```text
L_depth  = lambda_d / (W*H) * sum_valid mask * sqrt(
               ((D_render - D_prior) / max(D_prior, 1e-4))^2 + epsilon^2)
L_normal = lambda_n / (W*H) * sum_valid mask * (1 - dot(N_render, unit(N_prior)))
```

Depth uses `rendered.median_depth`; the normal is the existing alpha-normalized
rendered mixture, not a separately learned normal field. Both losses accumulate
their derivatives into the main rasterization backward pass. There is no extra
full render for the prior losses. If both weights are zero, no cache is read and
AI priors do not request any geometry channels. If either is positive, the
current renderer still allocates depth **and** normal geometry attachments and
workspace together; setting only one weight does not split that workspace. The
existing bounded host/device view caches include the enabled target maps.
Offline MoGe GPU allocations are released with the preparation process before
training starts.

Studio also supports automatic preparation inside the training child process:
`--splat-prior-model model.gguf --splat-prior-tokens 1200` uses the finalized
training scene and options, then destroys the MoGe session before allocating
the Gaussian model. `splat::prepare_training_priors` is shared with the
standalone utility. It validates and reuses compatible maps before loading
MoGe, and emits per-view preparation progress. The Studio geometry inspector
offers automatic generation or an existing aligned cache, independent depth
and normal weights, and project persistence. See [Studio AI priors](STUDIO_AI_PRIORS.md).

`photara.splat.ai_prior` checks cache validation, progressive resolution,
RGB-mask independence, checkpoint fingerprints, held-out evaluation without a
prior, and analytic/finite-difference gradients on CUDA for depth-only,
normal-only, combined, disabled, invalid and fractionally masked pixels.
Vulkan loss checks are a separate `photara.splat.ai_prior_vulkan` test; current
training acceptance focuses on CUDA.
`python/models/moge/test_splat_priors.py` reproduces the four-way held-out ablation
with fixed seed, cameras and budgets, and DN/normal field/multi-view/filter off.
It records exact commands, binary hash, selected views, PSNR/SSIM, training time,
and sampled device memory. Its memory readings are **not** an allocator peak.
These priors are optional regularizers; passing inference/gradient checks does
not establish a rendering-quality improvement.

### Initial CUDA ablation (2026-10-15)

The indoor iPhone capture was subsampled to 96 registered views, with 84 training
and 12 held-out views, maximum image dimension 640, seed 42 and 30,000 iterations.
Initial Gaussian budget was 100,000 and the densification cap was 300,000. All
four runs disabled DN, normal fields, multi-view geometry and filtering. Priors
were prepared with Vulkan F32 MoGe and aligned using SfM sparse tracks; no dense
MVS reconstruction was used.

| lambda_d | lambda_n | Held-out PSNR | Held-out SSIM | Training seconds |
|---|---|---|---|---|
| 0 | 0 | 21.0023 | 0.880123 | 62.382 |
| 0.01 | 0 | 21.4727 | 0.884348 | 175.681 |
| 0 | 0.01 | 20.6427 | 0.883844 | 172.553 |
| 0.01 | 0.01 | 21.1260 | 0.884268 | 117.730 |

Depth-only improved PSNR by 0.4704 dB in this single-scene, single-seed experiment,
with approximately 2.82 times the baseline training time. Normal-only reduced
PSNR despite improving SSIM; enabling both did not outperform depth-only. Start
experimentation with `lambda_d=0.01, lambda_n=0`, keeping both product defaults
zero. These measurements do not establish a general quality or throughput gain.
The four runs can produce different Gaussian geometry and render workloads;
single-run timings under a desktop session are not a controlled performance
benchmark. Preparation took another 13.14 seconds, outside the training times.
Commands, binary hash and results are recorded locally in
`build/ai-prior-ablation-indoor-30k-20261015/summary.json` and per-run logs.

## Geometry Conventions

- Inputs are pinhole RGB images; callers handle fisheye and panorama resampling.
- Integer coordinates denote pixel centers, with the image origin at the top left.
  Camera axes point right, down, and forward.
- Depth types explicitly distinguish camera Z, inverse Z, and distance along a ray;
  scale distinguishes relative values from meters.
- Point maps contain recovered camera-space points, not unrecovered affine model
  outputs.
- Normals are camera-space unit vectors and must declare predicted or derived
  provenance.
- `valid` covers the entire output with values 0/1. Invalid pixels may contain NaNs;
  consumers must check the mask first. Confidence may be empty, and a validity mask
  must not be treated as confidence.
- Intrinsics always describe the input image. Output pixels map to input pixels
  through scale/offset; the mapping is identity at the restored input resolution.
- `GeometryPrior::validate()` checks dimensions, finite values, and basic semantics.
  It does not replace multi-view consistency, depth/point-map consistency, or scale
  alignment checks.

Reconstruction consumers own prior alignment, initialization, and geometric
filtering; the experimental Splat path aligns to SfM sparse observations without
running MVS. Splat converts priors into training constraints. Model adapters must not depend on SfM/MVS/Splat
or modify scene cameras or training parameters. CLI/Studio orchestration loads
models and passes their results to consumers.

## Build

| Option | Default | Meaning |
|---|---|---|
| `PHOTARA_ENABLE_INFERENCE` | ON | Build shared ggml and Inference |
| `PHOTARA_INFERENCE_VULKAN` | ON | Build Vulkan; warn and retain CPU support if the SDK/glslc is missing |
| `PHOTARA_INFERENCE_CUDA` | OFF | Optional existing SAM CUDA backend |
| `PHOTARA_ENABLE_SAM` | ON | Build the external SAM model |
| `PHOTARA_GGML_ROOT` | `third_party/sam3/ggml` | Shared ggml source location |

Legacy `PHOTARA_ENABLE_SAM_VULKAN/CUDA` options only initialize the defaults of
their corresponding new options. Once the new options are in the CMake cache,
use the new names for subsequent changes. Disabling SAM does not disable
Inference. Disabling Inference makes the SAM model unavailable, but retains
license/cache APIs and the CLI's segmentation-unavailable message. Inference sessions do not download weights themselves. Studio downloads the
optional MoGe GGUF before first AI-prior training and verifies its pinned size
and SHA-256; see [Studio AI priors](STUDIO_AI_PRIORS.md). SAM retains its separate
license-acceptance flow. License-acceptance state is not changed automatically.

New sessions request Vulkan by default. CPU fallback on initialization failure
is configurable, and the actual device is exposed. Unsupported operators fail
explicitly; graph execution does not silently mix in CPU operations. Sessions and
graph allocators do not support concurrent calls; each session is used serially
on one calling thread. The CUDA build option retains support for existing SAM;
the new public sessions support only Vulkan/CPU.

Sessions manage their own ggml backends and do not reuse the editor's VkDevice.
The initial module boundary uses CPU data to avoid cross-device synchronization.
Available device memory is a snapshot, and graph-buffer statistics are not a
global budget. Cross-rendering/reconstruction VRAM reservation, task queues, and
disk caching of inference results are not yet implemented. There is also no
guarantee of cancelling an in-flight GPU dispatch. `PredictOptions` defines
cancellation and progress contracts for model adapters, which must implement
them at stage boundaries.

## Validation

```powershell
cmake --build build --config Release --target photara_inference_test photara
ctest --test-dir build -C Release -R '^photara\.inference\.' --output-on-failure
```

Contract tests validate valid regions, scale/normal metadata, file-size checks,
CPU fallback, and strict device selection. Vulkan tests execute real graphs,
upload data, read results back, and verify them. Without a Vulkan device, they
report a skip, which must not be treated as successful GPU validation. These
tests do not use model weights.
