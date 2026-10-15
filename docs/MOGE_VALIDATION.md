# MoGe-2 RGB prediction validation

Date: 2026-10-15. Native backend: Vulkan, NVIDIA GeForce RTX 5090 D v2.
Model: `Ruicheng/moge-2-vitb-normal`, F32 GGUF.
Official reference: `microsoft/MoGe` commit
`74fbce054ebed49800de42d0ad0e83495065719a`, PyTorch 2.7.1, F32 inference.
Checkpoint SHA-256:
`16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`.

This checks implementation agreement with the official model. It does **not**
establish ground-truth depth accuracy, multi-view consistency, or improved 3DGS.
Input distortion is not estimated or removed by the predictor.

## User photographs

All six original JPEGs were decoded at **2560 × 3840**, without resizing the
source or output. Token budget 1200 selects a 42 × 28 grid (1176 actual patches).
Each image was tested with unknown intrinsics and supplied centered test
intrinsics (`fx=fy=1920`, `cx=1279.5`, `cy=1919.5`). The supplied K exercises the
known-intrinsics API; it is not a calibration of these photos.

All 12 comparisons passed. Preprocessing matched the CPU PyTorch F32 reference
exactly on these inputs. Masks matched on every pixel; all pixels were valid.
The following maxima include both intrinsics modes:

| Photograph | Depth relative L2 | Normal angular p99 (degrees) |
| --- | ---: | ---: |
| DCIM5021-HDR.jpg | 1.07e-6 | 0.000388 |
| DCIM5024-HDR.jpg | 1.37e-6 | 0.000296 |
| DCIM5012-HDR.jpg | 1.07e-6 | 0.000435 |
| DCIM5018-HDR.jpg | 5.26e-6 | 0.000492 |
| DCIM5027-HDR.jpg | 6.79e-7 | 0.000634 |
| DCIM5030-HDR.jpg | 7.76e-6 | 0.000329 |

The maximum individual normal angle was approximately 0.00348 degrees.
`DCIM5021-HDR.jpg` additionally passed both intrinsics modes at a 3600 token
budget. Three official indoor/portrait/outdoor example photographs passed six
additional Vulkan comparisons at 1200 tokens, including nontrivial invalid masks.

## Precision and solver limits

Acceptance uses Vulkan F32 with `GGML_VK_DISABLE_F16=1`,
`GGML_VK_DISABLE_COOPMAT=1`, and `GGML_VK_DISABLE_COOPMAT2=1` in the diagnostic
child process before backend initialization. No CPU fallback is permitted.
This keeps matrix operands in F32, rather than only requesting F32 accumulation.
The library itself does not mutate the host process environment or shared
device settings. The ordinary fast Vulkan path was also checked and exceeded
the strict normal tolerance; it and F16-weight prediction remain unaccepted.

Known-focal official LM uses `ftol=1e-3`, so the comparison also records an
independently converged float64 SciPy control. The 0.084% known-focal depth
difference on the official mountain photo comes from optimizer termination:
native and the tight control agree to about `9.4e-7` relative L2. The original
official `infer()` and its source are not altered to obtain a passing comparison.
See [inference architecture](INFERENCE_ARCHITECTURE.md#rgb-prediction-and-real-image-validation)
for gates and API details.

## Reproduce

Build `photara_moge_predict_test` in Release. The four
`photara.inference.*` CTest checks passed, including malformed input/options,
antialias constant preservation, ties-to-even grid rounding, and existing geometry
and Vulkan contracts. Weighted tests also verified stage-boundary cancellation,
reuse after cancellation, progress, output semantics and repeated prediction.

```powershell
python python/models/moge/compare_prediction.py model.pt model-f32.gguf build/photara/Release/photara_moge_predict_test.exe --moge-source path/to/MoGe --images room1.jpg room2.jpg --output build/moge-prediction
```

`compare_prediction.py` writes `report.json`, a `preview.jpg` contact sheet, and
per-case raw maps plus PNG previews. Pillow decoding, SciPy and PyTorch are
validation dependencies only; C++ inference uses ggml/Vulkan and standard C++.
The script configures strict Vulkan precision only for its native subprocess.

Local reports for this run are under:

- `build/moge-model/prediction-user-six-f32/report.json`
- `build/moge-model/prediction-user-high-f32/report.json`
- `build/moge-model/prediction-real-f32-final/report.json`

No AI-prior cache, SfM/MVS alignment, or 3DGS supervision was introduced in this
change. Those follow only after this RGB prediction baseline.
