# ADC-IGS reconstruction quality experiment — 2026-09-14

已在指定数据集上完成两轮串行对比：有效像素 PSNR 平均 **34.7230 → 35.1802 dB（+0.4572）**，SSIM **0.981712 → 0.982926**。全图 PSNR **21.4001 → 21.2931 dB（−0.1069）**，全图 SSIM **0.967802 → 0.968851**。有效像素由相机映射确定，两版使用同一掩码；全图退步主要涉及未观测的去畸变黑边，不隐去该结果。已检查的新视角中，底边黑色尖片减少；这不是对所有漂浮物的几何定量证明。改进版使用更多高斯点，训练耗时也增加。

## Scope and reproducibility

Dataset: `D:/ScanVideo/WeChat_20250712175936/sparse/0`, using its current 198 images and 38,405 sparse points. The reference is the user's `D:/SoftwareTool/photara-v1.0/photara.exe`. Earlier reports using a different sparse point count are not comparable.

The final comparison uses 30,000 iterations, a 1,000,000 Gaussian cap, full source resolution, RGB supervision, no foreground mask, and every eighth camera held out (173 training / 25 evaluation views). Depth-normal, multi-view geometry and NCC losses are explicitly zero for both executables. Runs are serial; two fresh processes per executable use the default seed. This checks numerical repeatability, not robustness across seeds or scenes. The dataset was used for tuning, so these held-out-camera results are not an untouched benchmark.

`experiments/benchmark_igs_quality.py` records executable and sparse-file SHA-256 hashes, commands, logs, PLY models, and per-view metrics. `artifacts/igs_quality_20260914_confirm/comparison.json` is the authoritative final result. The earlier ablations are in `artifacts/igs_quality_20260913`; their timings are not a controlled speed comparison.

## Implementation

Reference implementations read from `D:/ProgramCode/C++/spirula-studio/src/engine/EngineDensify.cpp`, `src/shaders/densify.slang` and `src/shaders/per_splat_losses.slang` informed these changes:

- Accumulate the mean of powered observation scores, rather than powering their mean. Combine image error evidence with scale-normalized world-position gradients using a geometric blend of 0.5.
- Correct revised position noise to use Gaussian scales as the eigenvalues of the covariance square root. Its schedule is dimensionless and stops with refinement. The old square-root-of-scale and scene-dependent factor caused excessive motion.
- Accumulate logarithmic screen-oversize evidence across observations; avoid selecting a split solely from one large projected footprint.
- Use covariance-preserving long-axis splitting. Reset split optimizer states; inheriting moments was tested and rejected for poorer reconstruction.
- Normalize growth and opacity decay for the 100-step refinement interval. Require at least two contributing observations within the window for replication, and bound net growth by unresolved projected-gradient evidence (threshold 0.00125). Observations are not guaranteed to be distinct cameras. The cap is a limit, not a target to fill.
- Add a 0.001 mean-square prior on non-DC SH coefficients to constrain excessive view-dependent appearance without penalizing DC color.
- Exclude undefined undistortion rays from RGB loss and densification error evidence. Their camera-derived validity uses the existing packed alpha channel and creates no alpha/foreground loss. Valid black source pixels remain supervised. The current validity-only path applies to distorted, unmasked images rendered through the pinhole undistortion path; real foreground masks keep their previous behavior.

The new settings are selected by the `adc_igs` preset. Core changes are in `photara/src/splat/{densification.cpp,densification_adc_plus.cpp,cuda_ops.cu,trainer.cpp,training_data_loader.cpp}` and the corresponding public/internal headers. Existing unrelated workspace edits were preserved.

## Evaluation domains

Both reference and candidate PLYs are re-rendered with the same `photara_splat_quality_eval` executable, camera loader, rasterizer and SH degree. Primary PSNR/SSIM use the same camera-derived source validity mask for both models: approximately 98.08% of pixels are valid. The remaining pixels have no source observation after undistortion; their black fill is not measured scene color. Validity is determined by camera geometry, not by prediction quality or pixel brightness.

Full-frame PSNR/SSIM are also recorded. Ignoring undefined black borders during training can lower full-frame PSNR while improving measured source pixels. The report retains that regression rather than comparing the old full-frame score with the new valid-pixel score. Fixed central-90% PNG measurements are supplemental only; native float-render valid-pixel scores determine the comparison.

Midpoint-pose images provide qualitative novel-view checks with identical poses for both models. They have no reference photograph and are not assigned PSNR. The checks found that the black edge fragments introduced by fitting undistortion borders in the pre-validity candidate disappear in the revised candidate. Several views of the camera body, lens, strap and tablet were inspected. This does not establish a global floater count or prove clean geometry from arbitrary unseen viewpoints; there is no ground-truth surface for this scene.

## Validation and outputs

The Release CLI, editor, splat test and canonical evaluator were built successfully. The complete `photara_splat_test` passed; its final log is `artifacts/igs_quality_20260913/final_full_tests.log`. Added checks cover score accumulation, world/noise scale behavior, oversize evidence, SH gradients, growth budgets, observation support, and validity-mask cache/loss behavior. One earlier full-suite attempt had an asynchronous neighbour-supervision failure; subsequent full runs including the final run passed, and that existing synchronization code was not changed.

After CMake regeneration, all four targets were rebuilt and the full splat suite passed again (`artifacts/igs_quality_20260914_confirm/build_verify.log` and `splat_tests.log`). The exported full model's vertex count, binary payload size and all 66 float properties were checked: every parameter is finite (`final_full/validation.json`). `delivery_hashes.json` identifies the final build and retained benchmark binaries separately.

Comment text in four source files changed during the final rebuild, so the benchmark and delivery source hashes are retained separately. The rebuilt CLI/evaluator CPU code sections and CUDA binary sections match the retained benchmark binaries exactly (`rebuild_code_check.json`); binary file hashes differ in read-only metadata. The retained confirmed executables are available for reproducing the exact benchmark binaries.

## Serial confirmation results

All values below are means over the same 25 held-out views, measured by the common native evaluator.

| Run | Valid PSNR (dB) | Valid SSIM | Full PSNR (dB) | Full SSIM |
| --- | ---: | ---: | ---: | ---: |
| v1, repeat 1 | 34.660940 | 0.981701 | 21.388956 | 0.967781 |
| Revised, repeat 1 | 35.190852 | 0.982967 | 21.294732 | 0.968894 |
| v1, repeat 2 | 34.784968 | 0.981722 | 21.411168 | 0.967823 |
| Revised, repeat 2 | 35.169452 | 0.982884 | 21.291500 | 0.968809 |
| v1, mean | 34.722954 | 0.981712 | 21.400062 | 0.967802 |
| Revised, mean | 35.180152 | 0.982926 | 21.293116 | 0.968851 |
| Difference | +0.457198 | +0.001214 | −0.106946 | +0.001049 |

Mean final Gaussian counts are 464,287 (v1) and 712,274 (revised), approximately 53% more. Mean logged training times on this RTX 5090 D v2 are 129.809 s and 153.958 s, approximately 19% longer; these are the trainers' reported durations, excluding the separate canonical evaluation. Equal caps do not imply equal final model sizes. Exact counts and durations are in `run_summary.json`.

Repeat 1 improves valid PSNR on 22/25 cameras and valid SSIM on 24/25. The representative visual report uses repeat 1, selected by run order, not by the best score: `artifacts/igs_quality_20260914_confirm/report/comparison.html`. It includes all held-out views and available midpoint views. In midpoint view 40, a bottom-edge black shard visible in v1 is absent in the revised output. `report/novel_comparison.jpg` contains a compact comparison of views 40 and 120.

The full-198-camera model is `artifacts/igs_quality_20260914_confirm/final_full/reconstruction_splat.ply`: 706,717 Gaussians, 186,575,035 bytes, 30,000 steps, 137.815 s reported training time. Its cameras are training data, so its rendered scores are not reported as held-out validation. Exact command and model hash accompany the output. The CLI and evaluator used for confirmation are retained as `build/photara/Release/photara_igs_confirmed_20260914.exe` and `photara_splat_quality_eval_confirmed_20260914.exe`.

To repeat the comparison in a new output directory:

```powershell
python experiments/benchmark_igs_quality.py --dataset D:/ScanVideo/WeChat_20250712175936 --reference D:/SoftwareTool/photara-v1.0/photara.exe --candidate build/photara/Release/photara.exe --evaluator build/photara/Release/photara_splat_quality_eval.exe --output artifacts/igs_quality_repeat --iterations 30000 --cap 1000000 --repeats 2 --split 8
```

The evaluator is a CMake target alongside the splat tests (`PHOTARA_BUILD_TESTS` and `BUILD_TESTING` enabled). Rebuild with `cmake --build build --config Release --target photara photara_studio photara_splat_test photara_splat_quality_eval -j 12`.
