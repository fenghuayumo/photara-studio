# Intrinsic / Delight 模型许可边界

`--delight`（图像域去光照）使用 **Intrinsic** 的 `stage_0.onnx` … `stage_3.onnx`
权重，通过 C++ ONNX Runtime 推理（`photara/src/texture/delight.cpp`）。

## 现状

- 仓库**不分发**这些权重。请自行获取并放到 `PHOTARA_INTRINSIC_MODELS_DIR`
  （默认 `${CMAKE_BINARY_DIR}/Models/Intrinsic`，见 `cmake/FetchIntrinsicModels.cmake`）。
- 上游（`compphoto/Intrinsic`）按项目记录为**学术 / 非商用**许可。
  该结论来自项目内部记录，**不是法律意见**；以商业化产品发布前，
  必须由法务或负责人对照上游仓库的 LICENSE 与模型卡逐条复核。
- 未放置权重时 `--delight` 会报错；此时改用 `--texture` 仍可得到投影烘焙结果。

## 影响面

- 该许可只约束 Delight 权重与推理，不改变仓库其余部分（SfM / MVS / Splat /
  aether_drender 贴图）的边界；
- 导出产物只有在显式使用 `--delight` 时才包含由这些权重生成的 albedo；
- GGGS 参考 CUDA rasterizer 的许可边界另见
  `photara/third_party/gggs_reference/NOTICE.md`，
  fused SSIM 移植见 `photara/third_party/fused_ssim/LICENSE`。
