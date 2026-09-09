# 原生鱼眼 splat 前后向修复

## 范围

修复 `CameraModel::opencv_fisheye` 的 CUDA 渲染、深度采样和训练辅助路径。
相机模型仍为正 Z 半球的 OpenCV 四系数鱼眼；没有扩展为超过 180° 的镜头模型。
本次没有修复 equirectangular 的协方差/几何公式。

## 修复内容

- `fisheye_geometry.h` 统一计算投影 Jacobian、二维协方差、抗锯齿透明度补偿、
  径向深度平面和相机法线。对相机空间位置和世界空间对称协方差的九个独立变量
  使用 CUDA 前向模式自动微分，补全位置经投影 Jacobian 影响协方差的二阶导数。
  这里的自动微分使用解析运算规则，不在训练中运行数值差分。
- 鱼眼 backward 传播透明度补偿的协方差导数；针孔分支原有 detach 策略保留。
  光轴附近采用 Taylor 展开，避免半径为零和小半径相消。
- 鱼眼深度平面使用投影 Jacobian 的切平面逆映射，替代针孔的直接除焦距。
  输出 median depth 仍为相机 Z 深度；其径向距离转换与 backward 均使用实际鱼眼射线。
- 深度采样返回真实射线上的三维点，并补全对查询点位置的逆投影导数。
- 鱼眼 alpha 达到 0.99 上限时，停止传播被截断的 opacity/conic 梯度。
- 原生鱼眼不再采用针孔 `Z > 0.2` 裁剪；保留 `Z > 1e-6` 数值有效性条件。
- 非 Brush 的鱼眼 3D filter 使用实际投影 Jacobian 的最大奇异值估计局部像素尺度。
  相机模型和畸变参数显式传入，不再通过 `width/fx` 猜测鱼眼。
- 原生相机不直接复用针孔 MVS 深度、法线图。额外关闭仍依赖针孔几何的 normal-field
  训练损失，并记录提示；已有 depth-normal 和多视图损失的保护保留。

## 验证

新增测试覆盖：

- 非单位相机旋转、各向异性高斯、非零 k1–k4。
- 普通离轴、接近视野边缘且 Z 小于 0.2、光轴中心、alpha 饱和四类场景。
- kernel size 为 0 和 0.3。
- 独立双精度 CPU 参考校验投影协方差对应的 alpha、法线和单高斯解析 median depth。
- RGB、alpha、深度、法线对位置、log-scale、原始四元数、opacity logit、SH 的梯度。
- 深度采样对模型参数与世界空间查询点的梯度。
- 总计 560 个标量参数数值差分检查；几何输出开启/关闭时纯 RGB 梯度一致性检查。
- 鱼眼滤波尺度，以及相同尺寸的 MVS 图不能自动成为鱼眼监督的回归检查。

已通过以下命令：

```powershell
cmake --build build --config Release --target aetherscan_splat_test -j 1
build/aetherscan/Release/aetherscan_splat_test.exe --fisheye-only
ctest --test-dir build -C Release -R '^aetherscan.splat.rasterizer$' --output-on-failure
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/compute-sanitizer/compute-sanitizer.exe' --tool memcheck --error-exitcode 1 build/aetherscan/Release/aetherscan_splat_test.exe --fisheye-only
```

完整 splat 测试通过；Compute Sanitizer 报告 `ERROR SUMMARY: 0 errors`。

## 使用边界

二维高斯投影和深度平面仍是局部线性近似，离散 tile/可见性变化与 median-depth
数值求解也不是处处光滑函数。测试选取固定有效像素，避开支持域切换。
如需使用 MVS 深度/法线或多视图几何约束，仍须实现对应的鱼眼重投影与遮挡处理，
不能仅因宽高相同就启用。没有进行真实数据集的长时质量或吞吐基准。
