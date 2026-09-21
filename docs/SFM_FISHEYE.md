# SfM 鱼眼相机支持

> 等距柱状全景（360°）相机是另一条独立链路，见 [SfM 全景相机支持](SFM_EQUIRECTANGULAR.md)。

## 使用

编辑器的 **Camera Alignment → Camera model** 提供 **Auto / Pinhole / OpenCV Fisheye**，新项目默认 Auto。下方 **Result camera** 显示已加载或已完成对齐的实际类型；尚无结果时显示 Pending alignment。选择模式随 `.ascan` 项目保存，旧项目保留原来的显式选择。

命令行默认 `--camera-model auto`；C++ `FrontEndOptions` 默认仍为 Pinhole，以保留已有程序行为。

### 自动选择

自动模式复用原始特征匹配，在每个已有相机尺寸组中抽取最多 16 对图像、每对最多 400 个匹配点。每对匹配按固定索引分成训练集（2/3）和验证集（1/3）：训练集用于相对姿态估计和共享内参 BA，验证集只用于按原始图像像素误差评价候选。

对针孔和鱼眼分别搜索 8 个粗焦距，所有具备基本几何支持的粗初值都尝试共享内参 BA，并比较局部焦距搜索前后的标定，避免只取评分前三个初值而漏掉正确焦距。只在多对图像上有明确验证优势时确定类型。任何一种候选发现平面退化时，对该图像对做对称排除；比较分母只计入未排除的图像对，避免退化图像制造或稀释模型优势。明确提供 `--focal` 时保留该焦距，仍可估计畸变。验证集参与候选选择，最终质量须由独立图像序列和参考位姿验收。

共享内参 BA 后，另用新内参重新拟合各图像对的训练匹配，并保留验证得分更高的姿态候选。这样，被 BA 排除的图像对也有机会在新焦距与畸变下重新估计姿态，避免旧姿态压低正确标定的得分；验证匹配仍不进入姿态拟合。2026-09-10 的局部、全量及 COLMAP 对照见 [继续验证报告](SFM_CONTINUED_VALIDATION_20260910.md)。

EXIF 中明确包含 fisheye 的镜头描述仅作为辅助提示，仍须得到几何支持。证据不足或两种模型相近时回退到针孔，保留原焦距和零畸变初值，继续普通针孔视图图自标定，避免把歧义候选的极端焦距、畸变注入重建。日志的 `auto camera:` 行记录两种模型得分、最终类型、焦距和选择原因。这是有回退策略的启发式选择，不保证从任何图片中确定真实镜头类型；单张图、纯旋转、窄视场或经过校正的图片尤其容易产生歧义，可以手动覆盖。

实际 `.asfm` 场景只保存已经确定的投影类型，不会把 Auto 当作一种投影。选择逻辑也应用于匹配缓存和融合 LightGlue 路径。

命令行支持 `--camera-model opencv_fisheye`（简写 `fisheye`）：

```powershell
.\build\photara\Release\photara.exe `
  --images D:\ScanVideo\alameda\images_2 `
  --output artifacts\alameda_fisheye.asfm `
  --camera-model opencv_fisheye --mode global
```

上述命令可用于全量运行；最新验收包括完整 1,742 张原始鱼眼数据，详见质量验证报告。

显式选择鱼眼但未提供焦距时，也会执行上述多初值标定，仅评估鱼眼模型；无有效几何时才保留图像长边 0.5 倍的初值。已知焦距时用 `--focal <像素值>`，数值应对应实际输入图像分辨率。未锁定时全场景 BA 仍可优化焦距和畸变。`--trust-focal` 会固定焦距、主点和零畸变，只适用于已标定的等距鱼眼，不能代替一般鱼眼标定。

## 模型与数据兼容

采用 OpenCV 的四阶角度多项式：`theta_d = theta * (1 + k1*theta² + k2*theta⁴ + k3*theta⁶ + k4*theta⁸)`。零系数仍是等距鱼眼投影。

- CPU/CUDA BA 共用模型投影及解析导数；相对姿态与重定位使用鱼眼模型，跳过针孔焦距自标定。
- 保留 `PinholeCamera` 类型名以兼容现有代码，通过 `camera.model` 选择投影。鱼眼的 k3/k4 存放于历史 p1/p2 字段；这时它们不是切向畸变。诊断 CSV 附加 `camera_model` 列以说明参数含义。
- `.asfm` 当前写入版本 3（颜色为增量字段）；旧针孔文件仍可读取。鱼眼文件要求 reader 2，避免旧版误读为针孔。检查点 schema 为 8；几何缓存键包含相机模型与实现指纹。
- 内部 MVS 和纹理仍把源图像去畸变到针孔工作相机。Gaussian splat 训练默认在原始鱼眼/全景图像上光栅化（OpenCV 鱼眼多项式或等距柱状全景），不再先拉成针孔；`--splat-undistort` 可恢复旧的鱼眼→针孔路径。直接 `.mvs` 导出暂不支持未校正的鱼眼照片，会明确报错；请保存为 `.asfm` / `.ascan`。
- 当前沿用 SfM 的正深度约束，支持前向半球内的单调鱼眼标定；不覆盖超过 180° 的后向光线。

## 历史验证（2026-09-08，改进标定前）

下面保留早期结果用于比较，不能代表当前实现。2026-09-09 的多初值标定与独立子集验收见 [标定质量验证](SFM_CALIBRATION_VALIDATION_20260909.md)。

用户提供的 `D:\ScanVideo\alameda\images\_2` 不存在，实际使用 `D:\ScanVideo\alameda\images_2`。源目录共有 1,742 张 JPG，按文件名排序取前 64 张（DSC06436.JPG 起），在 `artifacts/fisheye_alameda_64/images` 创建硬链接，不修改源图像。

```powershell
.\build\photara\Release\photara.exe `
  --images artifacts\fisheye_alameda_64\images `
  --output artifacts\fisheye_alameda_64\global.asfm `
  --camera-model opencv_fisheye --mode global --max-features 6000 --window 6
```

| 模式 | 对齐图片 | 稀疏点 | 平均重投影误差 | RMS | 耗时 |
|---|---:|---:|---:|---:|---:|
| Global | 64/64 | 26,204 | 0.779770 px | 0.931284 px | 12.31 s |
| Incremental | 42/64 | 15,196 | 1.146110 px | 1.433810 px | 10.79 s |

这是实际图像子集上的可运行性和重投影检查，不是全量验收或有真值的位姿精度评估。输出和日志位于 `artifacts/fisheye_alameda_64/`。

自动化测试覆盖光轴及接近 90° 入射角的投影/反投影、鱼眼相对/绝对姿态、BA 点坐标及全部内参导数的有限差分、CPU/CUDA 一致性与收敛、场景/项目/缓存往返，以及 MVS 源模型传递和错误导出保护。

### 自动模式验收

同一 64 张子集，`--camera-model auto --mode global --max-features 6000 --window 6`：抽取 8 对匹配，针孔得分 0.523489、鱼眼得分 0.589544，差距不足以确定模型，按规则回退针孔并保留 2102.4 px 初始焦距。最终对齐 63/64，22,884 个稀疏点，平均误差 0.778313 px、RMS 0.934591 px，总耗时 11.15 s，其中模型比较约 0.67 s。结果位于 `artifacts/fisheye_alameda_64/auto.asfm`。

启用同一个缓存再次运行，命中 tracks/reconstruction，实际类型和上述结果一致，耗时 0.225 s。这组数据未获得明确的自动分类结论；可继续手动选择鱼眼，不能把回退结果视为真实镜头类型标签。

新增合成验证覆盖：未知焦距鱼眼识别、普通广角的保守选择、已知焦距下的针孔识别、无几何证据回退，以及 Auto 项目设置的保存/加载。对应 two_view/project.io 测试通过，CLI 和 editor 编译成功。
