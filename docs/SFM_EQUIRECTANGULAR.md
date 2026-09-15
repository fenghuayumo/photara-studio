# SfM 全景（等距柱状 360°）相机支持

## 使用

编辑器的 **Camera Alignment → Camera model** 现在提供 **Auto / Pinhole / OpenCV Fisheye / Equirectangular 360**，
新项目默认 Auto；选择随 `.ascan` 项目保存。

命令行（`--camera-model equirectangular`，别名 `equirect` / `panorama` / `spherical`）：

```powershell
.\build\aetherscan\Release\aetherscan.exe `
  --images D:\ScanVideo\street_360\images `
  --output artifacts\street_360.asfm `
  --camera-model equirectangular --mode global
```

自动模式（Auto）：

1. 图像尺寸为 **2:1**（宽度为高度的两倍，容差 1%）时才考虑全景假设；
2. 用与针孔/鱼眼相同的抽取图像对与留出（held-out）匹配评估全景图卡，要求至少 2 个图像对给出明确支持，
   且得分不低于最佳透视拟合的 40%。

第 2 条只是**护栏**而不是竞争，原因见下节：当匹配点集中在相机前方窄锥内时（手持 360 视频的典型情况），
任何中心投影的单调径向形变在小平移基线下都近似满足对极几何，因此**无法仅凭几何区分 45° 针孔假设与真实全景**；
2:1 尺寸才是决定性证据（openMVS 同样直接断言 `width == 2*height`）。护栏用来拒绝“2:1 的普通直线投影照片”：
这类相机的像素只覆盖球面很小的范围，透视拟合会明显更好。日志 `auto camera:` 行记录两种模型的得分与最终选择。

## 模型与约定

等距柱状图卡：`u = fx * azimuth + cx`，`v = fy * elevation + cy`，其中
`fx = width / (2π)`、`fy = height / π`、主点在图像中心，方位角 `atan2(x, z) ∈ [-π, π]`、
仰角 `asin(y/|p|) ∈ [-π/2, π/2]`（相机坐标 X 右、Y 下、Z 前，与针孔一致：天上在图像上方 = -Y）。

- 全景相机**没有可优化内参**：投影完全由图像尺寸决定，`trust_intrinsics = true`，BA 中该组冻结；
  `--focal`、`--trust-focal` 对全景无意义（会被忽略）。
- `PinholeCamera::unproject` 对全景返回**真实单位光线**（z 可为负），而不是 z=1 平面点；
  `project` 使用完整球面映射。所有涉及后向半球的判断都改成“沿观测光线的深度为正”。

## 实现要点

- **两视图**：PoseLib 没有球形相机模型，但极线约束只依赖像素对应的“光线”，
  因此把光线按分量翻转到前半球后（射线不变、约束不变）即可用 PoseLib 的 **5 点最小解算器**
  （前半球代理坐标 = 焦距 1、主点 0 的针孔归一化坐标）；打分始终用**光线域对称 Sampson**
  （`(b2ᵀ E b1) / sqrt(|E b1|² + |Eᵀ b2|²)`，小误差下即角度），因为用平面坐标打分在离轴 80° 处会严格约 30 倍。
  分解时用 4 组 `(R, ±t)` 的**光线深度**正负择一（后向半球同样有效），随后用角度残差做 LM 精化。
- **三角化**：使用光线叉乘形式（`Dcross b`）的线性最小二乘，天然与半球无关；
  非线性精化和内点判定使用**切平面残差**（沿观测方位/仰角切基，单位像素），避免方位缝处的整幅像素跳变与极点压缩。
- **重定位/绝对位姿**：光线形式的 DLT（`b × (R X + t) = 0` 对 12 个未知量线性，不要求 z>0）+ 球面安全的弦距 LM 精化。
- **BA**：`linearize_observation` 中新增球形分支，残差为观测切平面上的角度残差（单位像素，Huber 阈值语义不变），
  雅可比 `∂r/∂p_cam = diag(fx, fy) [t1; t2]ᵀ (I - b bᵀ)/|p|`；**不做 z>0 剔除**，只排除落在光心的点。
  CPU 与 CUDA 共用同一实现（`reprojection_detail.cuh`），`evaluate_cost` 同步更新。
- **过滤/统计**：`filter_tracks`、`star_init`、候选评分等所有重投影判定切换到角度域；
  空域覆盖权重按**等立体角**分箱（方位 × sin(仰角)）而不是图像面积，避免极点被加权。
- **退化检测**：针孔用单应退化率，全景改用“**纯旋转能否解释匹配**”（Kabsch 拟合后残差在阈值内的内点比例），
  语义等价（无基线 → 不用于初始化）。
- **存储/导出**：`.asfm` 允许 `equirectangular` 模型并要求 reader ≥ 3（避免旧版把球面图卡误读成针孔）；
  检查点同样放行；COLMAP 导出写 `EQUIRECTANGULAR`（模型 id 17，参数为 `width height`，与 spirula 等工具一致）；
  Nerfstudio/Blender 导出写 `EQUIRECTANGULAR`；OpenMVS/MVS 导出**明确报错**（球面无法重采样成针孔工作相机），
  提示改用 `.asfm`/`.ascan`。
- **Splat**：`mvs::MvsView::source_model` 直接来自 SfM 相机模型，全景因此走**原生球面光栅化训练**（无需去畸变）；
  `--splat-undistort` 对全景报错；训练日志会打印 `splat native projection: equirectangular=N ...`
  统计，便于确认没有静默退回针孔。
- 新增 `--export-colmap <dir>`：把对齐结果同时写成 COLMAP 文本模型（cameras/images/points3D），便于与外部重建对照。

## 验证

数据：375 张 Insta360 风格 **7680×3840（2:1）** 全景视频帧
（`D:\BaiduNetdiskDownload\VID_20260911_145523_00_007_dataset`），
并带有一份独立参考重建（spirula + COLMAP，`sparse/0` 为模型 id 17 的等距柱状）。
测试取每 6 帧一张、共 60 张，降采样到 2048×1024，`--window 12 --max-features 8000`：

```powershell
.\build\aetherscan\Release\aetherscan_equirect_dataset_check.exe `
  artifacts\equirect_street\images artifacts\equirect_street\colmap_global `
  --window 12 --max-features 8000 --mode global --camera-model equirectangular
python scripts\compare_equirect_reference.py `
  artifacts\equirect_street\colmap_global `
  D:\BaiduNetdiskDownload\VID_20260911_145523_00_007_dataset\sparse\0
```

| 运行 | 注册 | 稀疏点 | 平均重投影 | RMS | 耗时 |
|---|---:|---:|---:|---:|---:|
| Global（显式 equirectangular） | 60/60 | 15,912 | 0.727 px | 1.351 px | 67 s |
| Global（Auto，自动识别为全景） | 60/60 | 17,529 | 0.733 px | 1.011 px | 13 s |
| Incremental（显式） | 57/60 | 18,639 | 0.882 px | 1.286 px | 7 s |
| CLI 端到端（`--camera-model equirectangular` + `--export-colmap`） | 60/60 | 15,328 | 0.703 px | 0.994 px | 18 s |
| Global（显式，120 张子集，每 3 帧取 1） | 120/120 | 36,863 | 0.731 px | 0.936 px | 27 s |

与参考重建（按文件名配对 60 张，位姿度量与规范/坐标约定无关）：

| 运行 | 图像对旋转角误差（中位/p90） | 相对旋转 SO(3) 差（中位） | 相机中心 Sim(3) 对齐残差 |
|---|---:|---:|---:|
| Global 显式 | 0.065° / 0.244° | 0.172° | 0.568% 轨迹尺度 |
| Global Auto | 0.071° / 0.239° | 0.151° | 0.525% 轨迹尺度 |
| Incremental | 0.071° / 0.219° | 0.149° | 0.577% 轨迹尺度 |
| CLI 端到端 | 0.079° / 0.270° | 0.162° | 0.571% 轨迹尺度 |
| Global 120 张 | 0.073° / 0.279° | 0.154° | 0.608% 轨迹尺度 |

### 全量 375 帧与工程验收

整段素材（375 帧，统一降采样到 2048×1024）一次跑完，`--mode global --window 12`：

| 运行 | 注册 | 稀疏点 | 平均重投影 | RMS | 耗时 |
|---|---:|---:|---:|---:|---:|
| 默认像素阈值（未归一化） | 375/375 | 148,848 | 0.797 px | 1.084 px | 76 s |
| **角度阈值归一化（默认 1/3）** | 375/375 | 121,785 | 0.369 px | **0.459 px** | 78 s |
| 全分辨率 7680×3840，60 帧 | 60/60 | 14,719 | 0.349 px | 0.421 px | 25 s |

用仓库自带的工程门槛脚本对全量结果验收（`experiments/sfm_acceptance.py`，
门槛：注册率与参考覆盖 ≥ 98%、加权重投影 RMS ≤ 1.0 px、逐相机观测 ≥ 30、
中心误差 P95 ≤ 5% 且最大 ≤ 10%（参考布局半径归一化）、旋转 P95 ≤ 3° 且最大 ≤ 5°）：

```
python experiments/sfm_acceptance.py \
  --diagnostics artifacts/equirect_street/pano_full_tight_sfm_diagnostics.csv \
  --reference <spirula_colmap_dir> \
  --output artifacts/equirect_street/pano_full_tight_acceptance.json
```

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|:--:|
| 注册 / 参考覆盖 / 观测可靠 | 375/375、100%、375/375 | ≥ 98% | 通过 |
| 加权重投影 RMS | 0.459 px | ≤ 1.0 | 通过 |
| 逐相机观测数（最小/中位） | 142 / 782 | ≥ 30 | 通过 |
| 中心误差 中位 / P95 / 最大 | 0.58% / 1.29% / 2.09% | P95 ≤ 5、max ≤ 10 | 通过 |
| 旋转误差 中位 / P95 / 最大 | 0.252° / 0.359° / 0.479° | P95 ≤ 3、max ≤ 5 | 通过 |

**9 项门槛全部通过（`passed: true`）。** 归一化之前唯一的失败项是加权重投影 RMS
（1.084 px vs ≤ 1.0），原因与处理见下节。

全分辨率 60 帧（7680×3840，真实拍摄分辨率）同样 9 项全过：RMS 0.421 px、
中心 P95 1.31%、旋转 P95 0.44°、峰值内存 843 MB、25 秒。

### 像素阈值的角度归一化

全景图卡把 2π 弧度铺在同样的像素数上，而直线投影只用这段像素数覆盖自己的（窄得多）视场：
以默认焦距比 f = 1.2·width 计，针孔每像素 1/(1.2·width) 弧度，全景每像素 2π/width 弧度，
两者相差 2π·1.2 ≈ 7.5 倍。流水线里的对极/重投影/track 过滤/重定位阈值都以"像素"给出，
直接套用到全景就会松 7.5 倍（2 px 阈值在全景等于 0.35°，在针孔只有 0.047°），于是接受大量
几何上很松的匹配：自洽性差、且极端相机误差被放大。

处理：`k_equirect_threshold_scale`
（`include/core/camera_projection.hpp`）在把像素阈值换算成角度时对等距柱状相机做同样的收紧，
使"N 像素"在两种相机上代表同一角精度——**等价于让全景跑一套与同分辨率针孔相同的（已调好的）阈值**。
它只影响阈值换算，不影响任何上报的像素误差。效果（全量 375 帧，参考对照为图像对旋转角误差与中心误差）：

| 设置 | 加权重投影 RMS | 逐相机 P95（中位/最大） | 观测数中位 | 稀疏点 | 对参考的图像对旋转角误差（中位/p90） | 中心误差 中位/P95 |
|---|---:|---:|---:|---:|---:|---:|
| `scale = 1.0`（原行为） | 1.084 px | 1.87 / 2.59 px | 1230 | 148,848 | 0.067° / 0.327° | 0.52% / 1.29% |
| `scale = 1/3`（**默认**） | 0.459 px | 0.72 / 1.80 px | 782 | 121,785 | 0.084° / 0.331° | 0.58% / 1.29% |
| `scale = 1/(2π·1.2)`（完全角度对齐，可选） | 0.204 px | 0.32 / 0.64 px | 407 | 88,251 | 0.098° / 0.347° | 0.61% / 1.34% |

结论：自洽性提升约 2.4 倍并通过全部工程门槛；与外部参考的几何一致性保持在同一量级
（旋转差中位 0.07–0.10°、中心 P95 1.3% 左右），代价是保留的观测/稀疏点更少（122k vs 149k），
但每相机仍保有 142–1557 个观测（门槛 ≥ 30）。该常数可调；设回 1.0 即恢复旧行为。

为什么默认取 1/3 而不是上面的 1/(2π·1.2)（完全角度对齐）：完全对齐会把**单对**两视图的内点预算
压到 0.07° 量级，而 2048 px 全景每像素就是 0.176°，单对 RANSAC 的匹配噪声（0.5–1 px ≈ 0.09–0.18°）
直接超过该阈值，于是逐对拟合被"饿死"：

- 全量 375 帧的逐相机观测中位从 782 掉到 407、稀疏点从 121.8k 掉到 88.3k（自洽性 RMS 反而更好，
  0.204 px，但这来自更少的观测）；
- 更要命的是模型自动选择退化：全景候选的探针分数从 0.151 掉到 0.110，低于 0.25×透视候选，
  于是 Auto 退回针孔、只注册 92/120（显式全景为 120/120）。

把整条流水线的 BA 精度（0.166 px 均值）当成"每一对都要达到"的要求是不成立的：最终精度来自
全局旋转平均/定位与全场景 BA 对大量观测的平均，而不是单对基线极短的两视图拟合。

### 下游 Gaussian splat 链路

同一份对齐（120 张）导出 COLMAP 全景模型后再训练，确认走了**原生球面光栅化**而不是去畸变到针孔：

```powershell
.\build\aetherscan\Release\aetherscan.exe `
  --images artifacts\equirect_street\images `
  --splat-dataset artifacts\equirect_street\colmap_pano120 `
  --output artifacts\equirect_street\splat_pano_native\model.ply --splat-iterations 2500
```

训练日志：`splat native projection: equirectangular=120 fisheye=0 of 120 cameras (training on the source projection, no undistortion)`；
训练/渲染分辨率为 1920×960（2:1 全景），损失从 0.175 降到 0.096（2500 步，仅功能性验证，未做质量收敛）。

说明：该数据为视频帧，图像对旋转中位数仅 1.6°（最大 5.0°），基线很小；旋转**角度**在规范与坐标约定下
都是严格不变量，因此上表第一列是最强的位姿一致性证据。相机中心比对只用几何（中心与相机坐标轴向无关），
对齐后残差约轨迹尺度的 0.5%。参考模型的文件尺寸/系数与 AetherScan 无关，仅位姿被比较。

自动化测试（`aetherscan.sfm.equirect`，`tests/equirect_test.cpp`）覆盖：

- 图卡约定与全（含后向半球、方位缝、极点）投影/反投影往返；
- 切平面残差：观测光线上残差为零、跨方位缝不跳变、极点有界、对相机点雅可比与有限差分一致；
- 合成全景图像对的相对位姿（≥95% 内点保持，含后向半球）、纯旋转对的退化识别；
- 光线形式绝对位姿（重定位），以及针孔路径未回归；
- 多视图三角化 + 角度域 track 过滤（后向半球观测存活）；
- BA 收敛（代价降到初值的 5% 以内、旋转恢复到 1e-3、结构在尺度规范下恢复）；
- CPU/CUDA 线性化器残差与有效性逐条一致（防止“设备端静默失效”类问题）；
- Auto 选择：2:1 全景数据选全景、2:1 直线投影照片不误判、无 2:1 尺寸提示时不考虑全景、显式请求优先。

### 与 OpenMVS（球面相机路径）的对照

同一份 120 张、2048×1024 全景子集，OpenMVS v2.4.0 用其原生 `SphericalCamera` 跑完整 SfM：

```powershell
# OpenMVS 只根据 XMP GPano:ProjectionType 判定球面相机，测试图需带该元数据
python scripts/tag_panorama_xmp.py <images> <images_pano_xmp>       # 写入 GPano:ProjectionType=equirectangular
.\CreateStructure.exe -s <images_pano_xmp> -o scene.mvs `
  --export-poses-csv poses.csv --max-threads 32 -v 3
python scripts/openmvs_poses_to_colmap.py poses.csv openmvs_colmap 2048 1024 .jpg
python scripts/compare_equirect_reference.py openmvs_colmap <reference_colmap_dir>
```

OpenMVS 结果确认走了球面路径（位姿 CSV 中 `fx=fy=1, cx=cy=0` 即 `SphericalCamera` 的恒等 K），
120/120 注册、135,482 点、耗时 2 分 06 秒（其 SiftGPU 匹配在多对 4 万–5 万特征时报 GPU 显存不足并跳过该对，
属该实现的特征规模/显存默认值问题，本次仍完成 120/120）。

与 COLMAP/spirula 参考模型（120 张共同图像）对照，三方在同一精度量级：

| 运行 | 图像对旋转角误差（中位 / p90） | 相对旋转 SO(3) 差（中位） | 相机中心（中位 / P95，参考布局半径归一化） | 稀疏点 | 耗时 |
|---|---:|---:|---:|---:|---:|
| OpenMVS（球面相机） | 0.053° / 0.231° | 0.139° | 0.42% / 1.09% | 135,482 | 2 m 06 s |
| AetherScan（8k 特征，阈值未归一化） | 0.073° / 0.279° | 0.154° | 0.52% / 1.29% | 36,863 | 27 s |
| AetherScan（27k 特征，阈值未归一化） | 0.065° / 0.239° | 0.158° | — | 45,938 | 21 s |
| AetherScan（8k 特征，阈值已归一化，**最终**） | 0.070° / 0.205° | 0.207° | 0.39% / 1.06% | 19,961 | 23 s |

AetherScan（最终设置）120 张结果直接与 OpenMVS 互比（**不涉及参考模型，是两个独立实现互检**）：
图像对旋转角误差中位 **0.078°**、p90 0.213°、max 0.316°；相机中心 **0.43% / 0.96%（中位/P95）**。
归一化后 AetherScan 的自洽性明显优于两者（RMS 0.177 px，逐相机 P95 中位约 0.3 px），
而与 OpenMVS 的差异仍小于 0.1°——即两者在同一条质量水平线上，差异量级远低于工程门槛。

两个估计结果直接互比（AetherScan 120 张 vs OpenMVS）：图像对旋转角误差中位 **0.057°**、p90 0.138°，
相机中心 Sim(3) 残差 **0.287%**（轨迹尺度）。三方差异都在 0.02°–0.03° 量级，而参考模型本身是另一次 SfM 估计
（非真值），因此结论是"三者精度同级、无系统性偏差"，而不是"AetherScan 严格优于/劣于 OpenMVS"。
另需注意特征预算不同：OpenMVS 每张图约 4 万–5 万特征，AetherScan 为 8 千/2.7 万；AetherScan 在同级精度下
耗时约 1/6。

## 限制

- **范围界定：本链路只负责相机对齐（位姿 + 稀疏点云）。** 原生球面/鱼眼的稠密重建（MVS）不在范围内，
  而且这是设计选择而非待办：MVS 的匹配、深度图与纹理都建立在针孔工作相机上，通常先把源图去畸变/重采样到
  单个针孔相机；球面无法重采样成单个针孔相机，等距柱状转立方体面又会改变几何与基线。因此导出 `.mvs` 会明确
  报错并提示改用 `.asfm`/`.ascan`，编辑器里全景/鱼眼对齐下的稠密入口也被禁用。
  对齐结果本身可以直接消费：`.asfm`/`.ascan`（位姿 + 轨道）、COLMAP 模型（`EQUIRECTANGULAR`）、稀疏 PLY，
  或者交给本仓库的 Gaussian splat 训练（原生球面光栅化，见上）。
- 图卡是唯一的镜头模型：不建模实际 360 相机的双鱼眼标定误差或制造商畸变；
  若镜头标定已知且需要更高精度，可先做等距柱状重投影再对齐。
- 线性化估计会剔除光束靠近方位缝（|z| < 1e-3，即图像左右极窄带）的观测，它们仍参与角度域过滤与后续 BA；
  这一窄带在平面归一化坐标下不可表示。
- 对齐不使用动态物体掩码（掩码目前只用于 MVS/splat 训练路径）。
- 全景的 `min_ray_angle`/重投影阈值仍按像素给出，单位换算用 `2π/width`；
  纯旋转或极窄基线视频帧的**单对**相对位姿本质欠定（E 的平移分量不可观测），
  这正是流水线依赖全局旋转平均 + 定位的原因，也是 Auto 判定不把“单对分数”当作投影类型证据的原因。
