# SfM 自动标定与重建质量验证（2026-09-09）

本轮修复自动相机选择的焦距局部最优、针孔歧义候选污染内参，以及全局重建最终 BA 固定短迭代问题。所有 Photara 实拍运行均使用 `--camera-model auto --mode global --max-features 6000 --window 6`，未输入参考焦距、位姿、特征或三维点。

## 标定策略

每个相机组取最多 8 对原始匹配，每对最多 400 点，固定拆成 2/3 训练、1/3 验证。相对姿态和共享内参 BA 只使用训练点；验证点按原生像素重投影评价，从而避免不同模型、焦距的归一化误差不可比。每种投影模型尝试 8 个粗焦距，保留最多 3 个相隔足够远的初值，继续局部搜索和焦距/畸变联合优化。

必须有多对图像的明确验证优势才能自动确定模型。歧义时回退针孔，保留普通针孔自标定路径，不传递歧义候选的极端焦距和畸变。手动鱼眼、未给焦距时也估计标定；手动选择受到尊重。Editor 的 Auto / Pinhole / OpenCV Fisheye 选择及 Result camera 显示继续可用。

全局最终 BA 先运行 8 次；如达到迭代上限且最后 3 次成本仍下降至少 0.1%，追加最多 3 轮、每轮 16 次优化。已收敛或改善很小的场景停止，避免无条件扩大所有数据的计算预算。

## 数据和参考来源

- Alameda 原始鱼眼：`D:/ScanVideo/alameda/images_2`，1,742 张，1752×1168。用户最初的 `images/_2` 路径不存在。
- 前 64 张与紧接的 128 张为互不重叠子集，使用文件硬链接，不修改源照片。
- 两个子集分别新跑 COLMAP 4.1.0（fa8e3b3 CUDA），OPENCV_FISHEYE，共享相机，SIFT 6000/0.005，穷举匹配，CPU mapper 8 线程、seed 0。COLMAP 的初始焦距为 876；最终标定不传入 Photara。
- 普通针孔控制：`D:/ScanVideo/chuan/images`，109 张，参考为已有 `D:/ScanVideo/chuan/sparse/0`。
- 全量参考：已有 `D:/Models/alameda/sparse/0`，1,734 张，是相同拍摄序列的去畸变版本，不能称为本轮新跑的原始鱼眼 COLMAP 全量结果。仅去掉 `indoor_` 文件名前缀，旋转、平移原样保留。原参考相机为 PINHOLE 2789×1586；其 `images_2` 图像为 1394×793。人工核对首帧场景内容，并检查所有规范化文件名唯一且存在于输入目录。

全量参考与本轮独立 COLMAP 的前 64 / 后 128 张交叉检查：位置 P95 分别为 0.468% / 0.310%，旋转 P95 为 0.295° / 0.121°，支持将其用作全量位姿参考，但它不是测量真值。参考转换脚本、原 images.bin SHA256 和逐文件对应关系保存在 `artifacts/sfm_product_validation_20260909/alameda_existing_reference/` 及相邻 `prepare_reference.py`、`reference_crosscheck.json`。

## 评价方法

使用全部同名已注册相机估计一个正尺度、不含反射的 Sim(3)，不删除误差大的相机。位置差异除以参考相机中心的 RMS 半径，表示为百分比；旋转差异为角度。这个百分比不是厘米误差或整体质量百分比。重投影 RMS 来自每个系统自身观测，观测集合不同，不能单独用它判定系统优劣。

工程门槛：注册率 ≥98%、参考覆盖率 ≥98%、全部已注册相机通过可观测性检查、重投影 RMS ≤1 px、位置 P95 ≤5%、旋转 P95 ≤3°、每相机至少 30 个有效观测。门槛通过不等于普遍产品级认证；还必须查看最大误差和局部序列。

## 已验证的标定修复（v4，最终 BA 延长前）

| 数据 | 注册 | 参考覆盖 | 位置 P95 / 最大 | 旋转 P95 / 最大 | 重投影 RMS |
|---|---:|---:|---:|---:|---:|
| Alameda 前 64 | 64/64 | 64/64 | 0.192% / 0.287% | 0.069° / 0.098° | 0.631 px |
| Alameda 独立 128 | 128/128 | 128/128 | 0.834% / 0.916% | 0.219° / 0.242° | 0.597 px |
| Chuan 针孔 | 109/109 | 109/109 | 0.562% / 0.667% | 0.296° / 0.323° | 0.548 px |
| Alameda 全量 | 1741/1742 | 1734/1734 | 0.200% / 2.838% | 0.172° / 3.229° | 0.707 px |

独立 128 为 v3 结果；v4 仅修改歧义针孔回退，不影响该数据的明确鱼眼分支。全量 772,363 点、3,335,968 有效观测，最少每相机 86 个观测，耗时 347.7 s，峰值工作集约 5.24 GiB。一次并行执行了小子集/测试，时间仅作运行记录，不作为严格速度基准。

早期前 64 张自动模式的位置 P95 差异为 145.6%、旋转 P95 为 176.0°；显式鱼眼但默认焦距时为 88.6% / 26.7°。本轮改进来自独立标定，没有把 COLMAP 解直接导入。历史数据见 [首次 COLMAP 对照](SFM_COLMAP_COMPARISON_20260908.md)。

全量尾部集中在 DSC07012–DSC07020 的局部片段，和 DSC08184。唯一未注册图片为 DSC08185，缺少独立深度约束，参考模型也没有它。已注册的 7 张额外图片无参考位姿，不能声称其绝对精度已验证。v4 尾部结果推动了最终 BA 收敛修复；最终版本复测结果在下节记录。

## 最终版本复测

v5–v9 的实测产物已经保留；2026-09-10 继续验证及最终重跑见 [继续验证报告](SFM_CONTINUED_VALIDATION_20260910.md)。该报告补充固定相机集合对照、原始鱼眼 COLMAP 参考交叉检查、全量结果和仍未解决的局部退化，不能再把本节的旧“运行中”状态当作最新进展。

## 复现与证据

实拍日志、ASFM、逐相机诊断、逐对诊断、指标 JSON 均位于 `artifacts/sfm_product_validation_20260909/`；前 64 的新 COLMAP 基准位于 `artifacts/colmap_comparison_64_20260908/opencv_fisheye/sparse/0`，独立 128 的新基准位于本轮目录 `colmap_holdout128/0`。

```powershell
python experiments/run_sfm_acceptance.py --exe build/photara/Release/photara.exe --images D:/ScanVideo/alameda/images_2 --reference artifacts/sfm_product_validation_20260909/alameda_existing_reference --output-dir artifacts/sfm_product_validation_20260909 --name alameda_repeat --timeout-seconds 1800 -- --camera-model auto --mode global --max-features 6000 --window 6
python experiments/plot_sfm_comparison.py --diagnostics artifacts/sfm_product_validation_20260909/alameda1742_v4_sfm_diagnostics.csv --reference artifacts/sfm_product_validation_20260909/alameda_existing_reference --output artifacts/sfm_product_validation_20260909/alameda1742_v4.png
```

绘图脚本同时导出全部同名相机的误差 CSV，可检查任意尾部，不局限于前十个位置误差。新增测试覆盖未提供焦距的显式鱼眼标定、歧义针孔不注入畸变、错误长度匹配的保守回退。前 64 的缓存重放逐行诊断与原运行完全一致。
