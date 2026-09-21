# SfM 半分辨率与全分辨率对比回归（2026-09-06）

本轮应要求测试 `D:/Models/nyc/images_2` 与 `D:/Models/alameda/images_2`，随后追加
`D:/Models/alameda/images` 全分辨率对照。请求中的
`images/_2` 目录不存在；实际目录为 `images_2`，是与 `images` 同文件名、同数量的
半分辨率版本（nyc 990 张、alameda 1734 张，约 1393x793）。SHA-256 审计确认两套
输入无重复文件。评价口径沿用 2026-09-05/06：相邻 `sparse/0` 的 COLMAP 模型仅为
对照基线而非测量真值；所有公共相机一次正尺度、无反射 Sim(3)，不删除离群点；
门槛为注册率>=98%、参考覆盖>=98%、重投影 RMS<=1 px、位置 P95<=参考半径 5%、
旋转 P95<=3 度、每相机有效观测>=30、全部注册相机对齐可观测。

参考模型由全分辨率图像构建并全部注册（nyc 990/990、alameda 1734/1734）；位姿
对比不依赖内参，因此对半分辨率输入仍有效。环境：Windows，Ryzen 9 7950X、32 GB
RAM、RTX 5090 D v2，Release 构建，改进后默认配置（不传人工焦距，global 模式，
SiftGPU + gpu_mutual_ratio，最大 27,000 特征，window=3，BoW 检索）。冷启动无缓存；
热启动复用冷启动缓存目录，命中 tracks 与 reconstruction checkpoint。

## 半分辨率结果

| 运行 | 注册 | 重投影 RMS / px | 位置 P95 / % | 位置最大 / % | 旋转 P95 / ° | 墙钟 / s | 峰值内存 / MB | 缓存 | 质量门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| nyc_halfres_auto_final | 988/990 | 0.649 | 0.386 | 7.39 | 0.182 | 166.5 | 2440.9 | 无 | 仅对齐可观测性失败 |
| nyc_halfres_warm | 988/990 | 0.649 | 0.386 | 7.39 | 0.182 | 5.9 | 552.7 | tracks+reconstruction | 与冷启动一致 |
| alameda_halfres_auto_final | 1730/1734 | 0.620 | 0.254 | 11.46 | 0.188 | 326.2 | 4034.1 | 无 | 仅对齐可观测性失败 |
| alameda_halfres_warm | 1730/1734 | 0.620 | 0.254 | 11.46 | 0.188 | 14.2 | 1228.3 | tracks+reconstruction | 与冷启动一致 |

两组数据的注册率、参考覆盖、重投影、位置 P95、旋转 P95、相机支持门槛全部通过。
唯一失败项是逐相机对齐可观测性：nyc 有 2 台（DSC02604/02605）、alameda 有 1 台
（DSC08109）桥接相机被保守标记不可靠。nyc 未注册 2 张（DSC02536/02537，连续帧，
稳定 resection 尝试后未接受）；alameda 未注册 4 张（DSC06842/08041/08110/08184）。
热启动输出与冷启动逐位一致，checkpoint 回放具有确定性。

冷启动阶段拆分：nyc 前端 85.1s（提取 30.3s、检索 10.0s）、全局定位 23.8s、映射
共 76.5s；alameda 前端 183.8s（提取 55.6s、检索 9.9s、匹配 59,472 对）、全局定位
50.6s、映射共 133.1s。折合冷启动吞吐约 5.9 张/s（nyc）与 5.3 张/s（alameda）。
逐相机 P95 重投影中位数约 1.44-1.49 px、最大约 1.85 px（2 px 内点阈值内的尾部），
全观测加权 RMS 为 0.62-0.65 px。

## 结构审计

- nyc：单一连通分量，2 条桥（02604-02605、02604-02606）；主块 986 台，02604/02605
  各自成割点块。02605 位置误差最大 7.39%，与不可靠标记一致。
- alameda：单一连通分量，2 条桥（08107-08108、08108-08109）。此前全分辨率的
  DSC07344-07359 十六相机悬挂支路本次全部进入主块，位置误差中位 0.276%、最大
  0.364%，旧支路问题在半分辨率输入上未复现。08109 实际误差仅 0.240%，保守标记
  在该相机上偏严但正确反映弱连接。
- alameda 新的最弱局部为 DSC07006-07025：位置误差中位 9.56%、最大 11.46%，但旋转
  误差仅约 0.3 度，形态符合低视差走廊中沿视线方向中心弱约束；整体 P95 门槛通过，
  产品输出仍应暴露该局部置信度。

## 全分辨率对照

同一构建、同一默认配置对 `D:/Models/alameda/images`（1734 张，约 2789x1586）冷启动
重跑。结果与 2026-09-05 的全分辨率运行（1701/1734、P95 3.08%、最大 66%）基本一致，
证明旧弱支路问题在全分辨率输入上是系统性复现，不是随机波动。

| 运行 | 注册 | 重投影 RMS / px | 位置 P95 / % | 位置最大 / % | 旋转 P95 / ° | 墙钟 / s | 峰值内存 / MB | 验证对 | 质量门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| alameda_fullres_auto_final | 1702/1734 | 0.599 | 3.13 | 60.96 | 0.568 | 973.9 | 7310.3 | 10,004 | 仅对齐可观测性失败 |
| alameda_halfres_auto_final | 1730/1734 | 0.620 | 0.25 | 11.46 | 0.188 | 326.2 | 4034.1 | 11,726 | 仅对齐可观测性失败 |

全分辨率热启动回放 27.9s，输出与冷启动逐位一致。全分辨率下最终注册审计剔除 26
台支持不足相机（多为 0 观测），稳定 resection 补回部分末端相机后为 1702 台；
32 张未注册；28 台桥接相机被标记不可靠；图审计为 2 个连通分量（主块 1673 +
15 相机独立块）加 12 个单点块、14 条桥。

逐区域对比（同一参考、同一 Sim(3) 口径）：

| 区域 | 全分辨率位置误差 | 半分辨率位置误差 |
|---|---|---|
| DSC07344-07359（旧 16 相机支路） | 中位 41.11%、最大 55.68% | 中位 0.28%、最大 0.36% |
| DSC07518-07524 | 中位 46.03%、最大 60.96% | 中位 0.15%、最大 0.24% |
| DSC07006-07025（半分辨率最弱走廊） | 中位 0.39%、最大 37.34%、3 张未注册 | 中位 9.55%、最大 11.46% |

内参不是全分辨率失败原因：焦距交叉检验一致（Fetzer 1208.92、Q75 1219.43，取
1208.92，与默认 1.2x 边长初值 3346.8 不同但两者一致故采用）。真正差异在前端视图
图：全分辨率仅 10,004 条验证对，半分辨率 11,726 条（多约 17%）；弱支路在全分辨率
下被孤立为 15 相机独立块，在半分辨率下全部进入主块。更密的每图特征并未带来更密的
可用配对图，弱纹理支路在 ratio/mutual 筛选与几何验证下反而更难连回主体。

结论：当前默认管线对 alameda 的弱支路失败模式仍存在于全分辨率输入；半分辨率的
通过不能外推为"该数据集已完全修复"。在修复视图图连通性（例如分辨率自适应检索/
匹配参数或跨分辨率配对融合）之前，全分辨率 alameda 不能按产品级交付，只能作为
可检测的风险场景输出（28 台不可靠相机与 15 相机独立块均已被审计正确标记）。

### nyc 全分辨率对照

nyc 全分辨率（990 张，约 2787x1586）与半分辨率结论相反：全分辨率不仅没有退步，
精度反而更高。首次运行因磁盘耗尽在 checkpoint 写入时失败（保留为
`nyc_fullres_auto_final`，exit_code=1，"Checkpoint write failed"；当时 D 盘仅剩
0.10 GB）。清理 alameda 全分辨率与失败残留缓存后以 `nyc_fullres_auto_final_r2`
冷启动重跑成功。本轮会话外部还删除了 alameda 的 `images_0/_4/_8` 目录（非本会话
操作），计划的更低分辨率扫描因此未执行；nyc 的 `images_4/_8` 仍存在但未使用。

| 运行 | 注册 | 重投影 RMS / px | 位置 P95 / % | 位置最大 / % | 旋转 P95 / ° | 墙钟 / s | 峰值内存 / MB | 验证对 | 质量门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| nyc_fullres_auto_final_r2 | 989/990 | 0.618 | 0.133 | 7.64 | 0.123 | 554.1 | 4322.9 | 6,467 | 仅对齐可观测性失败 |
| nyc_halfres_auto_final | 988/990 | 0.649 | 0.386 | 7.39 | 0.182 | 166.5 | 2440.9 | 7,213 | 仅对齐可观测性失败 |

nyc 全分辨率唯一未注册 DSC02536（半分辨率为 02536/02537 两张），唯一不可靠相机
仍为单桥悬挂的 DSC02605（位置 7.64%、旋转 3.40 度），逐相机最少观测从半分辨率
的 50 升至 146。nyc 的验证对同样随分辨率升高而减少（7,213 -> 6,467），但图结构
保持健康（仅 1 条桥），说明"全分辨率配对图稀疏化"本身不必然破坏对齐；alameda
的失败是配对稀疏化与弱纹理支路叠加的结果，属于数据集相关失败模式。nyc 全分辨率
热启动回放 28.3s，与冷启动逐位一致。

## 结论边界

对齐质量在两组半分辨率数据上已进入产品级区间：>99.7% 注册、亚像素重投影、
位置 P95 < 0.4% 参考半径、旋转 P95 < 0.2 度，1734 张冷启动 5.4 分钟。但全分辨率
alameda 复现 40-61% 的支路漂移，产品级结论必须按输入分辨率区分。本机未安装
RealityScan 或 COLMAP CLI，未做同机同数据对拍，因此不能据本轮宣称"达到 RealityScan
同等质量/性能"；该结论需要同一数据、同一硬件的 RealityScan 对照运行。剩余产品级
缺口：少量失败帧（2/990、4/1734）与桥接相机的产品化置信度输出、局部弱约束区域
提示、重复性与多初值统计、混合传感器/变焦/低视差泛化，以及下游 MVS/渲染质量
验证。本轮未执行 MVS 或 splat 训练。

## 复现

```powershell
python experiments/run_sfm_acceptance.py `
  --exe build/photara/Release/photara.exe `
  --images D:/Models/nyc/images_2 --reference D:/Models/nyc/sparse/0 `
  --output-dir artifacts/sfm_acceptance_halfres_20260906 --name nyc_rerun `
  --timeout-seconds 1800

python experiments/run_sfm_acceptance.py `
  --exe build/photara/Release/photara.exe `
  --images D:/Models/alameda/images_2 --reference D:/Models/alameda/sparse/0 `
  --output-dir artifacts/sfm_acceptance_halfres_20260906 --name alameda_rerun `
  --timeout-seconds 1800

python experiments/run_sfm_acceptance.py `
  --exe build/photara/Release/photara.exe `
  --images D:/Models/alameda/images --reference D:/Models/alameda/sparse/0 `
  --output-dir artifacts/sfm_acceptance_halfres_20260906 --name alameda_fullres_rerun `
  --timeout-seconds 3600

python experiments/sfm_graph_audit.py `
  --diagnostics artifacts/sfm_acceptance_halfres_20260906/nyc_rerun_sfm_diagnostics.csv `
  --pairs artifacts/sfm_acceptance_halfres_20260906/nyc_rerun_sfm_pairs.csv `
  --output artifacts/sfm_acceptance_halfres_20260906/nyc_rerun_graph_audit.json
```

证据位于 `artifacts/sfm_acceptance_halfres_20260906/`：每次运行的 `.run.json`、
控制台日志、模型、逐相机/配对诊断、图审计、重复文件审计与对比图
（`nyc_halfres_comparison.png`、`nyc_fullres_comparison.png`、
`alameda_halfres_comparison.png`、`alameda_fullres_comparison.png`）。构建指纹：
git `c7e8ea08b6bac870785db0806fceff61ab7eee07`，
photara.exe SHA-256
`255bace6fb8cbfb917905282b589e9ac2614fdcc6114e6b956ad9693648ba9ae`。

验证：ctest 相关 10/10 通过（ba.optimizer、features、sfm.two_view/mapping/
vocabulary/checkpoint/submap_recovery/hierarchical/export_mvs、project.io）；
Python 验收 5/5、图审计 3/3 通过。
