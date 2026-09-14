# 室内重建：哪套相机更可信？

结论：确实存在 AetherScan 明显更好的照片，也存在已有重建更好的照片。不能把“偏离已有重建”直接当成错误，也不能把 739/739 注册或更低的自身重投影 RMS 当成全面胜出的证据。

## 对象与方法

- 已有模型：`D:/BaiduNetdiskDownload/室内iphone/sparse/0`，COLMAP 格式，738 张相机，222484 个点；生成算法未知，不是真值。
- AetherScan：`iphone_quality.asfm`，27000 特征配置，739 张相机，643244 个点；新匹配测试同时加入 `iphone_verified_sfm_diagnostics.csv` 的 6000 特征配置。
- 没有修改相机、点云或照片，也没有将参考模型喂入重建。

### 双向留一视图三角化

分别使用两套模型自己的特征轨迹作为两个观测库。对每个目标像素，确定同一组另外三张照片，以两套相机分别重新三角化，再预测目标照片上的像素；每张最多随机抽取 400 条轨迹，固定种子，保留失败预测，检查目标和三个锚点的正深度。无位姿对齐、无参考异常点剔除。

这是从**三角化中**留出目标视图；两套模型原来的相机优化仍可能使用过这些像素，因此不是严格训练外验证。使用自身轨迹也有筛选偏差，必须同时看两个观测库。

| 观测库 | 检查数 | 已有模型预测中位误差 / px | AetherScan 27000 预测中位误差 / px |
|---|---:|---:|---:|
| 已有轨迹 | 276644 | 1.227 | 1.547 |
| AetherScan 轨迹 | 281594 | 1.051 | 0.911 |

总体上各自在自己的轨迹库更好，所以不能用这两个整体平均数简单宣布获胜者。但个别照片有双向一致证据：DCIM5783.jpg 在已有轨迹库上为 5.71 / 0.91 px，在 AetherScan 轨迹库上为 99.56 / 0.56 px（已有 / AetherScan），都支持 AetherScan。

### 原照片重新匹配

为进一步避免只依赖已有轨迹，从原照片重新运行 OpenCV CPU SIFT（最多 8000 特征），双向匹配、0.7 ratio test，然后用独立的 USAC MAGSAC F 模型筛选（2 px）。相机参数不参与匹配或筛选。

选取此前差异大的六张照片，合并两套轨迹各自最强的三个邻居，得到 32 对；28 对满足共同注册和至少 30 个独立 F 内点。对相同匹配，分别计算两套模型的去畸变像素 Sampson 极线距离。它衡量图像对几何一致性，不是 3D 位置误差，也不是新视角渲染质量；新提取特征仍可能与原训练特征重合，不能称为严格未见测试集。

下表为各邻居对的**中位极线距离范围**，单位 px，越小越好；不是置信区间。

| 目标照片 | 对数 / 匹配数 | 已有模型 | AetherScan 6000 | AetherScan 27000 | 证据解释 |
|---|---:|---:|---:|---:|---|
| DCIM5783.jpg | 5 / 4225 | 4.58–23.22 | 0.17–0.48 | 0.16–0.34 | 多邻居一致支持 AetherScan |
| DCIM5833.jpg | 5 / 7397 | 3.77–8.13 | 0.19–0.44 | 0.19–0.23 | 多邻居一致支持 AetherScan |
| DCIM5630-HDR.jpg | 4 / 1770 | 0.70–15.18 | 0.24–0.81 | 0.24–0.58 | 27000 配置四对均更好 |
| DCIM5012-HDR.jpg | 5 / 3609 | 0.23–0.35 | 2.11–5.13 | 2.18–27.39 | 多邻居一致支持已有模型 |
| DCIM6228.jpg | 5 / 1340 | 0.67–22.14 | 3.60–28.31 | 2.14–16.80 | 组内与跨组证据冲突 |
| DCIM5732.jpg | 4 / 350 | 0.20–4.55 | 2.32–5.93 | 0.20–3.59 | 证据混合，支持仍较少 |

具体而言，DCIM6228 与 DCIM6226 的距离为已有 22.14 / AetherScan 27000 2.14 px，但与 DCIM6172 为已有 1.72 / AetherScan 14.35 px。这提示局部变焦组内部可以自洽，但跨区域约束仍不正确；还不能仅凭这项检查确定单张相机或整组的唯一故障来源。

DCIM5012 的 27000 配置比 6000 更差也说明：增加特征、点数以及降低整体 RMS，不能保证每张相机的位置更正确。

## 对之前“质量门槛”的修正解释

原来的参考相机 Sim(3) 差异只能用作**差异报警**。其中最大异常 DCIM5783 的新匹配证据反而强烈支持 AetherScan，不应继续把它直接记为 AetherScan 的质量失败。应优先修复同时被独立像素证据判差的照片，例如 DCIM5012 和部分跨组连接。

本轮新匹配只覆盖选出的六个问题视图，不能据此给整套模型一个绝对排名。判断完整产品效果还需要更广泛的独立匹配覆盖，或真实测量/严格留出的图像、下游新视角渲染测试。

## 复现与验证

脚本：`experiments/compare_sfm_evidence.py`、`experiments/fresh_sfm_pair_audit.py`。完整数据在 `artifacts/sfm_model_comparison_20260913/evidence.json` 和 `fresh_both.json`，包含每张相机/每对图像指标及跳过原因。

几何实现通过合成验证：带畸变的正确相机对应极线误差最大约 5.7e-13 px，故意偏移像素后误差上升；留一视图的已知三维点预测正确，完全相同的两个模型输出一致。

```powershell
python experiments/compare_sfm_evidence.py --reference 'D:\BaiduNetdiskDownload\室内iphone\sparse\0' --asfm artifacts/sfm_quality_20260913/iphone_quality.asfm --output artifacts/sfm_model_comparison_20260913/evidence.json
python experiments/fresh_sfm_pair_audit.py --reference 'D:\BaiduNetdiskDownload\室内iphone\sparse\0' --asfm artifacts/sfm_quality_20260913/iphone_quality.asfm --images 'D:\BaiduNetdiskDownload\室内iphone\images' --extra-diagnostics artifacts/sfm_quality_20260913/iphone_verified_sfm_diagnostics.csv --output artifacts/sfm_model_comparison_20260913/fresh_both.json
```
