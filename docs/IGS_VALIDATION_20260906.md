# IGS / ADCPlus 重建验证（更新：2026-09-08）

## 当前方案

IGS 与 ADCPlus 分别由 `IgsStrategy`、`AdcPlusStrategy` 管理，位于各自的
`densification_igs.hpp/.cpp`、`densification_adc_plus.hpp/.cpp`。
共用 GPU 拓扑修改和 Adam 状态映射；CUDA 分裂实现仍放在公共 `cuda_ops.cu`。

IGS 保留 ADCPlus 的贡献可见性、证据剪枝、屏幕覆盖修正采样、透明度衰减与
透明度分裂规则，以及相同的初始化、优化器和训练日程。新增两点：

1. 增长候选排除已被 replacement / oversized 选中的父点，避免重复消耗预算。
2. 沿协方差最大特征轴进行对称分裂，只缩小这一轴。设原最大方差为 λ，
   子点缩放为 k，则偏移为 ±sqrt((1-k²)λ)。因此子高斯协方差加上均值偏移
   外积等于原协方差，等权混合分布的中心与完整协方差保持不变。
   ADCPlus 同时沿多个局部轴偏移，会引入非零的轴间协方差。

这是归一化高斯混合的一、二阶矩保持，并不表示 alpha 合成后的图像完全不变。
对超大投影仍复用 ADCPlus 的屏幕阈值限制；垂直于分裂轴的尺度保持不变。

## 公平对比设置

实际 Train 路径：`D:/Models/tandt_db/tandt/train/sparse`。
最初给出的 `D:/Models/tandt/_db/tandt/train/sparse` 不存在。
已核对 Train COLMAP 输入哈希，和 9 月 6 日基线一致。

- 同一份 300 相机 COLMAP 模型和原始图片，按文件名排序。
- 每 8 个视角留出 1 个：262 个训练视角，38 个留出视角。
- 10,000 步，最长边上限 1280；Train 原图为 980×545，实际保持原尺寸。
- 两者同为 2,000,000 高斯上限，固定分辨率，无 mask，纯外观 3DGS。
- 每个运行使用新进程，CPU 和 Tensor 随机数默认 seed=42。
- CUDA 原子运算仍有非确定性；相同设置复测存在波动。
- PSNR 为程序在未参与训练的视角上计算的平均值。
- 补充 SSIM 来自导出 8-bit PNG 与双线性缩放后的 pinhole 原图，11×11
  Gaussian 窗口、sigma=1.5、总体协方差；与训练用浮点 fused SSIM 不是同一实现。

此处 ADCPlus 是用户工作区已有的、加入证据剪枝和 footprint weighting 的版本。
不能将结果解释为对所有外部 ADCPlus 实现的比较。

## 试验记录

| 方案 | Train PSNR | 处理 |
| --- | ---: | --- |
| 仅消除重复采样，3k / 800 | 20.2281 | 对应 ADCPlus 20.2133，差异太小 |
| 持续观测增长权重，10k | 21.5844 | 对应 ADCPlus 21.5890，撤回 |
| 梯度辅助替换，10k | 21.5435 | 撤回 |
| 增长权重 1.5 次幂，10k | 21.6305 | 对应当日 ADCPlus 21.6569，撤回 |
| 最大轴矩保持分裂，10k | 21.9449 | 保留并进行配对复测 |

第五版与当日 ADCPlus 比较：PSNR +0.2880 dB；SSIM 0.82771 对 0.82382。
38 个视角中 29 个 PSNR 提高；最差视角下降 1.1992 dB，并非每个视角都改善。
已检查最差视角 216 的两张渲染，仍需以逐视角结果评估局部差异。

第五版完整日志与模型：`artifacts/igs_train_10000_v5`。
对应基线：`artifacts/igs_train_10000_v3/train_adc_plus`。
清理后的最终配对复测：`artifacts/igs_train_10000_confirm`。

| 最终配对复测 | ADCPlus | IGS | 差值 |
| --- | ---: | ---: | ---: |
| PSNR (dB) | 21.6098 | 22.0317 | +0.4219 |
| PNG SSIM | 0.822546 | 0.827438 | +0.004892 |

两次 IGS / ADCPlus 对比的 PSNR 提升分别为 +0.2880 和 +0.4219 dB，
SSIM 也都提高。复测中 30/38 视角 PSNR 提高，最差局部差值为 -1.3194 dB。
这些是固定 seed 的重复运行，不能替代多个不同 seed 的统计检验。

最终模型：`artifacts/igs_train_10000_confirm/train_adc_igs/reconstruction_splat.ply`。
模型重新加载后，全部 38 个留出视角渲染成功，平均 PSNR 仍为 22.0317 dB。
重载日志位于 `train_adc_igs_reload/run.log`。

## 验证和复现

Release 构建、`aetherscan_splat_test.exe` 通过。
新增测试覆盖：已选 oversized 父点不会耗掉另外两个增长槽位；高斯与 Adam
行数一致；分裂结果有限；IGS / ADCPlus 日程一致；分裂保持中心和完整 3×3
协方差。`git diff --check` 通过。

```powershell
python experiments/compare_igs.py --scenes train --iterations 10000 --resolution 1280 --output artifacts/igs_train_new_run
python experiments/summarize_igs.py artifacts/igs_train_new_run --scenes train
python experiments/igs_image_metrics.py artifacts/igs_train_new_run --scene train
```

脚本按运行保存命令、程序 SHA-256、日志、模型、留出渲染和结果 JSON，拒绝覆盖
已有的同名运行目录。可用 `--strategies adc_igs` 只跑 IGS，用 `--baseline` 指向
另一个运行目录分析配对结果。

## NYC 状态

3k / 800 初步结果：ADCPlus 23.5873，早期 IGS 23.6379 dB（124 个留出视角）。
这不是当前第五版的结果。
10k NYC 最终评估未完成；重载模型后评估曾触发异常巨大尺寸的 CUDA 分配请求，
尚未定位，因此不宣称 NYC 达标。按用户最新要求，先确保 Train。

目前是在同一留出集上选择方案，没有独立第三个测试集或多随机种子的统计证明，
结论只适用于上述数据、配置和已完成运行。
