# WeChat 3DGS 渲染器迁移质量复核

数据：`D:/ScanVideo/WeChat_20250712175936/sparse/0`，图像来自同目录的 `images`。198 张图、22,229 个初始点，1280×720。源码基线为 `dda519d`，本次修改尚未提交。

## 对比方法

- ADC+，30,000 步，初始点上限 500,000，增密上限 10,000,000，SH 3。
- 关闭 progressive resolution、mask、depth-normal、MV geometry、MV NCC，匹配数据目录中历史 RGB 训练日志的关键设置。
- 每隔 8 张留出 1 张：173 张训练、25 张测试，固定默认 seed 42。每个后端重复两轮。重复相同种子用于观察 CUDA 数值与增密轨迹波动，不代表多随机种子统计。
- 参考适配层来自迁移前提交 `d52919c21152599685edb56fc4ef586f964f887f` 的 `rasterizer.cu`，仅增加可选梯度导出。它与新后端链接相同的当前训练器和图像加载器，隔离 CUDA 后端差异。
- 额外测试保存于 2026-09-11 的 `aetherscan_validation.exe`。该文件没有可确认的源码提交标记，不能将其视为已验证的特定提交构建。
- GPU 为 RTX 5090 D v2；训练串行执行。全部结果写入 `artifacts/splat_quality_20260913`，未覆盖原始数据。

## 训练结果

PSNR 是各留出视图 PSNR 的算术平均，单位 dB。mask 关闭，因此此处 `average_psnr` 与 `average_masked_psnr` 相同。

| 版本 | 第一次 | 第二次 | 均值 |
|---|---:|---:|---:|
| 当前训练器 + gggs_reference | 32.9062 | 33.0617 | 32.9840 |
| 修复前 splat_drender | 33.0086 | 33.0219 | 33.0153 |
| 最终修复后 splat_drender | 32.9671 | 32.9046 | 32.9359 |

最终版相对同训练器参考均值为 **-0.0481 dB**，相对旧模型统一评估结果为 **+0.0196 dB**。这些微小差异不支持声称 PSNR 显著提高，也不足以证明系统性质量回归；本次确认的改进是实际梯度缺陷修复。两轮重复不足以排除更小的统计效应。

旧保存程序训练出的模型，旧评估器报 32.7652 dB，同一个模型经当前评估器重算为 **32.9163 dB**。评估器/图像加载差异不能混入后端训练质量的比较。将修复前新模型交给参考后端重渲染，25 视图仍为 33.0219 dB，与新后端自己的评估一致到日志精度。

另外，使用全部 198 张图训练、只抽查 3 个训练视图的一轮结果为：参考 32.1539、新后端修复前 32.1200 dB，差 -0.0339 dB。这小于参考后端两轮留出集自身的 0.1555 dB 波动，不能单独作为迁移回归证据。第一轮留出集中单个视图最多下降 1.1577 dB，但第二轮同一视图仅下降 0.0531 dB；平均值不代表每张图均有提升。

## 实际定位并修复的问题

### 1. 极薄高斯的前后向 alpha 不一致

从 30,000 步模型导出的 671,010 个高斯中抽取编号 25584，单独重放即可复现。前向颜色、alpha 几乎相同，但旧 bucket backward 的 SH DC 梯度与 `C0 * sum(alpha * upstream_color)` 不一致，refine 相对误差约 **0.00914255**。这不是训练随机性，也不是大量高斯求和顺序造成的：单个高斯也发生错误。

高斯二次型中的大数相消会放大编译器 FMA 收缩/CSE 顺序的差异。最终实现通过显式 CUDA 舍入指令固定二次型，按参考前向 PTX 的顺序：x 项乘法与 y 平方项做一次 FMA，之后单独计算乘 -0.5 和交叉项减法。像素 RGB、median traversal、bucket backward 共用这一实现。

### 2. 无几何监督时仍执行病态几何反传

编号 344928 的最薄轴约 `3.337e-8`，轴比约 147,700。新旧后端都会在没有几何上游梯度时执行逆协方差/法线代数，产生 NaN，污染 RGB 的 means、scale、rotation 梯度。优化器的非有限值保护只能避免写坏参数，不能使这些梯度变正确。

现在当 ray-plane 与 normal 上游梯度全部为零时，跳过这条数学上严格为零的分支，保留 RGB/协方差梯度。该大场景中的新后端非有限梯度已消除；没有为追求参考相等而复制参考 NaN。

## 验证与边界

- `aetherscan_splat_test` 完整通过，包括新增真实薄高斯夹具、颜色梯度与前向 alpha 的恒等关系、梯度有限性，以及原有 RGB/alpha/depth/normal 有限差分与优化器测试。
- 最终固定输入 RGB/geometry A/B：两组均通过原有 rtol=1e-4 / atol=1e-5 门槛，未放宽阈值。几何 opacity 相对误差约 1.26e-5。
- 671k 高斯大场景：实例数同为 1,155,522，前向颜色相对误差 3.39e-7；新后端非有限梯度为 0，参考 means/scale/rotation 分别有 3/3/4 个。refine 相对误差从 2.718e-3 降至 1.436e-4，仍略超 1e-4，故这一组不是严格全通道通过。
- 不宣称所有病态高斯都能达到 `1e-4` 的参考相对误差。参考本身存在 NaN，极端条件数会放大舍入误差，必须区分数学正确性、参考一致性和完整训练质量。
- 本次完整训练验收是该数据集的 RGB ADC+；未执行开启 mesh、多视图几何损失或 fisheye 的长程训练。相关单元/有限差分测试不等价于那些模式的完整训练验收。

## 复现和产物

```powershell
python experiments/compare_splat_quality.py `
  --new build/aetherscan/Release/aetherscan.exe `
  --reference build/aetherscan/Release/aetherscan_reference_current.exe `
  --dataset D:/ScanVideo/WeChat_20250712175936 `
  --output artifacts/splat_quality_repeat --iterations 30000 --split 8 --repeats 2
```

脚本保存每轮命令、二进制 SHA256、逐视图 PSNR、PLY 和 PNG。默认均值回归门槛 0.2 dB 是可调整的工程门槛，不是统计显著性标准。参考程序构建使用本机现有 reference 静态库；本次适配代码、链接命令、日志、原始二进制哈希已保存在结果目录，换机器需要相应调整路径并重建。

- `comparison.json`：最终汇总。
- `quality_comparison.png`：训练曲线与逐视图比较。
- `visual_comparison.html`：按最终版第一轮 PSNR 差异选出最差、中间、最好视图的并排渲染。
- `canonical/run{1,2}_splat.ply`：最终修复版留出集训练模型。
- `new_train_splat.ply`：修复前全视图训练模型，仅用于迁移诊断。
- `canonical_splat_test.log` 与 `canonical_{rgb,geometry}/summary.json`：最终测试日志。

已重新编译命令行 `aetherscan.exe` 与编辑器 `aetherscan_editor.exe`，均位于 `build/aetherscan/Release`。
