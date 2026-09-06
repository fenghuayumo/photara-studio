# SfM 改进与新数据回归（2026-09-06）

本轮用 `D:/ScanVideo/ori_img/images` 替换 lego1。用户消息中的 `D:/ScanVideo/ori/_img/images` 在本机不存在；找到的实际目录为 `ori_img/images`，共 76 张 1000×1000 图片，SHA-256 审计确认 76 张全部不同。参考为相邻 `sparse/0`，是 COLMAP 对照模型，不是测量真值。沿用所有公共相机一次正尺度、无反射 Sim(3)，不删除离群点的验收口径。

## 已修复的新数据问题

旧默认路径把自动焦距初值 1200 px 改到约 524 px，得到全部注册、亚像素重投影但严重错误的轨迹。另一次诊断运行选择约 626 px，也失败。这表明图自标定目标的内部极小值同样可能退化，不能只拒绝搜索边界上的解。

现在交叉检查 Fetzer 图估计与已有两视图 Q75 估计。两者相差超过已有两视图支持半径（1.5 倍）时，不将任一冲突值当作测量覆盖初值，保留可由 BA 优化的初值，并输出 `conflicting_estimators` 警告。未知内参仍必须完成严格几何过滤；没有锁定焦距、使用参考位姿或从参考导入标定。原先针对图目标边界退化的 Q75 回退保持原有行为。

这是保守的一致性保护，不是完整可观测性或多初值选择方案。两个估计一致也可能同时错误；保留的默认初值同样不保证正确，仍需数据回归和失败反馈。

| ori 运行 | 注册 | 重投影 RMS / px | 位置 P95 / 参考半径 % | 旋转 P95 / ° | 墙钟 / s | 缓存命中 | 质量门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ori_baseline，修复前默认 | 76/76 | 0.782 | 94.61 | 82.34 | 33.30 | 无 | 失败 |
| ori_prior_control，锁定默认 1200 的诊断对照 | 76/76 | 0.584 | 0.80 | 0.36 | 25.44 | 无 | 通过 |
| ori_auto_improved，改进后默认 | 76/76 | 0.555 | 0.64 | 0.18 | 31.20 | 无 | 通过 |
| ori_auto_final，最终默认（关闭实验补配） | 76/76 | 0.556 | 0.65 | 0.18 | 34.82 | 无 | 通过 |

`ori_prior_control` 的 1200 来自默认 `1.2 × 图像边长`，不是可信的外部标定；该运行只用于定位原因。改进后的默认运行不传 `--focal` 或 `--trust-focal`。其 Fetzer 值约 560、Q75 值约 1355，触发冲突保护后使用可优化的默认初值，最大相机位置误差约为参考半径的 0.69%。

图：`artifacts/sfm_acceptance_20260906/ori_before_after.png`。左侧旧默认轨迹错误，右侧新默认与参考接近重合。两者均使用所有对应相机，未去除离群点。

最终配置对比图为 `ori_final_original_reference.png`。中断期间源参考目录在 10:30 新增了 `images.bin` 等文件，脚本默认优先读二进制，造成午后首次评估切换参考。因此将原有 `images.txt`、`cameras.txt` 固定到 `reference_ori_original_text/`，记录 SHA-256，并用同一文本参考重评上述四项输出。正式比较使用 `*_original_reference.run.json`；原始运行和首次评估报告原样保留。最终 ori 的正式位置 P95 为 **0.65%**、旋转 P95 为 **0.18°**，不是新二进制参考下的 0.20% / 0.50°。重评没有重跑重建。

chuan 最终默认回归 `chuan_auto_final`：109/109 注册，重投影 RMS 0.567 px，位置 P95 0.54%，旋转 P95 0.17°，全部质量门槛通过；无缓存墙钟 43.78 秒。`chuan_auto_improved` 的此前运行也通过。新数据和 chuan 的最终默认运行均未传人工焦距。

## 支路补匹配改进

原有弱视图扩展只按验证配对的度数筛选，会漏掉内部高连接、外部单桥连接的支路。实验筛选在不使用世界位姿或旧三维点的配对图上计算去桥主分量，将主分量外相机纳入补匹配。大场景继续使用有上限的时序和检索候选，先消耗可能连接主体的候选，再补支路内部候选。

结构风险相机允许用原有 rescue 匹配配置重试此前失败的主匹配，已验证配对不重复加入。补配结果仍通过几何验证和后续校准后的严格筛查，不因为新加了一条图边就宣称相机坐标正确。原有独立子图恢复入口及写回门槛保持有效。

**Alameda 实验失败，因此该选项默认关闭。** CLI 仅在显式传入 `--sfm-structural-rescue` 时启用；选项参与 geometry/tracks 缓存指纹，避免实验结果与默认结果混用。默认只启用本轮焦距一致性保护，仍使用原有按低度数触发的渐进补配。

`alameda_structural_expansion` 从无缓存完整运行：新增补配尝试 9153 对，接受 564 对；注册 1695/1734，重投影 RMS 0.599 px，整体位置 P95 2.14%、最大误差 58.67%，旋转 P95 0.50°。注册覆盖和结构门槛失败。不能把整体 P95 通过视作支路修复。

原 `07344–07359` 的 16 相机支路全部注册，但位置 P95 从旧对照的 55.85% 变为 58.14%，最大从 56.24% 变为 58.67%（均用全体公共相机拟合的变换和全体参考半径归一化）。更关键的是，这 16 个相机的结构可靠标记从 0 个变成 16 个：新增图边掩盖了风险，却没有消除几何偏移。`branch_comparison.json` 保存这项反例，不将补匹配实验发布为默认修复。

独立恢复工具新增可选的历史风险图片名单：`aetherscan_sfm_realign checkpoint.bin NEW_OUTPUT_DIRECTORY ADDITIONAL_RISK_NAMES.txt`。名单仅将相机加入待验证集合，不能提供坐标或提升主体可靠性，用于防止风险仅因图中增加一个环而被遗忘。

`alameda_independent_after_expansion` 对补配后的模型带入原支路名单进行独立验证：16/16 局部注册，4303 个局部点，223 个边界匹配，仅 2 个双侧独立共享点（旧实验为 0）。返回 `insufficient_independent_shared_points`，没有接受坐标写回。相邻 `branch_names.txt` 仅包含图片名；输入模型和 COLMAP 参考坐标不参与独立相似变换拟合。该结果再次说明支路还没有修复。

```powershell
build/aetherscan/Release/aetherscan_sfm_realign.exe `
  artifacts/sfm_acceptance_20260906/cache/alameda_structural_expansion/reconstruction-e901675f28dfa481.bin `
  artifacts/sfm_acceptance_20260906/alameda_independent_rerun `
  artifacts/sfm_acceptance_20260906/branch_names.txt
```

## 验证与复现

证据位于 `artifacts/sfm_acceptance_20260906/`，每个运行保留 `.run.json`、控制台日志、模型和逐相机/配对诊断。`ori_input_audit.json` 保留重复文件检查，`improved_build.json` 保留改进版执行文件指纹及替换后的三组数据路径。

```powershell
python experiments/run_sfm_acceptance.py `
  --exe build/aetherscan/Release/aetherscan.exe `
  --images D:/ScanVideo/ori_img/images `
  --reference artifacts/sfm_acceptance_20260906/reference_ori_original_text `
  --output-dir artifacts/sfm_acceptance_20260906 --name ori_rerun `
  --timeout-seconds 600
ctest --test-dir build -C Release --output-on-failure -R 'sfm.(submap_recovery|mapping|hierarchical)'
python -m unittest discover -s experiments -p test_sfm_acceptance.py
python -m unittest discover -s experiments -p test_sfm_graph_audit.py
```

C++ 三项测试通过，包含新增的“高连接支路仍触发补配”和“第二条连接闭合桥”回归；Python 验收 5/5、图审计 3/3 通过。现阶段时间记录不构成整体提速认证，不能把单次 33.30/31.20 当成确定的加速比。本轮未执行 MVS 或 splat，下游表面和渲染质量尚未验证。
