# 全景训练复查（2026-09-17）：mask 模式、边界带与等距柱状几何

第 2 轮把全景静态区从 22.59 提到 27.02 dB，但"还是有问题"的感觉是对的。这一轮把问题
拆开定位：先看错误到底集中在哪里，再逐个排除 mask 与投影的嫌疑。

## 结论（先看这个）

1. **默认 mask 模式（`--splat-alpha-mode transparent`，CLI 默认）在这类全景上明显更差，
   而且目视很糟。** 同一命令只差模式：静态区 `25.68 / 25.84 dB`（两次对照）对
   `masked + leak=0` 的 `27.02 dB`；渲染里出现成片的**条纹拖影**（见下），
   因为背景 BCE 把静态背景往"透明"推，优化器只能用细长高斯去补。
2. **修复：`--splat-alpha-leak-weight` 现在同时作用于 transparent 模式**（只缩放 BCE 的
   背景半边，前景半边不变；权重 1 与 pygsplat 逐位一致）。默认行为不变，但
   `--splat-alpha-leak-weight 0` 在两种模式下都能得到干净的全景：默认模式加这一个开关后
   `20.275 / 26.997 dB / 0.8602`，与 `masked + leak=0` 的 `20.285 / 27.022 / 0.8604`
   在噪声内相同（见下表），渲染里的条纹拖影消失。
3. **剩下的误差集中在主体轮廓外侧的一条带**（±32 px：`22.8 dB`，全图静态 `27.0 dB`）。
   这不是 mask 覆盖不准：把 mask 腐蚀 16/48/96 px 后，静态区与这条带**单调变差**
   （`27.02 → 26.91 → 26.62 → 26.25`），说明那条带必须继续被监督，误差来自"背景在本视角
   被人挡住、只能靠其他视角补"的固有难度。
4. **接缝与两极没有问题**：seam（左右各 1% 列）`26.5–26.7 dB`、pole（上下各 6% 行）
   `27.6–28.3 dB`，都高于或接近全局静态 `27.0 dB`。等距柱状投影/光栅化本身是健康的。
5. **修了两处等距柱状几何口径**（深度=沿射线距离、采样射线用球面方向），见第 4 节。
   诚实地说：**它们不改变当前全景训练的数字**——原生相机下深度/多视角损失被显式关闭
   （且要 `--mesh` 才开），但全景的深度图/网格/采样路径此前是错的。
6. 旧二进制 + `masked + leak=0` 复现第 2 轮（`26.57 / 0.860` 对 `26.56 / 0.860`），
   说明这两轮的对比是可信的。

## 误差到底在哪里

离线诊断脚本 `artifacts/native_camera_20260916_round3/analyze_mask_bands.py` 把每个留出
视角的误差拆成四块（全部用**数据集原本的 mask** 评估，所以不同 mask 变体之间可比）：

| 全景 47 留出视角 | 全图 | 静态区 | 轮廓带(±32px) | 接缝列 | 两极带 |
|---|---:|---:|---:|---:|---:|
| 第 2 轮 masked + leak=0 | 20.285 | **27.022** | **22.812** | 26.697 | 28.340 |
| 默认 transparent（对照 A） | 20.043 | 25.835 | 21.653 | 26.102 | 27.621 |
| 默认 transparent（对照 B） | 19.960 | 25.680 | 21.694 | 25.642 | 27.385 |
| 默认 transparent（对照 C，新二进制） | 20.024 | 25.778 | 21.630 | 26.021 | 27.765 |
| masked + leak=0 + 腐蚀 16 px | 20.267 | 26.911 | 22.346 | 26.710 | 28.063 |
| masked + leak=0 + 腐蚀 48 px | 20.220 | 26.618 | 21.447 | 26.503 | 27.691 |
| masked + leak=0 + 腐蚀 96 px | 20.120 | 26.248 | 20.871 | 26.663 | 27.563 |
| **transparent + leak=0（默认模式 + 新开关）** | 20.275 | **26.997** | **22.826** | 26.783 | 28.411 |
| masked + leak=0，cap 2M | 20.306 | 26.971 | 22.798 | 26.497 | 28.206 |
| masked + leak=0，30k 步 | 20.324 | **27.084** | **22.880** | 26.693 | 27.922 |
| 参考：no-mask（人体进模型） | 21.668 | 25.912 | 20.791 | 27.013 | 27.951 |

三个直接结论：

- 轮廓带是唯一的系统性弱区（比全局静态低 4–6 dB），接缝/两极反而更好。
- 逐视角看，最差的是 `02534 / 02052 / 02892 / 03132` 这些**主体占画面很大**的视角
  （mask 黑色占比 20% 以上）：人物挡住的背景在那几个视角里几乎全丢。
- 腐蚀 mask 让轮廓带更差——那条带里的像素被直接监督时，模型反而更准。
- **容量和训练长度都不是瓶颈**：cap 从 1M 提到 2M 是 `26.97`（噪声内不变），步数从 15k
  提到 30k 是 `27.08`（+0.06 dB，噪声 0.07 dB 量级）。
- 旧二进制 + `masked + leak=0` 复现第 2 轮：`26.57 / 0.860` 对 `26.56 / 0.860`
  （该 run 目录名里的 "mv_default" 有误导：多视角权重在没有 `--mesh` 时会被 CLI 置 0，
  配置行里就是 `multi_view_geo_weight=0`，所以它实际是旧二进制的 masked+leak0 复跑）。
- 默认模式的三次对照跨批次一致（`25.68 / 25.84 / 25.78`），所以跨批次比较没有被其它
  改动污染；噪声约 0.1 dB。

### 目视：默认模式 vs masked + leak=0

视角 192（`02892.jpg`，最差视角之一）：

- 默认 transparent：整幅出现放射状条纹拖影（白板、窗户、天花板都被拉成细长高斯）。
- `masked + leak=0`：只有画面顶端中间一条褐色拖影残留——那是**持镜人的手臂从镜头正上方
  扫过**，而数据集 mask 的顶部中间是白的（没遮住）；腐蚀 48 px 也去不掉它，因为那片白色
  离最近的黑色区域有几百像素。这一块只能靠更好的 mask（把手臂也标出来）解决。

## mask 模式与泄漏权重

`--splat-alpha-leak-weight` 现在在两种模式下都生效：

| 模式 | alpha 目标 | 权重含义 |
|---|---|---|
| `transparent` | `BCE(alpha, mask)` 全图 | 前景半边 `mask=1` 保持全权重；背景半边 `mask=0` 乘权重 |
| `masked` | 仅前景 RGB + 背景泄漏惩罚 | 泄漏惩罚乘权重 |

`1` 与 pygsplat 逐位一致（`test_mask_loss_modes` 覆盖 `0 / 0.5 / 1` 的梯度与损失值），
`0` 表示"只要求主体不透明、不要求被遮挡的静态背景变空"。

另外训练日志在"全景 + mask + 权重>0"时会提示一句：
`panorama masks keep the background alpha target at weight …; use
--splat-alpha-leak-weight 0 when a moving subject occludes static geometry`，
免得默认参数一路跑出条纹拖影还不知道开关在哪。

## 等距柱状几何：深度与采样口径

1. **`pixel_unit_ray` 对全景返回的是透视射线。** 等距柱状内参（`fx = fy = width/2π`、
   `cx = width/2`）让像素偏移就是方位角/仰角（弧度），射线应当是球面方向
   `(cos el·sin az, sin el, cos el·cos az)`；旧代码返回的是 `(x, y, 1)/|·|`。
2. **`pixel_ray_z` 把射线长度换算成"相机 Z"，这对全景没有意义**：半个全景的 `Z` 是负的。
   全景的深度就取沿射线距离（系数 1）。
3. 点采样反向（`sample_depth_backward`）对全景走的是透视分支，与本轮的球面射线不一致；
   现在和 fisheye 共用"射线 + 投影 Jacobian"的形式，等距柱状用自己的极坐标 Jacobian。

影响范围：`rendered.median_depth`、网格提取（`mesh.cpp`）、`sample_depth` 采样路径。
目前**不影响训练数字**，因为原生化相机下 `use_multi_view` 与深度/法线损失被显式关闭
（`--mesh` 才打开多视角损失）。新增 `test_equirect_depth_convention` 把"深度=沿射线距离"
钉住：+Z、−Z、斜上方三个方向的单高斯都必须报出真实距离（修复前 −Z 方向会给出 0.3×）。

## 仍未解决

1. **主体轮廓带的补全**仍是最大误差源。腐蚀 mask 证明"多给监督"更有效，那么可试的方向是
   按视角覆盖度给损失加权（只在被别的静态视角看过的像素上监督），以及更强的多视角先验——
   但原生相机目前拿不到 MVS 深度/法线，这条链路要重新设计。
2. **持镜人手臂这类 mask 漏标**只能靠更好的 mask；管线侧目前只能靠
   `--splat-alpha-leak-weight 0` 减少它造成的间接伤害。
3. 本轮没有验证：更高分辨率下的 mask 模式对照（第 2 轮只有 1920 的原尺寸对照）、
   以及更细粒度的 mask（把持镜人手臂、影子也标出来）能拿回多少轮廓带误差。

## 推荐命令

```powershell
# 全景实拍（有持镜人）：默认 mask 模式 + 关掉背景 alpha 目标
.\build\photara\Release\photara.exe `
  --images D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset/images `
  --splat-dataset D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset/sparse/0 `
  --output artifacts/native_camera_20260916_round3/pano_transparent_leak0/model.ply `
  --splat-strategy adc_igs --splat-iterations 15000 --splat-densification-cap 1000000 `
  --splat-max-resolution 1920 --splat-progressive-resolution=false `
  --splat-ppisp=false --splat-bilateral-grid=false --splat-use-mask=true `
  --splat-alpha-leak-weight 0 `
  --splat-undistort=false --splat-depth-normal-weight 0 `
  --splat-mv-geo-weight 0 --splat-mv-ncc-weight 0 `
  --splat-eval-split-every 8 --splat-prefetch-views=0 `
  --splat-cache-auto=false --splat-device-cache-mb=0
```

实验目录 `artifacts/native_camera_20260916_round3/`：每个 run 有 `command.json`、
`train.log`、`image_diagnostics.json`、`mask_bands.json`；
脚本 `make_mask_variants.py`（生成腐蚀 mask）、`analyze_mask_bands.py`（误差分带）、
`estimate_shift.py`（亚像素偏移检查，12 个视角全为 0，没有系统性半像素错位）。
