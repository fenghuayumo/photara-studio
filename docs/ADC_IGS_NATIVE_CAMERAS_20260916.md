# ADC-IGS 鱼眼及全景实测（2026-09-16）

## 结论（先看这个）

原生相机路径**能跑通**：COLMAP 模型 5 / 17 都按源投影训练，没有走去畸变针孔，15,000 步无 CUDA 报错。

CLI **已经有 mask 参数**，默认就是开的：

- `--splat-use-mask BOOL`（默认 `true`）
- `--splat-alpha-mode masked|transparent`（默认 `transparent`）
- `--masks DIR`（默认找 `images/` 同级 `masks/`）

第一组全景实验脚本写了 `--splat-use-mask=false`，等于把默认关掉了。已按同一协议补跑
`--splat-use-mask=true --splat-alpha-mode masked`。

质量上**还不能当成已经优化好**：

- 鱼眼室内中景可用，近相机物体和后院室外明显差。
- 全景不开 mask：办公室结构能成型，但人体/接缝鬼影进模型。
- 全景开 mask：人体不再被拟合，但静态区 PNG 指标下降，人物位置留下半透明气泡/雾。
- 全景位置梯度有限差分回归未通过，不能宣称 equirect 反传已经对齐。

训练器在 `use_mask=true` 时只对 mask 前景报 PSNR，**不能和全图 PSNR 混比**。
下面公平对照用同一套静态 mask，在两组 PNG 上离线计算。

## 范围与数据

沿用已提交的 IGS 初始 opacity（默认 0.1）。不启用 PPISP / 双边网格，不重估位姿，不开深度/多视角损失，不开渐进分辨率，GPU 图像缓存和预取关闭。

实验目录：`artifacts/native_camera_20260916/`。二进制 SHA256
`95278349ba3124596f6802e636cccbe808a46d8cfd9ef7d28de720c82f168738`。

| 数据 | 输入相机 | 注册视角 | 训练 / 留出 | 源分辨率 | 训练分辨率 |
|---|---|---:|---:|---|---|
| `D:/ScanVideo/alameda/images_2_dataset` | OPENCV_FISHEYE，COLMAP ID 5 | 1741 | 1523 / 218 | 1752×1168 | 1752×1168 |
| `D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset` | EQUIRECTANGULAR，COLMAP ID 17 | 375 | 328 / 47 | 7680×3840 | 1920×960 |

alameda 的 `images` 有 1742 张，sparse 注册 1741 张，只训练注册视角。
鱼眼内参 `fx/fy ≈ 604.1/604.6`，`cx/cy ≈ 879.4/581.5`，
`k1..k4 ≈ 0.0354, 0.0111, -0.0031, 0.0010`。
全景原图 8K，`--splat-max-resolution 1920` 是 4× 缩小，**不能据此推断 8K 细节**。
全景 `masks/` 有 375 张人物 mask（白=静态，黑=人体）。

所有运行 15,000 步，100 万高斯上限，每 8 张固定留出一张。

## 投影与回归

- 鱼眼：`equirectangular=0 fisheye=1741 ... no undistortion`
- 全景：`equirectangular=375 fisheye=0 ... no undistortion`

`photara_splat_test --fisheye-only` 在
`artifacts/native_camera_20260916/projection_tests.log` 以

`Panorama mean gradient differs from finite differences`

失败。测试对 `(0.2, 0.1, 2)` 和过接缝后向点 `(0.02, 0.1, -2)` 比较 `dL/dmean`
与 `ε=1e-3` 中心差分。训练仍收敛，但不能把这次 run 当成 backward 已验证。

## 总表

| | 鱼眼 | 全景无 mask | 全景 masked |
|---|---:|---:|---:|
| 目录 | `fisheye/` | `panorama/` | `panorama_masked/` |
| `--splat-use-mask` | false | false | **true** |
| `--splat-alpha-mode` | — | transparent（未用 mask） | **masked** |
| 初始稀疏点 | 1,072,214 | 174,370 | 174,370 |
| 最终高斯 | 1,000,000 | 1,000,000 | 1,000,000 |
| 训练墙钟 | 659 s | 205 s | 189 s |
| 训练器 15k 留出 PSNR / SSIM | 21.132 / 0.708 | 20.934 / 0.791 | 22.347 / 0.776 |
| 像素集合 | 全图 | 全图 | **仅静态区** |
| PNG 全图 PSNR | — | 21.088 | 19.191 |
| PNG 静态区 PSNR | — | **24.979** | 22.585 |
| PNG 静态球面 PSNR | — | **24.289** | 21.679 |

PNG 诊断是 8-bit + Pillow 双线性缩放，不是训练器浮点指标；两组用同一张静态 mask，可以互比。
训练器 22.35 dB **不是**比 20.93 dB 提升了 1.4 dB。

masked 训练器曲线：1k 19.37 → 5k 19.12 → 10k 20.33 → 15k 22.35。
无 mask：1k 19.51 → 5k 20.47 → 10k 20.82 → 15k 20.93（平台）。

## 鱼眼：室内中景可以，近景和室外不行

稀疏点 1.07M 超过 cap。第 200 步剪掉 72,214，之后几乎没有 IGS 增长，只做替换。

| 视角 | 文件 | PSNR | 内容 |
|---:|---|---:|---|
| 56 | `DSC06492.JPG` | 11.42 | 门口近距离婴儿车 |
| 1320 | `DSC07764.JPG` | 13.00 | 后院天空、白桦细枝 |
| 584 | `DSC07024.JPG` | 13.55 | 角落书架近拍 |
| 168 | `DSC06604.JPG` | 26.44 | 客厅中景 |

客厅桶形窗和家具说明原生鱼眼前向是对的。差视角按区域成簇（后院 `DSC076xx–077xx`、门口推车），不是随机坏投影。

## 全景无 mask（`panorama/`）

`mask_dir` 已自动找到，但脚本写了 `--splat-use-mask=false`。人体被拟合进高斯。

| 视角 | 文件 | 训练器 PSNR | PNG 静态 |
|---:|---|---:|---:|
| 208 | `03132.jpg` | 17.16 | 20.73 |
| 8 | `00133.jpg` | 18.61 | 22.91 |
| 0 | `00014.jpg` | — | 25.43 |
| 280 | `04212.jpg` | 25.88 | 27.67 |

`00014.jpg` 左右边缘是方位缝上的同一个人。不开 mask 时重建成肉粉色团。
办公室桌子、灯、绿植仍在，等距柱状前向和经度环绕是通的。

## 全景开 mask（`panorama_masked/`）

命令与无 mask 组相同，只改：

```
--splat-use-mask=true --splat-alpha-mode masked
```

日志：`use_mask=1 alpha_mode=masked`，第 1 步 `rgb=0.317 alpha=0.066`（无 mask 组 alpha=0）。
`--masks` 未显式传，默认 sibling `masks/` 生效。

`masked` 模式：只在静态像素上算 RGB，并对人体区域做 alpha leakage 惩罚，避免把人学成实心几何。
这是这套人物 mask 该用的模式；`transparent` 会把人体轮廓逼成透明 alpha，更不适合挡在墙前面的持镜人。

目视：

- 接缝两侧的人脸/身体不再被拟合（这是对的）。
- view 0 / 8 左右墙和窗户上出现大块半透明气泡；view 208 走廊玻璃处有圆形雾团，就是原本人体所在。
- 静态办公室比无 mask 组更糊、更灰。PNG 静态 PSNR 每个抽查视角都更差，例如 view 280：27.67 → 25.81，view 0：25.43 → 23.64。

原因（实现层面，本轮不改代码）：360 里人体挡住的是后面那面墙。mask 掉这些像素后，这些射线在该视角没有 RGB 监督；`masked` 还在同一位置压 alpha。其他视角看见的不透明墙，和这里的“变透明”互相打架，容易长出大气泡高斯。这不是“没开 mask”的问题，是**开了 mask 之后 leakage 和遮挡补全还没做好**。

## 和“优化是否 OK”直接相关的判断

1. **加载和训练协议 OK。** 原生鱼眼/全景都在源投影上训练。
2. **CLI mask 链路 OK。** 默认 `--splat-use-mask true`，`--masks` 可省略；本组成对验证了 false / true。
3. **鱼眼前向 OK。** 中景室内构图正确。
4. **全景前向基本 OK。** 2:1 图、灯带、方位缝正确。
5. **全景反传未验证。** 有限差分失败。
6. **全景必须开 mask**，否则人体进模型。开了之后静态区目前变差，不能把训练器 22.35 dB 当成改进。
7. **IGS 在鱼眼上几乎没增长**（初始化已超 cap）。

## 后续改进（按优先级，本轮不改代码）

1. **修全景 `dL/dmean` 有限差分。**
2. **mask 后不要把遮挡当透明背景。** 人体是动态遮挡，不是天空/绿幕。值得试：只屏蔽 RGB、减弱或关掉背景 alpha leakage；或只在该像素从未被任何静态视角看到时才罚透明。
3. **比较指标必须用同一静态 mask。** 训练器 `foreground_psnr` 目前和 `psnr` 打成同一个数，且随 `use_mask` 换像素集合。
4. **鱼眼初始化不要超过 cap。**
5. 鱼眼室内/后院分开报。
6. 全景若要看上限，至少再跑长边 3840；1920 只验证结构。
7. 颜色校正本轮关闭，不要和投影/mask 问题混谈。

## 复现

```powershell
# 鱼眼（无人物 mask）
.\build\photara\Release\photara.exe `
  --images D:/ScanVideo/alameda/images_2_dataset/images `
  --splat-dataset D:/ScanVideo/alameda/images_2_dataset/sparse/0 `
  --output artifacts/native_camera_20260916/fisheye/model.ply `
  --splat-strategy adc_igs --splat-iterations 15000 --splat-densification-cap 1000000 `
  --splat-max-resolution 1920 --splat-progressive-resolution=false `
  --splat-ppisp=false --splat-bilateral-grid=false --splat-use-mask=false `
  --splat-undistort=false --splat-depth-normal-weight 0 `
  --splat-mv-geo-weight 0 --splat-mv-ncc-weight 0 `
  --splat-eval-split-every 8 --splat-prefetch-views=0 `
  --splat-cache-auto=false --splat-device-cache-mb=0

# 全景（必须开 mask；masks/ 与 images/ 同级，可省略 --masks）
.\build\photara\Release\photara.exe `
  --images D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset/images `
  --splat-dataset D:/BaiduNetdiskDownload/VID_20260911_145523_00_007_dataset/sparse/0 `
  --output artifacts/native_camera_20260916/panorama_masked/model.ply `
  --splat-strategy adc_igs --splat-iterations 15000 --splat-densification-cap 1000000 `
  --splat-max-resolution 1920 --splat-progressive-resolution=false `
  --splat-ppisp=false --splat-bilateral-grid=false `
  --splat-use-mask=true --splat-alpha-mode masked `
  --splat-undistort=false --splat-depth-normal-weight 0 `
  --splat-mv-geo-weight 0 --splat-mv-ncc-weight 0 `
  --splat-eval-split-every 8 --splat-prefetch-views=0 `
  --splat-cache-auto=false --splat-device-cache-mb=0
```

完整 argv 见各 run 的 `command.json`。
`panorama_mask/` 是一次被 120s 超时杀掉的半成品（停在 14k），不要当结果用。

## 后续轮次

上面的 1（全景反传有限差分）和 2（mask 后遮挡处理）已在第二轮完成：
`docs/ADC_IGS_NATIVE_CAMERAS_R2_20260916.md`。要点：反向 Jacobian 存成了转置、位置二阶导
用了透视公式，修好后全景静态区 PSNR `22.59 → 24.14 dB`；mask 的 alpha 泄漏惩罚就是气泡
的来源，新增 `--splat-alpha-leak-weight`（全景实拍用 `0`）后到 `27.02 dB`。
