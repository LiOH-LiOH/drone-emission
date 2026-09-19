# 固定色表与两种渐变：第一版试色

此页为历史记录，已被深海蓝—陶土暖色方案替代。最新要求见
[图片配色标准](figure_color_standard.md)。新试色为
`palette-harmony_20260911_02`，全图试用为 `harmony-gef_20260911`；
色盲适配和发散两端等明度不再是当前约束。

状态：用户已确认固定色表和两种渐变；已用于 `two-scales_20260911` 全图 review，尚未正式采用。
本轮用户已确认的设计约束优先于上一轮按人口／GEF／减排分别建色阶的方案。

## 已确认的约束

- 全局仅保留一个顺序 colormap 和一个发散 colormap，其他单色来自固定色表。
- 基础色采用 Paul Tol Bright 的六个彩色和灰色，补充炭黑、白。
- 渐变不直接使用灰、黑、白作控制色；允许以固定彩色为依据生成受控的
  明暗变体，并将所有控制色登记在同一处。
- 顺序色阶仅做小范围色相变化，靠单调明度提供足够的上下限区分度。
- 发散色阶零点允许带轻微色相的低饱和度颜色，避免纯灰、纯白和近白底。
- 试色板先于全部论文图重绘；其散点是固定随机种子的设计样例，不是科研数据。

## 这版候选

配置：`src/drone_emission/tools/palette_proof.json`。
生成器：`src/drone_emission/tools/palette_proof.py`。
样张：`figures/review/palette-proof_20260911/palette_proof.png`，同目录有 PDF。

| 色阶 | 控制色 | 来源 |
| --- | --- | --- |
| 顺序，低 → 高 | `#69BDDA` → `#1B5080` | cyan → blue 的受控变体 |
| 发散，负 → 零 → 正 | `#215A8E` → `#BEBAA7` → `#88343F` | blue → yellow 低饱和度变体 → coral 深色变体 |

使用 CAM02-UCS 中的线性路径生成 257 个颜色，包含精确的零点。
控制点明度／色度均由配置指定；若路径超出 sRGB 色域，则整体压缩色度，
不直接裁剪原始渐变。这版两个色阶的压缩系数均为 1。
十六进制仅用于显示，完整精度 LUT 在 `ramps.npz` 中。
`palette_registry.json` 保存基色、控制色、来源、色空间坐标和七档色块。

## 检查结果和边界

- 正常视觉顺序色阶的 CAM02-UCS J' 为 74 → 34，明度差 40。
- 正常视觉发散色阶为 38 → 78 → 38，两端明度相同。
- Protanopia、Deuteranopia、Tritanopia 三种 severity=100 模拟下，
  顺序及发散的每半边均保持明度单调。
- 色觉模拟会改变明度对称性，不能据此声称对每种色觉完全等效。
- 采用 Machado 等人的模型，通过 colorspacious 计算。模拟显示时的
  超色域值会裁剪；各模式裁剪比例在 `audit.json` 中记录，结果为近似预览。
- 白底透明度 0.8 的最低对比值约为顺序 1.81、发散 1.68。
  这些仅作诊断，没有将它们包装成通用的散点可见性通过标准。
- 单调性和色差诊断不等于最终视觉验收；小点、重叠和实际印刷仍须检查。

## 重现

在沙盒外运行。当前临时依赖路径内仅安装 colorspacious，未修改项目依赖：

```bash
PYTHONPATH=src:/private/tmp/drone-palette-proof-deps .venv/bin/python \
  -m drone_emission.tools.palette_proof \
  --output-dir figures/review/palette-proof_YYYYMMDD
```

输出目录必须是新目录，以免覆盖之前的试色记录。

设计依据：[Paul Tol](https://sronpersonalpages.nl/~pault/data/colourschemes.pdf)、
[Matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html)、
[Seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html)、
[colorspacious 模拟说明](https://colorspacious.readthedocs.io/en/latest/tutorial.html)。
