# 可微分 Live2D 光栅化与合成（A3）

写于 2026-11-06。目标：用 nvdiffrast + PyTorch 实现与 runtime（`packages/runtime`，Cubism Framework R5 WebGL 渲染器，
**MSAA 0**）逐像素等价、且对顶点 / 不透明度 / 颜色 / 变换参数可求导的图层光栅化与合成，给 GPU 拟合用。

## 文件

| 文件 | 内容 |
|---|---|
| `l2d/raster.py` | 图层光栅化：贴图（预乘 + GL mipmap + 自写三线性采样）、clipping mask、三种 drawable 混合、乘算色/屏幕色、剔除、变换与相机辅助函数 |
| `l2d/composite.py` | 合成器：blur / motionBlur / resample / color 滤镜、圆角矩形羽化 clip、opacity、normal/add/multiply/screen、多图层 |
| `l2d/dumpio.py` | A3 dump 的读取（只依赖 numpy / PIL） |
| `scripts/a3_dump.ts` | 在 harness 页面里用 runtime 渲染并 dump drawable 数据、参数值、参考 PNG（可选图层缓冲与 mask 缓冲） |
| `scripts/a3_variants.py` | 派生场景：覆盖原片没有用到的 flipX / 负 scaleX / 相机 / add / multiply / screen / motionBlur / bicubic |
| `scripts/a3_compare.py` | 对拍：重渲 dump 的帧，按 metric.py 口径算 SSIM 与像素误差，分段定位（光栅化 / 滤镜） |
| `scripts/a3_gradperf.py` | 梯度检查（自动求导 vs 中心差分）与性能、显存测量 |
| `tests/test_a3_raster.py` | 单元测试 + 对拍回归（`python -I -m unittest discover -s tools/gpu/tests -p "test_a3_*.py"`） |

## 用法

```bash
# 0) 环境（WSL）：无头 Chrome 走 d3d12，MSAA 0
export CHARAFORGE_GPU=egl CHARAFORGE_MSAA=0 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
# 1) dump（--layers 另存图层缓冲与 Framework 的 mask 缓冲，用于分段定位）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a3_dump.ts \
  --scene scenes/deepseek-chan/scene.json --frames 2370,2400 --tag main --layers
# 2) 对拍
tools/.venv/bin/python -I tools/gpu/scripts/a3_compare.py --dump .data/gpu/frames/main --out .data/gpu/a3_compare.json
```

```python
from l2d.dumpio import Dump
from l2d.raster import load_rasterizer, LayerInput, layer_matrix, camera_matrix, affine_mul
from l2d import composite as C

d = Dump(".data/gpu/frames/main")
R = load_rasterizer(d, "dsc")                     # 贴图约 0.85 GB 显存
fr = d.frame(2400)
L = fr["layers"][0]
buf = R.render(LayerInput.from_dump([L]), 1280, 720)        # 图层缓冲 [1, 720, 1280, 4]，预乘
img = C.composite([(buf, L["draw"])], 720, 1280, None, B=1)  # 合成缓冲
```

接 A1（`l2d.deform`）的输出：`LayerInput.from_arrays(positions, opacities, multiply, screen, render_orders, visible, to_screen)`，
口径相同（顶点为画布像素，y 向下）。`to_screen` 用 `affine_mul(camera_matrix(...), layer_matrix(...))` 由图层 / 相机参数算出即可对它们求导；
世界空间到屏幕空间的约定见 `packages/scene/src/scene.ts` 第 2 条（space = "screen" 的图层不乘相机）。

`render()` 的选项：

| 参数 | 缺省 | 含义 |
|---|---|---|
| `aa` | True | 求导时附加边缘梯度直通项（见下）；不影响前向值；无梯度时自动关闭 |
| `quantize` | True | 每次混合后夹到 [0,1] 并 8 位取整（直通估计），与 GL 逐 drawable 写 RGBA8 一致；拟合时可设 False |
| `mask_grad` | False | 梯度是否穿过 mask 缓冲的生成传到 mask drawable 顶点（mask 缓冲是 2048² 级别，显存大）；False 时被裁切 drawable 仍经采样位置得到梯度 |
| `aa_layers` | None | 只给最前面 K 个剥离层加边缘梯度项，省显存（720p 建议 8） |

`composite` 的滤镜 / 输出函数都接受 `quantize`（缺省 True，每个 pass 写 RGBA8），标量参数可传张量求导。

## dump 格式（供 A1 / A5 使用）

目录 `.data/gpu/frames/<tag>/`（`a3_dump.ts` 写，`l2d/dumpio.py` 读）：

| 文件 | 内容 |
|---|---|
| `manifest.json` | `format`、`scene`、`gpu`（WebGL renderer 字符串）、`msaa`、`composition {width,height,background}`、`fps`、`frames`、`models {模型ID: 文件前缀}` |
| `model_<ID>.json/.bin` | 静态数据，见下 |
| `f<帧号6位>.json/.bin` | 每帧数据，见下 |
| `f<帧号>_ref.png` | runtime 合成结果，**非预乘** RGBA，与 `pnpm render` 输出逐字节相同的编码 |
| `f<帧号>_ref_premul.png` | 同一结果的**预乘** RGBA 原始字节（PNG 只作无损容器） |
| `f<帧号>_<图层>_model.png` / `_src.png` | （`--layers`）图层缓冲：Cubism 绘制结果 / 滤镜之后，预乘原始字节 |
| `f<帧号>_<图层>_mask<i>.png` | （`--layers`）Framework 的 clipping mask 缓冲 i（2048²，RGBA 原始字节，PNG 第 0 行 = GL 第 0 行 = mask 空间 y = −1） |
| `scan.json` | （`--scan`）只做规划不渲染：每帧每个图层的 visible / toScreen / draw 参数，用来挑帧 |

`.json` 的 `arrays` 字段：`{名字: {dtype: f4|i4|u1|u2, shape, offset, nbytes}}`，`.bin` 为小端数组首尾相接（8 字节对齐）。

**静态数据 `model_<ID>`**（与帧无关，来自 Core 公开 API）：
- JSON：`modelUrl`、`textures`（相对 model3.json 的贴图路径）、`canvas {width,height,originX,originY,pixelsPerUnit}`、
  `drawableCount`、`drawableIds`、`parameterIds/Min/Max/Default`、`partIds`、`partParents`、`offscreenCount`、`blendModeEnabled`、
  `usesCulling`、`maskBufferSize`（runtime 的 2048）、`maskRenderTextureCount`、`highPrecisionMask`、
  `clipContexts [{masks, clipped}]`（Framework 的 clipping context 列表，按建立顺序）。
- 数组：`vertexOffsets [n+1]`、`indexOffsets [n+1]`、`maskOffsets [n+1]`（CSR 偏移）；`uvs [V,2]`（Cubism UV，v 向上）；
  `indices [Σ索引]`（u2，每个 drawable 局部编号）；`masks`（i4）；`textureIndices [n]`；`constantFlags [n]`
  （bit0 Additive、bit1 Multiplicative、bit2 双面、bit3 反转遮罩）；`blendModes [n]`（Cubism 5.3 字段，本模型未启用）；`parentPartIndices [n]`。

**每帧 `f<帧号>`**：JSON `layers[]`，按场景图层顺序（只含该帧存在的图层），每项：
- `id`、`model`、`g`（frameHold 映射后的求值帧）、`visible`（runtime 是否画它；false 时数组是陈旧的，只保证 `params` 为该帧求值结果）、
  `toScreen [a,b,c,d,e,f]`（画布像素 → 合成像素：x' = a·x + c·y + e，y' = b·x + d·y + f，已含相机）、`space`、
  `draw {filters, clip, opacity, blendMode}`（runtime 的 ResolvedFilter / LayerDrawParams，已在 g 上采样）、`antialias`、`modelOpacity`、
  `clipContexts[]`（Framework 本帧的 `isUsing/channel/bufferIndex/layoutBounds/allClippedDrawRect/matrixForMask/matrixForDraw`，列主序 4×4，对拍用）；
- 数组（键 `<图层序号>/<名字>`，`dumpio` 已拆好）：
  - `positions [V,2]`：顶点，**画布像素**（px = originX + x·PPU，py = originY − y·PPU），按 drawable 顺序拼接（偏移见 `vertexOffsets`）；
  - `opacities [n]`、`multiplyColors [n,4]`、`screenColors [n,4]`（Core 的 drawable 值，rgba）；
  - `renderOrders [n]`（drawable i 在绘制序列中的位置）、`drawOrders [n]`、`dynamicFlags [n]`（bit0 可见、bit5 顶点已更新，绘制时刻的值）；
  - `params [P]`：该帧全部模型参数值（Core 输入，按 `parameterIds` 顺序；物理、motion、表情、参数轨都已应用）；
  - `partOpacities [nParts]`。

参考 PNG 与 `pnpm render` 走同一条路径（harness 的 `renderFrame` 顺序播放、同一个 `encodePng`）。

## 与 Framework / runtime 对齐的细节

**绘制**（`CubismRenderer_WebGL.drawModel`，非 5.3 混合模式路径；本模型 `blendModeEnabled = false`、无 offscreen）：
- 按 render order 依次画 `IsVisible` 的 drawable；不透明度 0 的贡献恰为 0，也跳过（不影响结果）。
- 片元：`tex = texture(预乘贴图, (u, 1−v))`；`rgb *= multiply`；`rgb = rgb + screen·a − rgb·screen`；`color = tex·opacity`。
- 混合：Normal (ONE, ONE_MINUS_SRC_ALPHA)；Additive (ONE, ONE | ZERO, ONE)；Multiplicative (DST_COLOR, ONE_MINUS_SRC_ALPHA | ZERO, ONE)。
  每次混合写 RGBA8：**结果先夹到 [0,1]**（一个加算 drawable 下面压着乘算 drawable 时这一点决定对错，见对拍记录），再 8 位取整。
- 贴图：`createImageBitmap(premultiplyAlpha)` 的 8 位预乘 round(c·a/255)；`generateMipmap` 逐级 2×2 平均取整；
  `LINEAR_MIPMAP_LINEAR` + `CLAMP_TO_EDGE`，λ = log2(max(|∂(s,t)·size/∂X|, |∂(s,t)·size/∂Y|))，λ ≤ 0 走 MAG LINEAR。
  贴图尺寸不一（4096×8192、8192²、2048² …），所以没有用 nvdiffrast 的 `texture`（只收 float32，整套约 3 GB），
  而是 uint8 mip 链 + 自写三线性采样（0.85 GB）。
- 背面剔除：Framework 以模型单位（y 向上）中的逆时针为正面；runtime 在翻转（det < 0）时交换 frontFace，所以正面判定与是否翻转无关。
  本模型 715 个 drawable 全是双面，剔除路径只做了实现、没有真实数据验证。

**clipping mask**（`CubismClippingManager` / `setupClippingContext`，非高精度）：
- 上下文按 drawable 顺序建立（`findSameClip` 语义），布局 `setupLayoutBounds`：本模型 49 个上下文、2 张 2048² 缓冲，
  每张 RGBA 4 个通道，R 通道 7 个、其余 6 个，3×3 分格；与 dump 的 Framework 值逐项一致（测试断言）。
- 包围盒：所有被裁切 drawable（**不论可见与否**）的全部顶点；最大值初值是 `Number.MIN_VALUE`（Framework 的怪癖：全为负时上界取 0）；
  外扩 5%；`matrixForMask`、`matrixForDraw` 与 Framework 的值相对误差 < 1e-6。
- mask 缓冲：清成 1，每个 mask drawable（只画「顶点已更新」的，**与可见性、不透明度、乘算色无关**）
  以 (ZERO, ONE_MINUS_SRC_COLOR) 写 `channelFlag·tex.a`，即该通道值 = Π(1 − tex.a)；片元在布局矩形外（isInside）不写。
  这里在每个上下文的布局矩形内真的按 2048 分辨率光栅化（只画「被裁切片元实际会采样到的区域 ∩ mask drawable 包围盒」的窗口，
  窗口外即清屏值 1）；所有窗口打包进一张画布一次剥离画完。与 Framework 读回的 mask 缓冲比：各上下文平均误差 0.01–0.7/255，
  把 Framework 的缓冲直接代进来渲染，结果与用自己的缓冲完全相同（误差不来自 mask）。
- 采样：`maskVal = 1 − texture(mask, clipPos)[channel]`（LINEAR、无 mipmap），反转遮罩取 `1 − maskVal`，`color *= maskVal`。

**合成**（`compositor.ts` / `shaders.ts`，每个 pass 写 RGBA8）：滤镜按数组顺序；blur 两遍、半径 ceil(3σ)、缓冲外透明、按全部权重归一化；
motionBlur 双线性 + 范围检查；resample 先 k×k 双线性区域平均缩小（round 用 JS 的 floor(x+0.5)）再像素中心对齐放大；
color 在反预乘颜色上做；输出 pass = clip 覆盖率 × opacity，Pixi 预乘混合（normal、add、multiply = [DST_COLOR, 1−αs, ONE, 1−αs]、
screen = [ONE, ONE_MINUS_SRC_COLOR, ONE, 1−αs]）。滤镜单独对拍（输入用 runtime 的图层缓冲）最大误差 ≤ 2/255。

## 可微分的实现

- **可见性**：nvdiffrast `DepthPeeler` 逐层剥离。深度 = render order（后画的更近）；同一 drawable 内的顶点加一个落在本 drawable
  深度槽内的散列偏移，自身重叠（网格折叠）的三角形也会被逐层剥出（顺序与 GL 的索引顺序不一定相同，只影响重叠处的半透明叠加次序）。
  本模型一帧约 24–29 层，720p 大尺寸角色约 480 万个片元。
- **片元着色在压缩后的片元列表上做**（不做整幅图）：重心坐标及其屏幕导数由三角形屏幕坐标自己算，对顶点可导；
  nvdiffrast 只提供「哪个三角形」。求导时每层的着色用 `torch.utils.checkpoint` 包起来，只保存每个片元的混合系数。
- **合成**：三种混合都是每通道仿射 D ← B_k + A_k ⊙ D，从后往前按 GL 顺序做（含每步夹取 / 取整）。
- **边缘梯度**：前向是无抗锯齿的（与 MSAA 0 的 runtime 逐像素一致）。每个剥离层把 (B_k, A_k − 1) 铺成整幅图过 `dr.antialias`，
  取 Δ = aa − aa.detach()（前向为 0），在 AA 改动过的像素上加 T_above ⊙ (ΔB + ΔA ⊙ D_below)：
  T_above（上方各层透射率）与 D_below（下方合成）都已知且不求导，这正是「该层边缘挪动一点」对最终像素的一阶影响。
- **变换**：`layer_matrix`、`camera_matrix`、`affine_mul` 是张量函数，对 x/y/scale/rotation/scaleX/scaleY/相机参数可导。
- mask 矩阵（包围盒）不求导（不光滑，Framework 也是逐帧重算）；mask 缓冲缺省不求导（`mask_grad`）。

## 对拍结果（2026-11-06，RTX 5080 Laptop，ANGLE D3D12，MSAA 0）

`a3_compare.py`：44 帧来自 `scenes/deepseek-chan/scene.json`（挑帧依据：`a3_dump.ts --scan` 的规划扫描），14 帧来自派生场景。
SSIM 口径同 `tools/qa/metric.py`（两边都按 alpha 合成到参考视频同一帧，遮罩 = dilate(α_runtime > 0.1, 半径 3)，RGB 高斯 σ = 1.5；
不扣 shots.json 的排除区，因为这里比的是两个渲染器）。像素误差在预乘 RGBA8 上算。

全部 58 帧 SSIM@360 > 0.99（最小 0.99934，均值 0.99960）；SSIM@720 最小 0.99736、均值 0.99854；
预乘 RGBA8 全画面平均误差 0.131/255（遮罩内 0.257），最大 113。分段：图层光栅化平均误差 0.10/255（最大一帧 0.17），
滤镜单独对拍最大误差 2/255。

| 类别 | 帧 | SSIM@360 最小 | SSIM@360 均值 | SSIM@720 最小 | SSIM@720 均值 | 最大误差 | 平均误差 |
|---|---|---|---|---|---|---|---|
| s24 单实例（含 color + resample 0.6 + blur） | 2370, 2385, 2400, 2415, 2430, 2441 | 0.99951 | 0.99956 | 0.99862 | 0.99870 | 19 | 0.120 |
| blur / 景深（σ > 1.5） | 1520, 1712, 2140, 2306, 2650, 2700, 3490 | 0.99934 | 0.99959 | 0.99736 | 0.99825 | 113 | 0.132 |
| resample | 800, 1000, 1750, 3000, 3120, 3700 | 0.99951 | 0.99960 | 0.99817 | 0.99862 | 24 | 0.125 |
| clip（s15 圆形开窗、画中画圆角矩形、s04/s13/s25/s27/s29） | 300, 520, 600, 640, 1100, 1360, 1380, 1398, 1450, 2500, 2620, 2660 | 0.99935 | 0.99961 | 0.99741 | 0.99841 | 107 | 0.137 |
| 多实例（2～3 个图层） | 760, 900, 1300, 1500, 1880 | 0.99956 | 0.99964 | 0.99842 | 0.99866 | 50 | 0.183 |
| 不透明度渐变 | 1356, 1362, 1730, 1760, 1780 | 0.99955 | 0.99962 | 0.99804 | 0.99852 | 59 | 0.118 |
| scaleX（s15.chibi 非等比） | 1229, 1232, 1236 | 0.99950 | 0.99956 | 0.99860 | 0.99869 | 32 | 0.053 |
| 派生：flipX + 相机（zoom 1.08、旋转 3°） | 2380, 2420 | 0.99955 | 0.99955 | 0.99867 | 0.99869 | 9 | 0.130 |
| 派生：负 scaleX + 非等比 | 3110, 3130 | 0.99947 | 0.99948 | 0.99843 | 0.99843 | 15 | 0.094 |
| 派生：blendMode screen | 800, 900 | 0.99958 | 0.99964 | 0.99866 | 0.99882 | 75 | 0.178 |
| 派生：blendMode add | 1850, 1890 | 0.99960 | 0.99964 | 0.99834 | 0.99860 | 29 | 0.186 |
| 派生：blendMode multiply | 720, 770 | 0.99966 | 0.99974 | 0.99885 | 0.99903 | 45 | 0.132 |
| 派生：motionBlur | 1530, 1560 | 0.99957 | 0.99960 | 0.99790 | 0.99808 | 2 | 0.105 |
| 派生：resample bicubic | 1000, 1040 | 0.99947 | 0.99956 | 0.99868 | 0.99884 | 17 | 0.129 |

（原片没有 flipX，所以翻转、负 scaleX、相机、add/multiply/screen、motionBlur、bicubic 用 `a3_variants.py` 的派生场景覆盖。）

**最差帧 f2650**（SSIM@360 0.99934，s29：近景大幅模糊的 `s29.body` + 远景全身 `s29.full`）：误差集中在 `s29.full` 上孤立的
亮点（全身缩得很小，大量三角形小于一个像素，个别像素中心的三角形归属与 GPU 不同，最大 113），以及 `s29.body` 大半径 blur 上
±1 的取整波纹。f2660（同一镜头，clip 类）同因。

逐帧数字见 `.data/gpu/a3_compare.json`（含每个图层「光栅化」「滤镜」两段各自与 runtime 缓冲的误差）。

**残差的来源**（按影响大小）：
1. 亚像素三角形与边上的像素：像素中心正好落在边附近时，GL（D3D12 硬件）与 nvdiffrast 的填充规则 / 定点精度不同，
   个别像素归属不同的三角形（误差可到几十～一百多，但孤立）。最差帧 f2650（`s29.full`，远景全身、大量亚像素三角形）就是这种情况，
   SSIM@360 仍为 0.99934。
2. 8 位舍入：GL 的混合单元、滤镜 pass 与我们的取整在 0.5 附近可能差 1；blur 大的图层上表现为 ±1 的波纹。
3. 贴图 LOD：硬件 LOD 的定点近似与这里的浮点 log2 在层级边界附近选到不同的插值权重（±1～2）。
4. 自身重叠的三角形叠加次序（见上）。

## 梯度检查与性能

`a3_gradperf.py`（s24 帧 2385，640×360 合成分辨率，只看光栅化：滤镜关掉、quantize=False）：

- **顶点**：目标图 = 24 个随机可见 drawable 整体平移 (4, −4) 画布像素后的渲染，L = mean((I − T)²)。
  逐个 drawable、x/y 两个方向（48 个方向，差分显著的 33 个）比较自动求导方向导数与中心差分：

  | 差分步长 ε（画布像素） | 符号一致 | 相关系数 | AD / FD 中位数 |
  |---|---|---|---|
  | 0.5 | 33/33 | 0.9991 | 1.085 |
  | 1.0 | 33/33 | 0.9989 | 1.089 |
  | 2.0 | 33/33 | 0.9976 | 1.112 |

  朝目标的整体方向导数 −5.2e−4 < 0（损失下降方向正确）。AD 比差分大约 9%：边缘项是一阶线性化，
  差分步长覆盖了离散像素的跳变，量级吻合。
- **图层平移 x**（经 `to_screen` 求导）：AD −3.37e−4，差分 −2.86e−4（ε 0.5 px）/ −2.49e−4（ε 1 px）。
- **drawable 不透明度**（8 个）：符号 100%、相关 0.9998、AD/FD 中位数 1.03；**乘算色**：符号 100%、相关 0.9987、1.045。

性能（RTX 5080 Laptop，**测量时 GPU 被其他代理占用约 60%**，数字偏慢；显存峰值为 PyTorch 分配器峰值，含 0.85 GB 贴图；
进程用 `set_per_process_memory_fraction` 限在 4 GB）：

| 配置 | 前向 | 前向 + 反向 | 显存峰值 |
|---|---|---|---|
| 640×360，1 帧，边缘梯度全开 | 0.36 s | 0.76 s | 1.91 GB |
| 640×360，1 帧，无边缘梯度 | 0.36 s | 0.68 s | 1.73 GB |
| 1280×720，1 帧，边缘梯度全开 | 0.41 s | 1.16 s | 3.47 GB |
| 1280×720，1 帧，边缘梯度只给前 8 层 | 0.40 s | 1.15 s | 2.33 GB |
| 1280×720，1 帧，无边缘梯度 | 0.45 s | 1.08 s | 1.74 GB |
| 640×360，32 帧，每次 1 帧，边缘梯度全开 | 6.4 s | 23.0 s | 1.94 GB |
| 640×360，32 帧，每次 4 帧，边缘梯度全开 | 7.0 s | 11.0 s | 3.54 GB |
| 640×360，32 帧，每次 8 帧，无边缘梯度 | 5.8 s | 8.2 s | 2.52 GB |
| 640×360，32 帧，每次 8 帧，边缘梯度前 8 层 | — | OOM（> 4 GB） | — |

GPU 空闲时单帧测过一次：640×360 前向 0.20 s、前向 + 反向 0.33 s；1280×720 前向 0.22 s、前向 + 反向 0.48 s。
剖析（640×360 前向）：CUDA 时间约 80 ms，其余是 CPU 端开销——约 1.3 万次小内核启动、约 150 次与数据相关的同步
（深度剥离每层要知道片元数；WSL 下一次同步约 0.4 ms）。这部分与批大小无关，所以一次多帧摊得越多越划算；显存是批大小的上限：
每帧每个剥离层要保留一张 rast（640×360 约 3.7 MB）给边缘梯度项，再加压缩的片元数据。
mask 缓冲（2048² 级别，本帧约 670 万像素）在不求导时逐帧画，显存与批大小无关。

进一步提速的方向（没做）：把片元着色写成一个融合的 CUDA / Triton 核（现在是几百个 PyTorch 小算子 × 25 层）；
mask 缓冲在相邻帧之间变化很小，拟合时可以缓存、只在 mask drawable 移动超过阈值时重画。

## 未对齐 / 未验证的特性

- **Cubism 5.3 混合模式与 offscreen**（`blendModeEnabled`、`ColorBlend/AlphaBlend`、部件 offscreen）：本模型没有用到（moc 版本不含），
  `render()` 只实现 Normal / Additive / Multiplicative 三种兼容模式；遇到 `blendModeEnabled = true` 或 offscreen 的模型需要补。
- **高精度 mask**（`useHighPrecisionMask`）：runtime 没开，未实现。
- **背面剔除**：已实现，本模型无剔除 drawable，未用真实数据验证。
- **MSAA**：只对齐 MSAA 0（本机 runtime 的统一设置）；MSAA 4 时 runtime 的边缘是 4 采样覆盖率，这里前向不模拟。
- **模型颜色 / 模型不透明度**：runtime 恒为 (1,1,1,1)，未作为输入。
- **frameHold**：属于「显示哪一帧」的调度，dump 里已映射到 g，渲染器不涉及。
- 自身重叠三角形的叠加次序（见残差 4）。
