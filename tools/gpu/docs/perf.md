# 前向模型提速（A7）

写于 2026-11-06。对象：A1 变形器（`l2d/deform.py`）、A3 光栅化（`l2d/raster.py`）与合成（`l2d/composite.py`）。
接口保持兼容，拟合器不改代码也能跑；默认结果与改动前一致（见「正确性」）。加速分两类：
一类缺省就开，拟合器自动受益；另一类要显式打开（见「拟合器怎样用」）。

## 结论

| 项目 | 改动前 | 改动后 | 目标 |
|---|---|---|---|
| deform B=64 前向 + 反向 | 294 ms（空闲）/ 547 ms（GPU 被占 ~80%） | **50 ms**（CUDA graph，GPU 被占 ~95%）；不开 graph 147～159 ms | ≤ 0.15 s ✔ |
| raster 640×360 单帧前向 + 反向（含边缘梯度） | 0.50 s（空闲）/ 0.84 s（被占） | **0.061 s**（空闲）/ 0.10～0.13 s（被占 ~89%） | ≤ 0.1 s ✔（空闲时） |
| 640×360、带边缘梯度，一次能放的帧数（4 GB） | 4 帧（8 帧 OOM） | **28 帧**（32 帧 OOM） | ≥ 16 ✔ |
| 32 帧前向 + 反向（640×360，带边缘梯度） | 11.5 s（每次 4 帧，被占） | 2.5～2.7 s（每次 16～28 帧，被占） | — |

计时都是中位数。WSL 下另有拟合代理在用 GPU，表里注明了测量时 nvidia-smi 的整卡占用。原始数据在 `~/cf-scratch/a7/final/*.json`
和 `~/cf-scratch/a7/logs/final_bench.log`（由 `a7_bench.py` 生成）。

## 拟合器怎样用（拿到最大收益）

```python
from l2d import raster, composite
raster.FAST_DEFAULT = True        # render() 缺省走 Triton 快速路径（也可以每次传 fast=True）
composite.CHECKPOINT = True       # 求导时滤镜做分帧检查点（也可以传 checkpoint_filters=True）
rig.l2d.auto_graph = True         # L2D.forward 自动走 CUDA graph（或显式 model.graphed(B, grad)）
# 然后把每次光栅化的批加大：640×360 带边缘梯度时 16～24 帧一批（28 帧是 4 GB 的上限）
```

- `auto_graph`：每个（B，是否求导，是否传 part_opacity）组合建一张图（约 1～2 秒），最多缓存 4 个，每张图占一份私有显存池
  （B=64 求导约 0.5 GB）。所以 B 要固定，比如物理 + 变形内层循环的 B 就是节点数，每步都一样。
  同一张图上一次前向的反向还没跑完又做前向时，会自动退回普通 forward，结果正确，只是慢。
- 快速路径的反向只能跑一次（保存的剥离层数据在反向后就释放）；需要 `retain_graph` 的地方要传 `fast=False`。
- `mask_grad=True` 求导时自动走原实现。
- 内层 B=37 的变形前向 + 反向：约 0.3 s 降到约 0.03 s（graph）；外层不求导渲染 37 帧：约 5 s 降到约 1 s。
  按 s24 的分项估算（未实测），拟合会从 519 s 降到约 250 s，剩下的大头是物理（A2）。见文末「离 1 小时还差什么」。

## 改了哪些地方

| 文件 | 改动 | 缺省开？ |
|---|---|---|
| `l2d/_kernels_deform.py`（新） | warp / rotation 变换的前向与反向 Triton 融合核（关闭 FMA 合并，IEEE 除法），一个核代替原来每段约 100（前向）/ 200（反向）个小算子 | 是（`deform.USE_TRITON`） |
| `l2d/deform.py` | `_WarpApply` / `_RotApply` 调用上面的核；`torch.prod` 换成逐项连乘（prod 的反向会同步，不能被 graph 捕获）；稀疏矩阵乘换成自定义 Function，反向用预先转好的 Aᵀ（原生反向每次现场转置，还有 CPU 拷贝）；前向里的常量张量预先建好；新增 `graphed()`、`auto_graph` | Triton 核缺省开；graph 是可选的 |
| `l2d/_kernels_raster.py`（新） | mask 剥离层核、主绘制合成核、两遍反向核（见下） | 否 |
| `l2d/raster.py` | `render(..., fast=None)`，以及 `FAST_DEFAULT` / `FAST_*_BUDGET` / `FAST_AA_GROUP`；`_render_fast`、`_render_masks_fast`、`_clip_matrices_fast`（向量化）、`_FastRender`、`_aa_backward` | 否（`FAST_DEFAULT = False`） |
| `l2d/composite.py` | `composite(..., checkpoint_filters=None)`、`apply_filters(..., checkpoint_filters)`；`CHECKPOINT`、`CHECKPOINT_FRAMES` | 否 |
| `scripts/a7_bench.py` | 耗时 / 显存测量、profile（`--l2d` 指向旧副本即可测改动前） | — |
| `scripts/a7_fast_run.py` | 打开快速路径后运行任意脚本（a3_compare、测试），不改那些脚本 | — |
| `scripts/a7_gradcheck.py` | 梯度对照：a3_gradperf 的差分检查，原实现与快速路径各跑一遍，再直接比较 autograd 结果 | — |

## profile 结论（改动前）

`a7_bench.py profile-deform|profile-raster`（torch.profiler，空闲 GPU）：

- **deform B=64 前向 + 反向**：37k 个内核、GPU 时间 118 ms，平均每个内核约 3 µs。B=1 时 CPU 发射是瓶颈，B=64 时内核个数本身就是瓶颈。
  热点是 warp 变换（`_warp_core` 每段几十个 `where` / `gather`，反向分块循环再翻倍），占内核数的大半；另有 1 次同步（prod 的反向）。
- **raster 单帧前向 + 反向**：33.7k 个内核、GPU 时间 174 ms、32k 次 `cudaLaunchKernel`、191 次 `cudaStreamSynchronize`、64 次可分页内存 H2D 拷贝
  （mask 打包的 `slot_map`，CH×CW 的 int64，每帧约 12 ms）。GPU 时间里最大的是 `index_put` / `indexing_backward_kernel`（共 86 ms，
  合成里每层的 `D.index_put` 和 gather 反向用的是排序式确定性累加），然后是几千个小的逐元素算子。
  mask 缓冲（2048 级，每帧约 670 万像素）在不求导时逐帧画，8 帧前向里约 75% 的时间花在 mask 上。
- 改动后单帧前向 + 反向：1186 个内核、GPU 时间 30 ms，其中光栅化（nvdiffrast，含 mask 画布）13 ms。8 帧不求导前向的 GPU 时间 167 ms，
  光栅化占 68%（大多是 2048 级 mask 画布），合成核只有 1.9 ms。nvdiffrast 的 CUDA 光栅化每次调用要同步一次（读回内部计数），
  每层剥离一次调用，这部分开销与帧数无关，所以批越大越划算。

## 实现说明

### 变形器

- warp / rotation 的 Triton 核与原 PyTorch 实现逐式相同。实测前向逐位相同；反向里对网格 / 仿射项的散射累加用原子加，
  只有求和顺序不同（原来的 `scatter_add_` 也是原子加）。
- CUDA graph：`graphed(B, grad)` 用 `torch.cuda.make_graphed_callables`（求导）或 `torch.cuda.graph`（不求导）。输出从静态缓冲复制出来，可以放心保留。
  能捕获的前提是前向和反向都没有 GPU→CPU 同步、也没有可分页内存拷贝，所以去掉了 prod、sparse.mm 的原生反向和前向里的 `torch.tensor(...)`。
- float32 下 `torch.sparse.mm`（warp 网格的 keyform 插值，A 有 300 万个非零元）本身不是逐次确定的：同一输入两次 eager 的顶点差 0.0065 px。
  graph、Triton 与 eager 之间的差都在这个量级。

### 光栅化快速路径（`render(fast=True)`）

1. **mask 缓冲**：窗口与原实现相同。打包时画布宽在 {1024, 2048, 4096} 里取面积最小的（只取 2 的幂，像素换算成裁剪坐标时不引入除法误差），
   画布面积比固定 1024 宽少约 22%。归属判断改用槽位矩形（不再拷贝 CH×CW 的 `slot_map`）。
   每个剥离层一个 Triton 核，在整幅画布上逐像素算重心、uv / LOD、贴图 alpha，再乘进 Π(1 − a)，同时判断本层是否为空。
   一次只剥离一帧（`FAST_MASK_BUDGET`），显存与批大小无关。
2. **主绘制**：深度剥离只留下每层每像素的三角形号（int32，640×360 每帧每层 0.9 MB）。之后一个合成核对每个像素从后往前走完所有层，
   每层现场着色（重心、uv、三线性采样、乘算 / 屏幕色、mask 采样、剔除、混合系数）再混合，每步夹取，按需 8 位取整。没有片元压缩，也没有逐层同步。
3. **反向**（`_FastRender`，按 `FAST_BWD_BUDGET` 分帧块）：
   - 第一遍从后往前，记录每层混合前的 D_below；前 LA 层还要记下 X = B_k、A_k，供边缘梯度项使用。
   - 边缘梯度项：在这里重新剥离出 rast，按 `FAST_AA_GROUP` 层一组，把各层当成 batch 一次调用 `dr.antialias`，求 VJP，
     得到 ∂X（加到 B_k 上）和 ∂裁剪坐标（换算后加到屏幕坐标上）。公式与原实现相同：extra = T_above ⊙ (Δ − Δ_α ⊙ D_below)。
   - 第二遍从前往后：夹取遮罩 → ∂B_k、∂A_k → 手写的着色反向，原子加到顶点屏幕坐标、模型单位坐标（经 mask 采样位置）、不透明度、乘算色和屏幕色。
4. **与 PyTorch 逐位一致要注意的三件事**（都实测过）：
   - Triton 的 `/` 是近似除法，与 PyTorch 不逐位相同，要用 `libdevice.div_rn`；
   - 要关闭 FMA 合并（`enable_fp_fusion=False`）；
   - 本机 torch 2.14 的 `clamp(0, 1)` / `clamp(min=0)` 反向在端点处**为 0**（只有 0 < v < 1 时才传梯度），`minimum` 在相等处各分一半。
     不透明片元的 alpha 混合结果正好等于 1，所以这条约定决定了不透明度梯度：照抄之前，与原实现的相对误差是 50%，照抄之后是 1e-6。
5. **显存**：求导时每帧要保存三角形号约 23 MB 和 mask 画布约 32 MB。快速路径的状态在反向结束后立即释放；不这样做的话，
   只要输出张量还在，就一直占着显存，上一批的约 1 GB 会叠到下一批上。
6. `composite.CHECKPOINT`：每个图层的「滤镜 + 输出 pass」整体做检查点，每 4 帧一段（`CHECKPOINT_FRAMES`），反向时重算。
   有逐帧张量参数时不分段，因为 blur 半径取整批的最大值。s24 的 color + resample + blur 每帧保存的中间量从约 47 MB 降到约 7 MB。

## 正确性

改动后的代码，全部在 RTX 5080 上重跑。

**A1 对拍**（`a1_compare.py random special scene partop`，Triton 核缺省开）：全部通过；render order、draw order、可见性、vertex_updated 全部一致。

| 参数集 | float64 顶点 max（改动前） | float32 顶点 max（改动前） | 不透明度 / 乘算 / 屏幕 max |
|---|---|---|---|
| random | 0.0434（0.0434） | 0.0450（0.0450） | ≤ 6e-7，与改动前相同 |
| special | 0.0633（0.0633） | 0.0718（0.0718） | 同上 |
| scene | 0.0224（0.0224） | 0.0243（0.0243） | 同上 |
| partop | 0.0228（0.0228） | 0.0229（0.0340） | 同上 |

float64 的各项 max / p99 / p99.9 与改动前逐项相同。float32 的 p99 差在 1e-4 px 以内，来源是上面说的 sparse.mm 不确定性。
deform 梯度（`a1_grad_bench.py grad`，float64，24 个参数）：autograd 与差分的最大相对误差 5.7e-9（改动前 5.7e-9）。
graph 与 eager 的梯度差（2.6e-6 相对）与 eager 自己两次之间的差同量级。

**A3 对拍**（`a3_compare.py --dump main --dump variants`，58 帧）：

| 路径 | SSIM@360 最小（帧） | SSIM@360 均值 | SSIM@720 最小 / 均值 | 预乘 RGBA8 平均误差 | 最大误差 |
|---|---|---|---|---|---|
| 改动前（`.data/gpu/a3_compare.json`） | 0.999343（f2650） | 0.999600 | 0.99736 / 0.99854 | 0.131/255 | 113 |
| 缺省路径 | 0.999343 | 0.999600 | 0.99736 / 0.99854 | 0.131/255 | 113（逐帧与改动前完全相同） |
| 快速路径（`a7_fast_run.py -- a3_compare.py …`） | 0.999343 | 0.999600 | 0.99736 / 0.99854 | 0.131/255 | 113 |

快速路径与缺省路径逐帧的 SSIM@360 之差 ≤ 2.8e-8。前向逐像素对比（s24 三帧）：mask 画布按原来的固定 1024 宽打包时，
1280×720 与 640×360、quantize 开 / 关全部逐位相同；换成新打包后，因为 mask 三角形在 2048 级画布上的定点光栅化位置不同，
每帧有 5～33 个通道值差 1/255（quantize 开）、或差 ≤ 1.7e-4（quantize 关）。

**单元测试**：`test_a1_*` 8 个、`test_a3_*` 10 个（缺省路径，以及用 `a7_fast_run.py` 打开快速路径各跑一遍）、`test_a2_*` / `test_a5_*`，全部通过。

**梯度**（`a7_gradcheck.py`，s24 帧 2385，640×360）：

| | 原实现 | 快速路径 |
|---|---|---|
| 顶点：方向导数 vs 差分（ε = 0.5 / 1 / 2 px），符号一致 | 33/33 | 33/33 |
| 相关系数 | 0.99907 / 0.99883 / 0.99755 | 0.99914 / 0.99891 / 0.99767 |
| AD/FD 中位数 | 1.074 / 1.088 / 1.112 | 1.080 / 1.089 / 1.112 |
| 朝目标的方向导数 | −5.16e−4 | −5.15e−4 |
| 不透明度 / 乘算色 相关系数 | 0.99982 / 0.99866 | 0.99983 / 0.99866 |
| 图层平移 x：AD（差分 ε 0.5） | −3.350e−4（−2.865e−4） | −3.354e−4（−2.865e−4） |

直接比较 autograd 结果（同一输入、同一损失）：
- 不带边缘梯度项时，顶点、to_screen、不透明度、乘算色、屏幕色的相对误差在 2e-5～1e-4 之间，cos ≥ 0.999999996。
- 带边缘梯度项时，不透明度和颜色仍一致（2e-5）。顶点与 to_screen 的差（0.17～0.40）和原实现自己连跑两次的差同量级（0.23～0.67）。
  原因是 nvdiffrast `antialias` 对顶点的反向用原子加，抵消很大，逐次结果就不确定。

## 耗时与显存

**deform**（float32，scene 参数）。改动前后是同一时段连续测的，测量时 GPU 被其他代理占用 58～98%；括号里是第一次测量时（GPU 较空）的改动前数据。
B=512 前向 + 反向在 4 GB 里放不下，按 2×256 分块（改动前后都一样）。

| B | 改动前 前向 | 改动前 前向+反向 | Triton（缺省）前向 | Triton 前向+反向 | graph 前向 | graph 前向+反向 | 前向+反向显存峰值（前 → 后） |
|---|---|---|---|---|---|---|---|
| 1 | 180 ms（91） | 466 ms（255） | 36～46 ms | 154～160 ms | 2.7～3.3 ms | 12.3～12.7 ms | 192 MB → 8 MB |
| 64 | 176 ms（96） | 547 ms（294） | 44～61 ms | 147～159 ms | 7.7～8.7 ms | **49.5～50.4 ms** | 613 → 434 MB（graph 491 MB） |
| 512 | 579 ms | 2.28 s（2×256） | 66～72 ms | 0.66～0.68 s | 42～44 ms | 0.41～0.44 s | 1.96 → 1.74 GB |

**raster**（s24b，640×360，含 color + resample + blur 合成，quantize=False，与 a3_gradperf 的口径相同）：

| 配置 | 改动前 | 快速路径 |
|---|---|---|
| 单帧前向（GPU 空闲 / 被占 ~80%） | 0.228 s / 0.456 s | **0.042 s** / 0.086 s |
| 单帧前向 + 反向，带边缘梯度（空闲 / 被占） | 0.498 s / 0.844 s | **0.061 s** / 0.130 s（最小 0.10） |
| 单帧前向 + 反向，无边缘梯度（空闲 / 被占） | 0.456 s / 0.772 s | 0.048 s / 0.115 s |
| 单帧显存峰值（含 0.85 GB 贴图） | 1.72 GB | 1.54 GB |
| 32 帧，每次 1 帧，带边缘梯度：前向 / 前向+反向 | 13.6 s / 27.3 s | 2.6 s / 3.9 s |
| 32 帧，每次 4 帧，带边缘梯度 | 7.0 s / 11.5 s，3.45 GB | 1.57 s / 2.53 s，1.80 GB |
| 32 帧，每次 8 帧，带边缘梯度 | OOM | 1.50 s / 2.46 s，2.07 GB |
| 32 帧，每次 16 帧，带边缘梯度 | OOM | 1.61 s / 2.68 s，2.65 GB |
| 32 帧，每次 24 帧，带边缘梯度 | OOM | 1.63 s / 2.51 s，3.21 GB |
| 32 帧，每次 28 帧，带边缘梯度 | OOM | 1.62 s / 2.46 s，3.59 GB |
| 32 帧，每次 32 帧 | OOM | OOM |
| 640×360 一次最多放 | 4 帧（带边缘梯度）/ 8 帧（无） | **28 帧** |

32 帧那几行测于 GPU 被占 76～93% 时。不求导时快速路径按显存预算自动分批（`FAST_TIDP_BUDGET`），不受上面的帧数限制。

## 没有达到 / 剩余的瓶颈

- **单帧前向 + 反向 ≤ 0.1 s**：GPU 空闲时 0.061 s，达到；被其他代理占用 ~90% 时是 0.10～0.13 s。
  剩下的大头是：nvdiffrast 每次光栅化调用的同步（每帧约 55 次：mask 约 7 层，主绘制约 25 层，反向重新剥离约 24 层），
  以及 2048 级 mask 画布的光栅化。
- **批量渲染每帧约 28～50 ms**（前向，视 GPU 占用）：主要花在 2048 级 mask 画布上：每帧画布约 800 万像素（窗口面积约 670 万），剥离约 7 层，由 nvdiffrast 光栅化。8 帧不求导前向的 GPU 时间 167 ms 里，光栅化占 114 ms。
  合成核本身每帧只要约 0.25 ms。可行的方向：
  1. 只在被裁切片元会采样到的纹素上按需求 mask（估计工作量少 30 倍），但要自己复刻 nvdiffrast 的定点覆盖规则才能逐位一致；
  2. 给 640×360 拟合加一个低分辨率 mask 选项（例如 1024 级）。这会改变前向结果，所以没有做成缺省。
- **deform 的 B=512**：GPU 计算已经是主导（sparse.mm 和反向原子加）。在 4 GB 里一次放不下 512 个样本求导，与改动前一样要分块。
- **离 1 小时还差什么**（按 `fit.md` 的分项估算，未实测）：
  - s24 的光流阶段，外层渲染约 5 s 降到约 1 s，内层变形约 0.3 s × 8 步降到约 0.03 s × 8 步；
  - 物理（A2）每步约 0.6 s，没动，现在是最大的一项；
  - 估计 s24 从 519 s 降到约 250 s，全片约 3.5 小时。要到 1 小时还需要物理提速，以及多镜头批处理（变形与光栅化已经能吃 16～28 帧的大批）。

## 复现

```bash
cd ~/CharaForge && export PATH=~/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
P="tools/.venv/bin/python -I"
# 改动前的版本：保存一份旧 l2d 包，用 --l2d 指向它的上一级目录（本次在 ~/cf-scratch/a7/orig）
$P tools/gpu/scripts/a7_bench.py deform [--fast] [--l2d ~/cf-scratch/a7/orig] --batches 1,64,512
$P tools/gpu/scripts/a7_bench.py raster [--fast] [--l2d ~/cf-scratch/a7/orig] --chunks 1,4,8,16,24,28,32 --both-aa
$P tools/gpu/scripts/a7_bench.py profile-deform --B 64 [--fast]
$P tools/gpu/scripts/a7_bench.py profile-raster [--fast] [--no-grad --B 8]
# 正确性
$P tools/gpu/scripts/a1_compare.py .data/gpu/ref/{random,special,scene,partop} --dtype float32   # 以及 float64
$P tools/gpu/scripts/a3_compare.py --dump .data/gpu/frames/main --dump .data/gpu/frames/variants --out /tmp/x.json
$P tools/gpu/scripts/a7_fast_run.py -- tools/gpu/scripts/a3_compare.py --dump .data/gpu/frames/main --dump .data/gpu/frames/variants --out /tmp/y.json
$P tools/gpu/scripts/a7_gradcheck.py --json ~/cf-scratch/a7/gradcheck.json
$P -m unittest discover -s tools/gpu/tests -p "test_a[13]_*.py"
$P tools/gpu/scripts/a7_fast_run.py -- tools/gpu/tests/test_a3_raster.py
```

# 物理提速（A8）

写于 2026-11-06。对象：A2 物理（`l2d/physics.py`），新增 `l2d/_kernels_physics.py`。接口与缺省行为不变，新路径都是可选的
（`rig.fused_rollout` / `rig.graphed_rollout`，或 `rig.auto_fused = True`）。接口与语义见 `physics.md` §9。

## 结论

拟合器口径（预热 90 帧 no_grad 后，T 帧按 24 帧一窗截断 BPTT，前向 + 反向，float64，loss = 随机权重 × 物理输出参数之和）。
每格 7 次的中位数，两轮（改动前后在同一轮里接连测）。测量时另有拟合代理（A5）在用 GPU，括号里是 nvidia-smi 整卡占用；
「改动前」是旧代码副本（`--l2d ~/cf-scratch/a8/orig`），compile 版，重编译上限同样放宽到 64（否则旧代码 B=16 会静默退回 eager）。

| T | B | 改动前 compile | 改动后 compile | graphed | **fused** | fused 提速（对改动前） | 显存峰值 compile / graphed / fused |
|---|---|---|---|---|---|---|---|
| 24 | 1 | 86.3 / 66.5 ms | 46.6 / 69.0 | 18.2 / 24.4 | **9.2 / 14.1 ms** | **9.4× / 4.7×** | 3 / 1 / 4 MiB |
| 24 | 4 | 87.9 / 71.9 | 55.8 / 62.9 | 16.2 / 17.6 | **9.9 / 11.1** | 8.9× / 6.5× | 8 / 2 / 16 MiB |
| 24 | 16 | 73.3 / 67.3 | 63.4 / 66.2 | 17.5 / 18.0 | **10.4 / 11.2** | 7.0× / 6.0× | 29 / 9 / 66 MiB |
| 200 | 1 | 522 / 641 | 569 / 572 | 159 / 161 | **73 / 72** | 7.1× / 8.9× | 28 / 5 / 35 MiB |
| 200 | 4 | 524 / 583 | 515 / 483 | 140 / 139 | **72 / 73** | 7.3× / 8.0× | 68 / 20 / 132 MiB |
| 200 | 16 | 533 / 622 | 555 / 495 | 143 / 142 | **76 / 75** | 7.0× / 8.3× | 227 / 77 / 522 MiB |

GPU 整卡占用：第一轮 0～97%，第二轮 25～98%（A5 在跑）。较空闲时（同日早些时候）改动前 T=24、B=1 是 49.5 ms，按它算 fused 也有 **5.4×**。
fused 的拆分（T=24、B=1）：前向 5.1 ms、反向 4.1 ms；T=200：前向 40 ms、反向 33 ms。显存是该次调用的峰值增量
（fused 的大头是反向用的磁带，每帧每个批元素约 150 KB；graphed 另有每个图实例的私有池，见 reserved）。
原始数据：`~/cf-scratch/a8/bench_{before,after}_{1,2}.json`、`~/cf-scratch/a8/logs/final.log`。

## profile 结论（改动前，compile 版，T=24、B=1，`a8_bench.py profile`）

- 每帧约 150 个 GPU 内核（前向约 50、反向约 100），24 帧 3468 次 `cuLaunchKernel`；每帧一次编译图调用加一次编译反向调用，
  dynamo 守卫 + AOTAutograd 包装的 CPU 开销约 1～1.5 ms / 帧（墙钟 ~2 ms / 帧 中大部分）。整段没有 GPU→CPU 同步（只有计时用的同步）。
- GPU 时间约 790 µs / 帧（前向 265、反向 520）。大头是每层一个的「粒子链」融合核（30～100 µs 一个，B=1 时只有几个线程在跑），
  耗时几乎与 B 无关。
- 反向里 gather 的转置用 float64 `atomic_add`：下标有重复，求和顺序随调度变，**梯度逐次不同**（已修，见 physics.md §9.3）。
- dynamo 重编译上限 8：每种 (B, 是否求导, in_cache 是否为 None …) 各一份，用到第 3 种 B 就超，之后静默退回 eager（已放宽到 64）。

所以只把每帧捕获成 CUDA graph（graphed）只能去掉 CPU 开销，剩下 GPU 时间，约 2.5～3.5×。fused 把整段写成两个核：

- **本卡的 float64 是 1/64 速率**，实测（`~/cf-scratch/a8/t_lat*.py`，单个 warp）：每条 fp64 指令约 25 ns，与有无依赖无关
  （两条独立链也是 2 倍时间）；IEEE 除法约 240 ns、开方 250 ns、sin 510 ns、atan2 1130 ns；一个 SM 上最多约 4 个 warp 并行不减速。
  所以融合核的耗时 ≈ 单个线程里 fp64 指令条数 × 25 ns，与 B 无关（B ≤ 60 时每个批元素一个 SM）。
- fused 前向约 210 µs / 帧（globaltimer 分段：4 层的粒子链 30 / 41 / 31 / 19 µs，输出 7 / 17 / 4 / 4 µs，其余 < 10 µs）。
  每个粒子约 95 条 fp64 指令（其中 4 次除法、1 次开方），每层的 setting 级（8 次 sin / cos / atan2）约 12 µs。
- 反向用前向记下的磁带，不重算前向，约 170 µs / 帧。

## 还能做的（没做）

- 前向：把速度更新（每粒子 2 次除法）和 `v·d`、`(f·d)·d` 移出链循环、改成按 (setting, 粒子) 并行的阶段，链内每粒子约 95 → 60 条指令；
  setting 级的 sin / cos 对用 [R, 2] 摊到两倍的线程上。估计前向再快 20～30%（仍要逐位相同，运算本身不能改）。
- B=1 时只用了一个 SM：把一层的 setting 分到几个 CTA、层间用全局栅栏，可再快约 2 倍，但实现与正确性风险都大。
- 拟合器把多个单元摞成一批：物理几乎与 B 无关。

## 复现

```bash
cd ~/CharaForge && export PATH=~/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
P="tools/.venv/bin/python -I"
$P tools/gpu/scripts/a8_bench.py bench --mode compile,graphed,fused --T 24,200 --B 1,4,16      # 改动后
$P tools/gpu/scripts/a8_bench.py bench --mode compile --l2d ~/cf-scratch/a8/orig                 # 改动前（旧 l2d 副本）
$P tools/gpu/scripts/a8_bench.py profile --mode compile --T 24 --B 1
# 正确性
$P tools/gpu/scripts/a8_compare.py --path fused --dtype f64,f32io            # 11 个序列；与 eager 逐位比较（原路径差 0）
$P tools/gpu/scripts/a8_compare.py --path graphed --dtype f64,f32io [--compile]
$P tools/gpu/tests/test_a8_fused.py      # 与 eager 的前向 / 梯度 / 确定性 + test_a2_grad 29 项走 fused
$P tools/gpu/tests/test_a8_graph.py [--compile]
$P tools/gpu/scripts/a2_compare.py --dtype f64,f32io [--compile]; $P tools/gpu/tests/test_a2_grad.py
```

## 正确性（A8）

- **a2_compare 口径对拍**（`a8_compare.py`，11 个序列，f64 与 f32io）：fused（24 帧一窗；以及 1 帧一窗）、graphed（eager / compile）
  全部通过，最大误差与原路径逐项相同（f64：sweep 3.0e-8 最大，s30_full 9.3e-10，step 4.7e-10，其余 ≤ 4e-15 或 0；
  compile 版 walk_dt50 1.5e-7、walk_dt144 1.2e-7，与改动前的 compile 相同）；与原路径（同一 rig 逐帧 step）的差：全部序列、全部帧 **0**，rig 输出差 0。
- 原路径本身：`a2_compare.py`（eager / compile）11 个序列的数字与改动前逐项相同；`test_a2_physics.py` 3 项通过。
- **梯度**：`test_a2_grad.py` 原路径 29/29、改走 fused 也 29/29（跨不可导点那 1 项照旧单列不计；各项相对误差与原路径相同到显示位数）。
  fused 与 eager autograd 直接比较：对输入 ≤ 2.4e-15、对初始状态各分量 ≤ 2.6e-14（含 200 帧不截断、dt = 1/50、1/144、全新状态、
  resolved=False、emulate_f32=False）。graphed 与同一 rig 的 rollout：梯度逐位相同（eager 与 compile 两种）。
- **确定性**：fused、graphed、以及原路径（eager / compile）同一输入跑两次，前向与梯度都逐位相同。
