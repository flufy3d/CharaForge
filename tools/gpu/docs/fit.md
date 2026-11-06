# GPU 整段轨迹联合优化（A5）

写于 2026-11-06。用 A1 变形器、A2 物理和 A3 光栅化拼成一条全在 GPU 上、可求导的 Live2D 前向链路，按**拟合单元**
（continues 链合并的镜头，一个或多个图层）对整段轨迹联合优化，结果导出成 scene.json 图层。
**最终指标只认 runtime**：导出后用 `pnpm render` 渲染，再用 `tools/qa/metric.py`、`tools/qa/flicker.py` 打分。

## 文件

| 文件 | 内容 |
|---|---|
| `fit/rig.py` | 模型装配：moc3（A1，缺省 float64，可复现用）、physics3（A2，float64 + `torch.compile`）、drawable 静态数据（直接从 moc3 生成）；虹膜中心标定；变形器 CUDA graph（批大小分桶） |
| `fit/vts.py` | VTS 映射与平滑的 torch 版本（语义同 `tools/fit/src/vts.ts`，与 TS 版对拍） |
| `fit/plan.py` | 拟合单元与图层规划：读 `a5_plan.ts` 的输出（= `tools/fit/src/plans.ts` 的 fitUnits + planAuto），补上新画面节点 / frameHold / 关键帧 / 锚点 / 帧类别；参考帧解码 |
| `fit/traj.py` | 单个图层的参数化与可微分前向；多图层的批量物理 `physics_batch` |
| `fit/optimize.py` | 单元级优化：合成渲染、光流目标（按遮挡分配给图层）、内层物理 + 变形、前倾 / 小实例尺度多假设、开关切换帧估计、外观常量 |
| `fit/loss.py` | SSIM（与 metric.py 逐项一致）、metric 口径的遮罩与合成、DIS 光流、粗尺度光度项 |
| `fit/export.py` | 导出 scene.json 图层，格式同 `traj2.ts` 的 `buildLayer`（clip、opacity、scaleX、逐帧 blur、开关斜坡） |
| `fit/tracks.py` | 场景图层求值的 numpy 参考实现（规范 a～g），用来核对导出语义 |
| `scripts/a5_plan.ts` | 调用 plans.ts 生成单元规划 JSON（逐帧量按 buildLayer 规则采样好） |
| `scripts/a5_fit.py` | 命令行：拟合一个单元，写出 `layers.json`、`fit.json`、`fit.log`、`preview/` |
| `scripts/a5_merge.ts` | 合并进临时全片场景（替换单元内全部镜头的图层），`parseScene` 校验通过才写出 |
| `scripts/a5_check.py` | 导出一致性对拍：GPU 前向按场景图层渲染 1280×720，与 runtime 同一帧比较 |
| `tests/test_a5_vts.py`、`tests/test_a5_loss.py` | VTS 与 TS 对拍、SSIM 与 metric.py 对拍 |

## 命令

```bash
cd ~/CharaForge && export PATH=~/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
# 拟合（镜头可以是单元内任一镜头；长任务用 systemd-run，内存 ≤ 8G）
systemd-run --user --unit=a5-fit --collect --working-directory=$HOME/CharaForge "--setenv=PATH=$PATH" -p MemoryMax=8G \
  tools/.venv/bin/python -I tools/gpu/scripts/a5_fit.py --shot s41 --out .data/gpu/fit/runs/s41
#   --quick（迭代 ÷4）  --seed 0  --fast 1 --graph 1 --deform64 1（缺省；A7 提速与可复现）
#   --switch-after F1（开关估计的时机）  --no-lean-trial  --deterministic 0
# 合并（--shot 给单元内全部镜头）→ runtime 渲染 → 打分 → 对照视频
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_merge.ts \
  --base scenes/deepseek-chan/scene.json --shot s41,s42 --layer .data/gpu/fit/runs/s41/layers.json --out .data/gpu/fit/runs/s41/scene.json
pnpm render --scene .data/gpu/fit/runs/s41/scene.json --frames 3478-3667 --out .data/gpu/fit/runs/s41/render
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/gpu/fit/runs/s41/render --shots s41,s42 --out .data/gpu/fit/runs/s41/metric.json
tools/.venv/bin/python -I tools/qa/report.py --shots s41,s42 --renders .data/gpu/fit/runs/s41/render --out .data/gpu/fit/runs/s41/report
# 以上四步的封装（调试用）：~/cf-scratch/a5/runfit.sh <镜头> <输出目录>
```

## 规划（与 plans.ts / assemble.ts 对齐）

规划不在 Python 里重写，而是 `a5_plan.ts` 直接调用 `fitUnits` + `planAuto`（同一份代码），逐帧量按 `traj2.ts buildLayer` 的规则采样：

- **拟合单元 / continues 链**：continues 连起来的镜头合成一个单元；同一条链 = 一个图层，物理从图层 in − 90 帧预热、不重置。
  合并场景时替换单元内全部镜头的图层（`a5_merge.ts --shot s41,s42`）。
- **clip**：circleWindow 作用的实例为圆形开窗（关键帧区间内裁切），其余为画中画圆角矩形（整层裁切）；clip 关键帧的不透明度作为图层 opacity。
  GPU 前向按 A3 的 `clip_coverage` 计算覆盖率（屏幕像素 × 分辨率比例），导出时逐帧烘焙，与 buildLayer 相同。
- **opacity 轨**：轨迹 opacity（没有 clip 时）。
- **景深 / 模糊段**：depthOfField 效果，或 blurSigma ≥ 1.5 的区间，blur = dofK × 强度曲线；dofK 在外观阶段拟合，段外为常量 blur。
- **cardFlip**：scaleX 曲线（to_screen 里乘在 x 方向，导出 `transform.scaleX`）。实现了，但本轮没有在 s15 上跑。
- **开关**：variant 写常量轨，switchEvents 写余弦斜坡（from = 切换前累计值）。切换帧与 fade 由拟合器估计，标注值只作初值（见下）。
- 新画面节点：参考帧重复检测（移植 `detectNewFrames`）→ cadence 覆盖 → frameHold 段（`holdSegments`）。
  另外补了一条判据：d < 2.5 且 d < ±3 帧内最大 d 的 0.2 倍时判为重复。原判据会漏掉三帧以上的保持（s21 2092～2094），
  原视频静止、渲染却在动，产生闪烁尖峰。

## 前向链路

面捕输入（每个新画面帧一个节点，节点间线性插值）→ VTS 映射与平滑 → prePhysics（常量、LeanInput、Clothes2、开关斜坡、VTS 输出，
set 语义夹取、float32）→ 预热 90 帧（输入取帧 in）→ A2 物理（dt = 1/60，float64；多图层一次批量推进）→ postPhysics add 偏移
（姿态输出与物理写入的全部参数，每节点一个值）→ A1 变形 → A3 光栅化 → 滤镜（color（规划要求时）→ resample → blur（逐帧））
→ clip × opacity → 按图层顺序合成。显示帧 f 上，图层 L 用 g = hold_L(f) 的状态。导出的每条曲线都是 GPU 前向逐帧用的数。

物理梯度按 A2 的建议截断：预热帧在 no_grad 里跑完、状态 detach；正式段每 24 帧 detach 一次。
后发、缎带等次级物理不靠长程梯度，改由 postPhysics 低维偏移直接控制。

## 目标与优化

1. **初值**：各图层变换由 shots 眼睛求相似变换（按当前姿态的实际虹膜中心，`reanchor`）。
2. **小实例尺度多假设**：眼距 < 90px 的实例，shots 的眼睛锚点常有几成的尺度误差。例如 s21.side 标注眼距 50px，画面上约 35px。
   光流拉不回整体缩放，所以把 ln s 乘以 ×{0.6, 0.7, 0.85, 1, 1.15}，在该实例 shots bbox 内比较粗尺度光度项，取最好的。
   比较时合成到中灰底上而不是参考帧上，否则画得越小越像参考帧，s25 的第一版就因此错选了 ×0.6。
3. **前倾多假设**：规划没固定前倾的图层，先用粗尺度光度项比较 {0, −10, −20, −25, −30}，再取前两名各跑 4 次光流粗对齐，按 SSIM 选定。
4. **光流对应阶段 F1 / F2 / F3**（320 → 640 → 640，Cauchy 尺度 12 → 5 → 3px）：
   外层渲染**合成图**，求合成图到参考帧的 DIS 光流；每个图层的顶点只在「自己在最前面」处取目标（自己的输出 α > 0.5，上方图层合成 α < 0.5）。
   内层 8 步 Adam 只跑物理和变形（不光栅化），顶点走向目标，加锚点项和二阶差分时间正则。
   长单元每次外层只渲染一个轮换子集（≤ 40 帧），外层次数按子集数放大（最多 3 倍）。
   - 多图层为什么在合成图上联合求光流、按遮挡分配：参考帧只有一张，实例之间互相遮挡、重叠。单独拿一个图层去对参考帧求光流，
     被别的实例盖住的地方会给出错的对应。合成图与参考帧结构一致，光流可信，再按「谁在最前」把目标归到图层。
5. **开关切换帧估计**（F1 之后）：标注的 est / fade 只作初值。在 est ± 6 帧内逐帧搜起点，在 {0, 3, 6, 9, 12, 18, 30} 里搜 fade，
   再对起点 ±2 复核。每个候选整段前向（只重画被搜索的图层，其他图层复用），数据项是该图层覆盖范围内的 SSIM@360 口径。
   最好的候选比标注只好不到 0.001 时保留标注值。放在 F1 之后，是因为 F2 起有逐节点的 postPhysics 偏移，
   姿态会去迁就标注的切换时刻，得分曲面在标注值处形成一个人为的尖峰（s38 实测）。
6. **外观常量**：在 1280×720 上缓存图层缓冲，只对滤镜求导（blur、dofK、颜色），resample ∈ {1, 0.8, 0.6} 离散搜索，
   数据项是 metric 口径的 SSIM@360。blur 上限 2px@720：几何没对齐时 SSIM 偏爱模糊，再大会把表情抹掉，目测帧上看不清。

## 可复现

固定种子（Python / numpy / torch / OpenCV），并做了以下三处处理，同一配置跑两次 runtime 指标一致到 1e-5：
- **变形器用 float64**：A1 的 float32 稀疏矩阵乘不是逐次确定的（同一输入两次差约 0.0065px，A7 也确认了）。
  这点噪声会经过光流目标的阈值判断（前后向一致性、α > 0.5）被放大。float64 下光栅化与梯度都逐位相同，速度差不多（Triton 核支持 float64）。
- **梯度规整**（`_canon_grads`）：A1 的 Triton 反向与 A2 物理的反向用原子加，float64 低位随求和顺序变化，Adam 会把它放大。
  每步保留 24 位有效数字，并把小于该张量最大值 1e-10 的分量置 0。
- **外观阶段打开确定性算子**：cuDNN 卷积反向。这一段很便宜。几何阶段不开，开了内层会慢约 3 倍，而且有了上面两条已经不需要。
- 结果：s24 两次（s24f / s24g）的几何（变换、全部参数轨）逐位相同。外观常量在加确定性开关之前差约 1e-4，
  runtime 普通帧均值 0.807908 vs 0.807910，特效帧 0.627743 vs 0.627743。
  加之前（float32 变形器）两次相差 0.010（0.817 vs 0.827），只有 float64 变形器、没有梯度规整时相差 0.0044。

## 结果（runtime 渲染 + metric.py + flicker.py；冷启动；shots.json 为 44497c3 时的版本）

见 `.data/gpu/fit/runs/<名字>/metric.json`；旧引擎为现有 `scenes/deepseek-chan/scene.json` 渲染后打分（`.data/gpu/fit/old/`）。

单元 / 帧类别：有效帧数，SSIM@360 均值 / 最低 / ≥ 0.6 帧数；闪烁 = flicker.py 尖峰帧数（单元内各镜头之和）。

| 单元（运行） | 普通帧 新 / 旧 | 目测帧 新 / 旧 | 特效帧 新 / 旧 | 闪烁 新 / 旧 |
|---|---|---|---|---|
| s24（s24f） | 50：**0.808** / 0.729 / 50 ← 0.765 / 0.711 / 50 | — | 24：0.628 / 0.363 / 16 ← 0.526 / 0.344 / 8 | 0 / 0 |
| s41+s42（s41g） | — | 173：**0.656** / 0.410 / 127 ← 0.545 / 0.293 / 66 | 15：0.491 / 0.298 / 3 ← 0.328 / 0.259 / 0 | **0** / 18 |
| s21（s21g） | — | 215：**0.592** / 0.388 / 105 ← 0.464 / 0.142 / 29 | 8：0.222 ← 0.114 | **0** / 33 |
| s38+s39（s38g） | — | 77：0.433 / 0.314 / 0 ← 0.343 / 0.275 / 0 | 10：0.373 ← 0.305 | 0 / 3 |
| s25～s28（s25f，画中画 clip + 开关） | 162：0.549 / 0.349 / 54 ← **0.647** / 0.427 / 119 | 25：0.436 ← 0.323 | 10：0.394 ← 0.365 | 4 / 1 |
| s23（s23，开关 fade + 模糊段） | 64：**0.739** / 0.380 / 56 ← 0.614 / 0.346 / 43 | — | — | 0 / 0 |

- s41 3540～3560：旧引擎在这里有 18 个闪烁尖峰（锚点把变换拉成脸部特写并来回跳）；新引擎半身与全身两个实例都稳定，没有特写跳变
  （`.data/gpu/fit/runs/s41g/shots/s41_3540_3560.jpg`，对照视频 `s41g/report/diff_sbs.mp4`）。
- s21：小实例 s21.side 的尺度多假设选了 ×0.70（与目测一致）。同一单元早一版（只缺尺度多假设与保持检测补充判据）是 0.470，3 个尖峰在 2093、2094。
  main 在 1990～2130 原视频双手合十，渲染里没有手（Movement2 切换在本模型上没有可见效果，需要别的参数；规划 / shots 的问题）。
- 开关切换帧估计：
  - s38.main（Clothes2，标注 3357 / fade 9）：在 F1 之后估计，得分曲面在 3357 / 9 处有清楚的极小（±3 帧之外 1−SSIM 高 0.01 以上），估计 = 标注。
  - **s38.side（Q 版，标注 3346 / fade 8，期望约 3348～3350 起、约 6 帧）：估计 3340 / fade 0，不对**。得分随起点往前单调变好、
    顶到搜索窗口下界，说明不是在找切换时刻：这个小实例（眼距约 40px）的几何没有对上（锚点偏差约 50px），
    Q 版只是更「像」那块区域。在几何对上之前，这个估计不可信（见剩余问题）。
  - s23 key8（标注 2318 / fade 30）：上一版（F2 之后估计、整幅 SSIM）估计为 2310 / fade 0；key3 保持标注。
  - s41 Movement3、s21 Movement2、s25 key5 / key7 / key12：切换前后没有可见差别，保留标注。

## 推广到全片的剩余问题

- **s25（画中画 clip）退化**：普通帧 0.549 vs 旧引擎 0.647。渲染出来的角色比画面里大约一成，头发伸出画中画窗口上沿；
  F3 末尾光流残差 1.2（s24 约 0.3），锚点偏差 24px；外观阶段 blur 顶到上限 2px（SSIM 偏爱模糊，说明亚像素对齐没到位）。
  fit4 在这个单元上用了约 4300 s 的逐帧精修。待查方向：窗口外被 clip 掉的顶点不进光流（只剩窗口内的局部），
  缩放只靠脸部约束、不稳；可以把 clip 边缘的结构也作为约束，或者对画中画实例同样做尺度多假设。
- **无画面效果的开关**：s41 的 Movement3、s21 的 Movement2、s25 的 key5 / key7 / key12，在本模型上切换前后渲染没有可见差别
  （整段 SSIM 的差 < 1e-6）。估计保留标注值。
- **cardFlip 与圆形开窗**（s15）实现了，但本轮没有跑。

## 耗时与全片估算

| 单元 | 帧数 | 图层 | 节点 × 图层 | 拟合耗时 | 显存峰值 |
|---|---|---|---|---|---|
| s24（s24f / s24g） | 74 | 1 | 37 | 266 s / 261 s | 4.6 GB |
| s23 | 64 | 1 | 32 | 258 s | 5.1 GB |
| s38+s39（s38g） | 96 | 2 | 157 | 913 s | 5.4 GB |
| s25～s28（s25f） | 197 | 3 | 109 | 1098 s | 4.5 GB |
| s21（s21g） | 227 | 2 | 374 | 1258 s | 5.6 GB |
| s41+s42（s41g） | 190 | 2 | 373 | 1313 s | 6.1 GB |

s24 的分项（s24f）：装载 26 s、前倾多假设 58 s、F1 38 s、F2 58 s、F3 42 s、外观 41 s。接入 A7 之前同一流程 519 s，
也就是说 A7 的光栅化快速路径、变形器 CUDA graph 和更大的渲染批，把 s24 降到约一半。
s41 的分项：内层（物理 + 变形）787 s（60%）、外层渲染 264 s、外观 145 s。

**全片估算**：长单元约 3.5 s / 节点 × 图层，短单元约 7 s（固定开销占比大）。按 28 个单元、约 6000 图层帧、
约 75% 是新画面估算，单卡串行约 **4.5～5 小时**。

**离 1 小时还差约 5 倍**：
1. **物理**是内层的大头（每步对 T ≈ 300 帧逐帧推进，再整段反向），耗时取决于内核发射次数，与批大小几乎无关。
   可以对 `rig.compile()` 的单帧函数用 `mode="reduce-overhead"`（CUDA graph），或者把截断窗口内的 24 帧整体捕获成一张图，估计能快 2～3 倍。
2. **多单元批处理**：物理、变形、光栅化都对 B 批量。把 4～8 个单元的图层摞成一批（各自的帧对齐到自己的起点），
   内核发射与同步摊薄，GPU 目前远没跑满（光流阶段 nvidia-smi 占用约 30～50%），估计 3～4 倍。DIS 光流在 CPU 上，可与 GPU 并行。
3. 次要的：外层次数可按收敛提前停止（s24 的 F3 后半段 SSIM 只涨 0.01）；前倾两支试拟合在小实例或已固定前倾的单元上不需要。
