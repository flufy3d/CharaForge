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

## 第三阶段（2026-11-07）：s25 修复、小实例、多单元批处理、A8 物理

### 改动

- **S-SSIM 精修**（`Fitter.run_ssim_stage`，F3 之后）：数据项直接是 metric 口径的 SSIM@360（640×360，合成到参考帧上），
  对全部几何未知量求导（物理整段保留图，各图层帧的参数与 to_screen 截成叶子，分块渲染带边缘梯度 → 合成 → SSIM → 反向）。
  A7 的快速光栅化让带边缘梯度的前向 + 反向降到约 0.1 s / 帧，这一段才划算（s24 约 26 s，s25 单元约 130 s）。
- **小实例粗配准**（`Fitter.register_small`，取代上一版的 bbox 光度尺度搜索）：眼距 < 90px 的实例，每个关键帧单独渲染该图层，
  用「中灰底合成后亮度的梯度幅值」做掩码 NCC 模板匹配（shots bbox 外扩一倍的搜索区），尺度 {0.55 … 1.3} 在全图层上投票，
  位置逐关键帧取最佳匹配（分数 < 0.25 的沿用原值）。s21.side 选 ×0.65（与目测一致），s38.side 选 ×0.55、关键帧移动约 90px。
- **小实例先验**：眼距 < 90px 时 small_k = (90 / 眼距)²（最多 16），面捕输入的幅度先验与 postPhysics 偏移先验按它加权。
- **开关分组估计**：同一图层上标注在同一时刻（±1 帧）的切换编成一组，一起平移起点、各自保留标注 fade
  （s26 2530 的 key5 / key7 / key12、Movement* 连同配套手臂参数 Param36/39/40/41/59–62）；单参数组另外搜 fade。
  规划与导出本来就按参数逐条处理（a5_plan.ts → planAuto），一个切换事件里的多个参数、各自的 fade 都原样进前向与导出。
- **前倾多假设**：试拟合改成 3 个分散的假设（光度第一名 + 两个与已选值相差 ≥ 10 的），多单元时所有单元的同名次假设一起试。
- **A8 物理**：`rig.phys.auto_fused = True`（`Rig(phys_compile=False)` 缺省），torch.compile 路径不再使用。
- **多单元批处理**：`plan.make_batch` 把几个单元摞成一个虚拟单元（各单元时间上不重叠，合成、光流、数据项天然互不干扰），
  物理与变形对全部图层一次批量求值；`a5_fit.py --shot s33,s34,...` 一批跑，结果按单元落盘到 `<out>/<单元>/`。
  `Rig.deform` 的 CUDA graph 按 64 行分块（大批整批建图会占满显存，s15 三图层曾因此 OOM）。
- **批量驱动** `scripts/a5_batch.py`：fitUnits 的单元列表，按时间顺序每 `--batch` 个一批（总帧数超过 `--max-frames-per-batch` 提前截断），
  每批一个子进程；每个单元的 `layers.json` / `fit.json` 单独落盘并写 `done.json`（指纹 = shots.json + 拟合代码 + 参数），
  原样重跑即断点续跑；最后 `scripts/a5_assemble.ts` 合成 `<out>/scene.json`（parseScene 校验；没有 GPU 结果的单元沿用基础场景的图层，
  在 meta.provenance 里列出；不覆盖 scenes/deepseek-chan/scene.json）。

```bash
# 全片批量（未启动；等确认）
systemd-run --user --unit=a5-full --collect --working-directory=$HOME/CharaForge "--setenv=PATH=$PATH" -p MemoryMax=8G \
  tools/.venv/bin/python -I tools/gpu/scripts/a5_batch.py --units all --batch 4 --out .data/gpu/fit/full
# 只重新合成：--assemble-only；只看分批：--dry
```

### s25 退化的诊断

- 「clip 掉的顶点不进约束、缩放只剩脸部」**不成立**：s25.main 在图像内可见的顶点约 14100 个，其中 11700 个（83%）在 clip 窗口内，
  光流有效点约 6100 个，与 s24（6700）相当。
- 前倾也不是原因：旧引擎 LeanInput = −26，新的 −12；把前倾固定为 −26 重跑（s25_l26），普通帧仍是 0.549。
- 新解比旧解大约 9%、眼距反而略小（头部角度不同），整体相对 shots 锚点偏 20～25px；光流阶段末尾残差约 1.0（s24 约 0.3），
  外观阶段 blur 顶到上限。结论是光流对应的精度在这个镜头上不够（参考帧偏软、窗口内大片浅蓝、clip 边缘），
  停在一个自洽但偏大的解上。加了 S-SSIM 精修之后（s25s）普通帧 0.549 → **0.774**（146 / 162 ≥ 0.6），旧引擎 0.647 / 119。
- s26 2530 的 key5 / key7 / key12：B3 用 runtime 核对有明显效果，本引擎的前向与导出也确实带上了它们（2540 之后渲染出 >< 眼）。
  之前报「无差别」是估计的问题：逐个参数单独移动时，key5 / key7 的部件被 key12 = 1 藏着，单独移动看不出差别；
  而且 ±6 帧窗口里受影响的帧在 2530 切点前后，大部分落在白色笔刷的排除区里。分组之后得分极差仍只有 0.0004，按规则保留标注值。

### 结果（runtime 渲染 + metric.py + flicker.py；冷启动；shots.json 738d6d0）

| 单元（运行） | 普通帧 新 ← 旧 | 目测帧 新 ← 旧 | 特效帧 新 ← 旧 | 闪烁 新 / 旧 | 耗时 |
|---|---|---|---|---|---|
| s24（s24i / s24j） | 50：0.819 / 0.710 / 50 ← 0.765 / 0.711 / 50 | — | 0.627 ← 0.526 | 0 / 0 | 200 s |
| s25～s28（s25i） | 162：**0.746** / 150 ≥ 0.6 ← 0.647 / 119 | 25：0.410 ← 0.323 | 0.351 ← 0.365 | 1 / 1 | 523 s |
| s25～s28（s25s，A8 之前） | 162：**0.774** / 146 ← 0.647 / 119 | 25：0.388 | 0.364 | 0 / 1 | 1251 s |
| s21（s21h，A8 之前，含手臂参数） | — | 215：**0.757** / 204 ≥ 0.6 ← 0.464 / 29 | 0.188 ← 0.114 | 1 / 33 | 约 1300 s |
| s12～s14（s12h） | 173：**0.671** / 130 ← 0.502 / 29 | — | 0.521 ← 0.371 | 5 / 8 | 423 s |
| s15（s15i，圆形开窗 + 翻牌 + 景深） | — | 152：0.495 / 41 ← 0.450 / 1 | 0.692 / 135 ← 0.676 / 127 | 3 / 17 | 1107 s |
| s38+s39（s38i） | — | 77：0.488 / 11 ← 0.343 / 0 | 0.387 ← 0.305 | 0 / 3 | 673 s |
| s33（s33i） | — | — | 48：0.766 | 0 | 166 s |
| s34 / s35 / s36（单独） | — | — | 0.262 / 0.485 / 0.545 | 1 / 0 / 1 | 123 / 161 / 181 s |

- **s38.side 的 Q 版切换**：粗配准与小实例先验之后，估计为 3346 / fade 6（标注 3346 / fade 8）。fade 与期望（约 6）一致，
  起点比期望（3348～3350）早 2～4 帧。得分曲面很平：3346/6 0.5997、3347/6 0.6006、3348/6 0.6021。
  这个实例的脸仍不对（ParamAngleY 约 +20，脸在头发里偏下），起点的分辨力受它限制，见剩余问题。
- **可复现**：接入 A8 与 S-SSIM 精修后，s24 两次（s24i / s24j）不再逐位相同（第一次分叉在 F3 外层 6），
  runtime 普通帧均值 0.8186 / 0.8184（差 0.0002）、特效帧 0.627 / 0.630（差 0.004），仍在 0.005 以内。
- **批处理**：s33 / s34 / s35 / s36 一批（batch_trial2）墙钟 483 s，同样四个单元单独跑合计 631 s，快约 24%；
  四个单元的特效帧均值 0.500（194 帧），单独跑 0.517。

### 耗时与全片估算（A7 + A8 之后）

- 分项（s24i，200 s）：装载 20 s、前倾多假设 33 s、F1 19 s、F2 33 s、F3 23 s、S-SSIM 26 s、外观 46 s。
  A8 之后物理不再是瓶颈（s24 内层从约 130 s 降到约 40 s），现在大头是外层渲染 + 光流、SSIM 精修和外观（都随帧数增长）。
- 单卡串行：单图层约 2.5～3.5 s / 帧，多图层长单元约 2.5～3.5 s / 帧（s25 单元 2.7、s15 3.4、s38 7.0（含开关估计）、s12 1.9）。
  全片约 3600 帧，按 3 s / 帧约 **3 小时**；4 单元一批快约 24%，约 **2.3 小时**。
- 离 1 小时还差约 2.3 倍。批处理的收益有限，因为现在的大头随帧数线性增长，而不是固定的内核发射开销。后续方向：
  1. 外观阶段（每个单元 40～110 s，确定性卷积反向很慢）：减少帧与迭代，或不开确定性（只影响约 1e-4 的常量）；
  2. 有了 S-SSIM 精修之后，F3 可以去掉、F2 缩短（F2 + F3 约占四成）；
  3. 前倾试拟合只在头高 ≥ 300px 的单元做（planAuto 已限定），试拟合的外层次数减半；
  4. DIS 光流（CPU）与 GPU 渲染流水线并行。

### 剩余问题

- **s38.side 的脸**：小实例在粗配准之后位置、尺度对上了，但面捕输入仍漂到 ParamAngleY 约 +20（脸在头发里偏下）。
  光流和 SSIM 在眼距 30px 的实例上对头部角度几乎没有约束，现在的先验还不够强。可以考虑把小实例的面捕输入限制到只有头部角度，
  或者用实例的 head 框中心约束头部在画面上的位置。
- **s34**（特效帧 0.262）、**s21 / s41 的特效帧**仍低：大多是特效遮挡（不是几何问题），按规则以目测为准。
- s21 的最终数字来自 A8 之前的代码（s21h）；其余单元是最终代码。
