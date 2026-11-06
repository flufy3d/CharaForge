# GPU 整段轨迹联合优化（A5）

写于 2026-11-06。用 A1 变形器、A2 物理和 A3 光栅化拼成一条全在 GPU 上、可求导的 Live2D 前向链路，对一个镜头的整段轨迹做联合优化，
结果导出成 scene.json 图层。**最终指标只认 runtime**：导出后用 `pnpm render` 渲染，再用 `tools/qa/metric.py` 打分。

## 文件

| 文件 | 内容 |
|---|---|
| `fit/rig.py` | 模型装配：moc3（A1）、physics3（A2，float64 + `torch.compile`）、贴图与 drawable 静态数据（A3，直接从 moc3 生成，与 dump 逐项一致）；虹膜中心标定 |
| `fit/vts.py` | VTS 映射与平滑的 torch 版本（语义同 `tools/fit/src/vts.ts`，平滑写成下三角矩阵，可求导） |
| `fit/plan.py` | 镜头规划：只读 shots.json（图层区间、新画面节点、frameHold、变换关键帧、锚点、帧类别）；参考帧按 metric.py 的口径解码 |
| `fit/traj.py` | 参数化与可微分前向：面捕输入节点 → VTS → prePhysics → 物理（截断 BPTT）→ postPhysics 偏移 → 变形 → 光栅化 → 滤镜 |
| `fit/loss.py` | SSIM（与 metric.py 逐项一致）、metric 口径的遮罩与合成、光流项（DIS）、粗尺度光度项、边缘项 |
| `fit/optimize.py` | 分阶段优化：锚点初值、前倾多假设、光流对应阶段（外层渲染、内层只跑物理和变形）、外观常量（全分辨率，缓存图层缓冲） |
| `fit/export.py` | 导出 scene.json 图层，格式同 `traj2.ts` 的 `buildLayer` |
| `fit/tracks.py` | 场景图层求值的 numpy 参考实现（规范 a～g、曲线、frameHold、draw 参数），用来核对导出语义 |
| `scripts/a5_fit.py` | 命令行：拟合一个镜头，写出 `layer.json`、`fit.json`、`fit.log` 和 `preview/` |
| `scripts/a5_merge.ts` | 合并进临时全片场景，`parseScene` 校验通过才写出；不覆盖基础场景 |
| `scripts/a5_check.py` | 导出一致性对拍：GPU 前向按场景图层渲染 1280×720，与 runtime 的同一帧比较 |
| `scripts/a5_vts_ref.ts`、`tests/test_a5_vts.py` | VTS 的 torch 版与 TS 版对拍 |
| `tests/test_a5_loss.py` | SSIM 与 metric.py 对拍 |

## 命令（s24）

```bash
cd ~/CharaForge && export PATH=~/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
# 拟合（约 9 分钟；长任务用 systemd-run，进程内存 ≤ 8G）
systemd-run --user --unit=a5-fit --collect --working-directory=$HOME/CharaForge "--setenv=PATH=$PATH" \
  --setenv=CUDA_HOME=/usr/local/cuda-13.0 -p MemoryMax=8G \
  tools/.venv/bin/python -I tools/gpu/scripts/a5_fit.py --shot s24          # → .data/gpu/fit/s24/{layer.json,fit.json,fit.log,preview/}
#   --quick（迭代 ÷4，调试）  --decouple 0|1（缺省 1）  --smooth-c 0.15  --mem-gb 12
#   --synth <scene.json>：诊断用合成基准，把参考帧换成该场景本镜头图层经 GPU 前向的渲染
# 合并成临时全片场景（只替换 s24.* 图层）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_merge.ts \
  --base scenes/deepseek-chan/scene.json --shot s24 --layer .data/gpu/fit/s24/layer.json --out .data/gpu/fit/s24/scene.json
# runtime 渲染与验收
pnpm render --scene .data/gpu/fit/s24/scene.json --frames 2368-2441 --out .data/gpu/fit/s24/render
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/gpu/fit/s24/render --shots s24 --out .data/gpu/fit/s24/metric.json
tools/.venv/bin/python -I tools/qa/flicker.py .data/gpu/fit/s24/render 2368-2441
tools/.venv/bin/python -I tools/qa/report.py --shots s24 --renders .data/gpu/fit/s24/render --out .data/gpu/fit/s24/report
# 导出一致性（GPU 前向 vs runtime）
tools/.venv/bin/python -I tools/gpu/scripts/a5_check.py --scene .data/gpu/fit/s24/scene.json --layer s24.main \
  --renders .data/gpu/fit/s24/render --frames 2370,2391,2400,2413,2425,2441 --json .data/gpu/fit/s24/check.json
# 单元测试
tools/.venv/bin/python -I tools/gpu/tests/test_a5_vts.py && tools/.venv/bin/python -I tools/gpu/tests/test_a5_loss.py
```

## 前向链路（与导出的图层逐项同语义）

1. **面捕输入**：每个新画面帧（节点）一组值，节点之间线性插值，区间外保持端点（同 `traj2.inputSeries`）。
   拟合的输入是 FaceAngleX/Y/Z、EyeOpenLeft/Right、EyeRightX/Y、MouthSmile、MouthOpen、Brows，以及 decouple 时的 BodyX/Y/Z；
   其余输入取中性值。
2. **VTS**（`vts.py`）：区间线性映射，不夹紧；指数平滑 α = 1/(1 + c·smoothing)，c = 0.15；自动呼吸。
   与 TS 版对拍的最大相对误差为 1e-15（`test_a5_vts.py`：两种 decouple、三个 c、部分输入缺省）。
3. **prePhysics**：写入顺序同导出的参数轨（variant 常量 → LeanInput → Clothes2 → 开关斜坡 → VTS 输出）。
   set 语义：夹到 [min, max]，存 float32。
4. **物理**（A2）：dt = 1/60；预热 90 帧，输入取帧 in，不做 stabilize。物理用 float64，并开 `rig.compile()`。
   按 A2 的建议截断梯度：预热帧放在 `no_grad` 里跑完再 detach；正式段每 24 帧 detach 一次状态（截断 BPTT）。
   这样只影响梯度，前向与 runtime 逐帧相同。姿态链只滞后 1 帧、低通 5～15 帧，24 帧的窗口够用；
   后发、缎带等混沌的次级物理，不靠穿过物理的长程梯度，改由下面的 postPhysics 偏移直接控制。
5. **postPhysics add 偏移**：姿态输出 6 个，加上物理写入的全部参数（本模型 292 个），每个参数一个自由度，每个节点一个值，节点间线性插值。
   对应 fit4 的「逐帧精修」，不同的是这里带二阶差分正则联合求解，而不是逐帧独立求解再事后平滑。
6. **变换**：稀疏关键帧（zoom、pan、scale 镜头每 12 帧一个，否则每 24 帧），关键帧之间线性插值，止于最后一个节点；
   参数为 x、y、ln scale、rotation，anchor 固定为参考姿态下两眼虹膜中点。导出时逐帧烘焙，值与 GPU 前向逐帧相同。
7. **frameHold**：先做参考帧重复检测（移植 `shotsutil.ts detectNewFrames`，判据相同），再用 shots.json 的 cadence 覆盖；
   frameHold 段由节点间隔生成。只渲染节点帧 g，显示帧 f 与 g = hold(f) 用同一张渲染，分别与参考帧 f 比较。
8. **变形 → 光栅化 → 滤镜**：A1 → A3（`quantize=False`），滤镜顺序为 color → resample → blur。
   在 640×360 上拟合时，to_screen 与 blur σ 乘 0.5；resample 因子相对于合成分辨率，因此等效因子取 f/0.5（≥ 1 时不处理）。

## 目标函数

- **SSIM@360**：与 metric.py 同一口径——合成到参考帧上，遮罩取 dilate(α > 0.1) 减去排除区，RGB 三通道，高斯 σ = 1.5，窗截断 3.5σ，
  边界用 scipy 的 reflect。`test_a5_loss.py`：随机图上逐像素误差 < 1e-9；用 runtime 渲染的 s24 帧走 metric.py 的完整流程，
  SSIM@360 差 5.6e-16。最终阶段（外观常量）就在 1280×720 上渲染，按 2×2 区域平均（等于 INTER_AREA）降到 360 后计算，与验收完全同口径。
- **光流项（选它的理由）**：实测 SSIM@360 的吸引域只有约 ±4px。在 fit4 解附近横移，±4px 时 SSIM 从 0.72 降到 0.51，±16px 时为 0.27。
  冷启动时姿态、前倾、身体的偏差有几十像素，直接对 SSIM 做梯度下降几乎不动：试过 Adam 直接优化 SSIM 加模糊 L1，
  45 次迭代后 SSIM@640 只有 0.29。光流的捕获范围有几十像素，而且对外观差异不敏感。做法是：
  渲染（不求导）→ 合成图到参考帧的 OpenCV DIS 光流（MEDIUM 预设，前后向一致性 < 1.5px）→ 每个可见顶点的目标 = 当前位置 + flow；
  损失为 Cauchy(|p − 目标| / c)，c 从 12px 退火到 5px、3px（以 720p 像素计）。
  目标固定之后，内层优化**只需要 VTS、物理和变形，不需要光栅化**（光栅化是最贵的一步）：外层每渲染一次，内层走 8 步 Adam。
  同样的时间里，这比「渲染 + 反向」逐步优化 SSIM 有效得多：合成基准上 160 秒内 SSIM 从 0.19 升到 0.55，而直接优化 SSIM 只到 0.29。
- **边缘项、粗尺度光度项**（`edge_loss`、`blur_l1`）已实现。粗尺度光度项只用于前倾的多假设比较（σ = 16、8px 的模糊 L1，在 ±16px 内单调）。
  边缘项在 σ = 1 时同样没有长程吸引力，最终流程里没有用到。
- **锚点先验**：shots 里置信度 ≥ 0.4 的 eyes，对应两眼虹膜中心。虹膜中心取虹膜网格顶点质心加一个标定偏移：
  参考姿态下渲染两眼，取网格附近青色像素的质心。网格含有被眼白 clip 和上眼睑遮住的部分，顶点质心比可见虹膜偏上约 0.05 眼距。
  损失用 Cauchy 核，尺度 max(10px, 0.15 × 眼距)，权重随阶段从 0.05 降到 0.02。
  s24 的 2378 帧锚点（置信度 0.5、由光流传播得到）偏了约 80px，被鲁棒核压掉。
- **时间正则直接写进目标**：面捕输入、变换关键帧、postPhysics 偏移都用非均匀节点上的二阶差分（斜率变化 × 平均间距），
  分别除以各自的尺度：输入为 INPUT_DIMS 的 tau；变换为 60px、60px、0.12、6°；偏移为 0.05 × 半范围。
  另有偏移幅度先验（0.15 × 半范围），以及变换速度上限的平方 hinge（每帧 30px、ln s 0.02、0.8°）。不做事后平滑。

## 优化（分阶段，由粗到细；Adam）

| 阶段 | 分辨率 | 内容 | s24 耗时 |
|---|---|---|---|
| 装载 | — | moc3、贴图（0.85 GB 显存）、参考帧解码、物理编译（FX 缓存热时约 4 s，冷启动见下） | 22 s |
| 初值 | — | 变换：shots 眼睛 → 相似变换（两点精确），按当前姿态的实际虹膜位置重算（`reanchor`），关键帧上做加权最小二乘 | <1 s |
| 前倾多假设 | 320×180 | LeanInput ∈ {0, −10, −20, −25, −30}，每个值都先 reanchor，再用粗尺度光度项比较，取最好的作初值，之后当连续量优化 | 11 s |
| F1 光流粗 | 320×180 | 外层 10 × 内层 8；变换 + 面捕输入 + 前倾；c = 12px | 123 s |
| F2 光流细 | 640×360 | 外层 14 × 内层 8；加 postPhysics 偏移；c = 5px | 167 s |
| F3 光流精 | 640×360 | 外层 10 × 内层 8；c = 3px | 121 s |
| A 外观常量 | 1280×720 | 15 个普通帧的图层缓冲渲染一次后缓存，只对滤镜求导（blur σ、颜色）；resample ∈ {1, .8, .7, .6, .5} 各优化 40 步，取 SSIM@360 最好的 | 76 s |

为什么用 Adam、不用 L-BFGS：数据项里有遮罩切换、夹取、8 位取整的直通估计，还有光流目标的重估，都不光滑；
各组未知量的量纲差别很大（像素、角度、0..1）。这里把未知量归一化到各自的单位，每组一个学习率，余弦退火。
L-BFGS 的线搜索在这种噪声目标上不稳，而且每次线搜索都要重新渲染。
试过在最后加一段直接优化 SSIM@360 的 Adam（640×360，每步要光栅化反向，约 12 s）。
学习率大时 SSIM 下降（0.70 → 0.62）；学习率小时 8 步也没有提升（0.68 → 0.67）。所以这一段没有放进缺省流程。

## s24 结果（2026-11-06，冷启动）

冷启动：初值只来自 shots.json（眼睛锚点、bbox、cadence、效果）和镜头规划，没有读现有 scene.json 的 s24 结果。
前倾、外观常量都从通用缺省值出发（lean 0、blur 1、resample 1、颜色恒等）。

| 指标（runtime 渲染 + metric.py） | A5（GPU） | fit4（现有） |
|---|---|---|
| 普通帧 有效 / ≥ 0.6 | 50 / 50 | 50 / 50 |
| 普通帧 SSIM@360 均值 / 最低（帧） | **0.793** / 0.707（2397） | 0.765 / 0.711（2396） |
| 普通帧 SSIM@720 均值 | 0.779 | — |
| 特效帧 有效 / 均值 / 最低 / ≥ 0.6 | 24 / 0.610 / 0.373（2369） / 14 | 24 / 0.526 / 0.344 / 8 |
| 闪烁尖峰（flicker.py） | **0** | — |
| 拟合墙钟 | 519 s（GPU 同时被 A2 占用约 30%） | 1290 s |
| 显存峰值（PyTorch 分配器） | 8.78 GB（外观阶段缓存 1280×720 图层缓冲；光流阶段约 4.3 GB） | — |

- 导出一致性（`a5_check.py`，6 帧，含 frameHold 保持帧）：GPU 前向 vs runtime 的 SSIM@360 最小 0.99951、均值 0.99956；
  预乘 RGBA8 平均误差 0.13～0.15/255，最大 4。两边各自对参考帧的 metric 分数相差 < 1e-4。导出没有引入语义偏差。
- 拟合结果：LeanInput −24.7、blur σ 1.34、resample 0.8、饱和度 0.69（参考视频的角色偏灰）。呼吸相位和周期没有优化（光流阶段不含），保持缺省值。
- frameHold：检测加 cadence 的结果是整段每 2 帧一个新画面（37 个节点）。只看 shots 的 cadence 时，2390～2391、2412～2425 被当成逐帧新画面，
  在参考帧的重复帧上产生了 2 个闪烁尖峰（2391、2413）；加入重复检测后为 0。
- 可重复性：用同一配置在 `~/cf-scratch/a5/rep2` 又跑了一次（GPU 的原子加法与 DIS 多线程不是逐位确定的）。
  结果为普通帧 50/50、均值 0.815、最低 0.733（2396），特效帧均值 0.615（14/24 ≥ 0.6），闪烁尖峰 0，耗时 517 s。
  调参过程中，不同配置之间的普通帧均值在 0.735～0.815 之间波动，主要来自前倾的局部极小：前倾多假设在 −20 与 −30 之间几乎打平，
  落在 −21 的那次只有 0.735。现在多假设加了 −25 一档。
- 对照视频：`.data/gpu/fit/s24/report/diff_sbs.mp4`（原视频 | 纯渲染 | 差异），抽帧拼图在 `.data/gpu/fit/s24/shots/sbs_tile.jpg`。
  普通帧没有重影或跳变。特效段开头几帧（2368～2372）嘴型不对（渲染张嘴，原视频闭嘴）：这几帧被白色划痕粒子与排除区盖住，光流约束少。

### 物理编译时间

`rig.compile()` 第一次调用会编译前向与反向。FX/inductor 缓存热的时候约 4 s；本机缓存全冷时约 100～160 s（另计，不在上表里）。
换机器或升级 torch 之后，第一次拟合会多出这一段。

## 推广到全片还需要做的

- **多实例**：`plan.make_plan` 现在只接受单实例镜头。前向与优化是按图层写的（`Traj`），要做的是把多个 `Traj` 的图层缓冲按 z 序合成，
  再联合求光流目标（光流在合成图上，顶点按图层归属）。A3 的 `composite` 已经支持多图层。
- **clip / opacity 轨**：A3 有可导的圆角矩形羽化 clip。规划要从 shots 读入 clip 关键帧与 opacity（plans.ts planAuto 的规则），
  并把它们写进 draw 与导出（`traj2.buildLayer` 的 clip 段）。目前遇到 clip 会报错。
- **景深 / 模糊段**：blur σ 改成逐帧（dofK × 效果强度曲线），外观阶段对 dofK 求导。目前遇到 blurSigma ≥ 1.5 会报错。
- **开关切换**：variant 常量与 switchEvents 斜坡已经进前向和导出（与 traj2 同语义），但没有在带切换的镜头上验证；
  切换帧的估计（fit4 有画面检测）还没做。
- **cardFlip / 翻牌 scaleX**：没做。
- **速度**：光流阶段每次外层约 12 s，其中渲染约 5 s，内层 8 步约 7 s（物理加变形，B = 37）。
  全片约 3600 帧，按 s24 每 74 帧约 9 分钟估算，单卡串行约 7 小时；要达到「全片 1 小时」，还需要
  多镜头批处理（物理与变形对 B 是批量的）、减少外层次数，或者给 A3 写融合核。
