# 可微分 Cubism 物理（A2）

`tools/gpu/l2d/physics.py`：把 Cubism SDK for Web Framework R5 的 `CubismPhysics`
（`vendor/cubism/framework/src/physics/cubismphysics.ts`，Live2D Open Software License）逐行移植成 PyTorch，
批量 `[B, P]`、可整段反向传播（BPTT），float64 下与 runtime 对拍逐帧误差 ≤ 3e-8（相对参数范围）。

## 1. 接口

```python
import sys; sys.path.insert(0, 'tools/gpu')
from l2d.physics import PhysicsRig

rig = PhysicsRig.from_files('assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json',
                            '.data/gpu/a2/ref/model_params.json',      # {ids, min, max}：Core 的参数表
                            device='cuda', dtype=torch.float64)       # emulate_f32=True（缺省）
rig.compile()                         # 可选：torch.compile 单帧函数（首次约 30～60 s 编译，之后快 10～20 倍）
st = rig.initial_state(B)             # = CubismPhysics.create() 之后（runtime 的 resetPhysics()）
st, out = rig.stabilize(st, params)   # layer.physics.stabilize = true 时（Cubism stabilization）
st, out = rig.step(st, params, dt=1/60)   # 规范 e：params [B, P] 物理之前的参数 → out [B, P] 物理之后
st, outs = rig.rollout(params_seq)    # [T, B, P] 逐帧推进；checkpoint=True 逐帧激活检查点
# A8 的两条可选快速路径（与 rollout 同语义，见 §9）：
st, outs = rig.fused_rollout(params_seq, st)     # Triton 融合核：整段前向一个核、反向一个核（推荐）
st, outs = rig.graphed_rollout(params_seq, st)   # CUDA graph 重放整段（compile 版的逐帧函数）
rig.auto_fused = True                 # 之后 rollout() 自动走 fused_rollout（auto_graph 同理）
```

- `params` 是**全部**模型参数（P = 509，顺序即 Core 参数索引），也就是规范 a～d 之后、e 之前的值；
  `out` 是 e 之后的值（只有物理输出参数被改写，其余原样返回）。
- `rig.input_params` / `rig.output_params`：物理读 / 写的参数索引（本模型 50 个 / 292 个）；`rig.output_param_index` 是对应的 LongTensor。
- `PhysicsState` 不可变（每步返回新对象），字段：粒子位置 / 速度、每条链的 lastGravity、
  `_currentRigOutputs` / `_previousRigOutputs`、`_parameterInputCaches`、`_currentRemainTime`；`st.detach()` 截断梯度。
- 输入可以是 float32 张量而 rig 是 float64（`out` 回到输入的 dtype，梯度照常穿过类型转换）。
- 规范第 4 条的预热：用 `rollout(params_in_frame_in.expand(warmup, B, P))` 推进 warmup 帧，再接正式帧；
  `layer.physics.stabilize` 时先 `stabilize`（输入同样取帧 in）。与 `LayerEvaluator` 的状态链完全一致（s24 等场景图层对拍已覆盖）。

## 2. 与 Framework 的逐段对应

| physics.py | cubismphysics.ts（R5） | 说明 |
|---|---|---|
| `_build` / `_build_level` | `parse()`（76）、`initialize()`（804） | 读 json；初始位置 = 上一粒子 + (0, radius)，lastGravity = (0, 1)；`translationScale` 不从 json 读（恒 0，见 §3） |
| `initial_state` | `create()` 后的状态 | 不做 stabilization（规范第 4 条） |
| `step` 的时间调度 | `evaluate()`（485）：`_currentRemainTime += dt`（509）、> 5 s 清零、`physicsDeltaTime = 1/Fps`、`while (remain >= physicsDeltaTime)`、`inputWeight = physicsDeltaTime / remain`（552）、`alpha = remain / physicsDeltaTime`（691） | 用 Python double 计算，整批共用 |
| `_frame` 子步开头 | `_previousRigOutputs ← _currentRigOutputs`；`_parameterCaches = lerp(_parameterInputCaches, values, inputWeight)` 并存回 InputCaches | `_parameterInputCaches` 为 null 时用本帧参数初始化（513 行起） |
| `_inputs` | `normalizeParameterValue`（1277）+ `getInput*FromNormalizedParameterValue`（909 起） | 夹到参数范围；中点 = 范围中点（不是参数缺省值）；正 / 负两段线性；`sign(NaN) = 0`；`isInverted ? r : −r`；× weight/100；按输入顺序累加 |
| `_inputs` 末尾 | `radAngle = degreesToRadian(−angle)`（603）后的旋转 | 原样保留：y 分量用的是**已旋转**的 x |
| `_update_particles` | `updateParticles`（1086） | 重力方向 (sin, cos) 再 normalize；force = g·acc + wind；delay = delay·dt·30；空气阻力：`directionToRadian(lastGravity, g)/5` 旋转（同样用已旋转的 x）；+ v·delay + f·delay·delay；半径约束；`|x| < 0.001·Normalization.Position.Maximum` 吸附为 0；delay ≠ 0 时 v = (p − lastP)/delay·mobility |
| `_stab_particles` | `updateParticlesForStabilization`（1182） | 位置 = 上一粒子 + normalize(force)·radius，速度清零 |
| `_outputs` | `getOutputTranslationX/Y`、`getOutputAngle`（1015） | Angle 的父方向：顶点 ≥ 2 取上一段，否则 −options.gravity；`directionToRadian` 折回 [−π, π]；Reflect 取反 |
| `_apply_rounds` | `updateOutputParameterValue`（1238）及「値を反映」 | × scale → 夹到目标参数 [min, max] → weight < 1 时与目标当前值混合 → 写入 Float32Array |
| `stages` 循环 | setting 顺序执行，`_parameterCaches` 在 setting 间传递 | 见 §4「并行化」 |
| `_interpolate` | `interpolate()`（701） | prev·(1−α) + cur·α，写进模型参数（同样走 updateOutputParameterValue） |
| `stabilize` | `stabilization()`（281） | 读模型参数（含前面 setting 已写入的输出），cur = prev = 输出值，InputCaches = 调用时参数 |

runtime 的用法（`packages/runtime/src/instance.ts`、`evaluator.ts`、`physicsState.ts`）：没有改 Framework；
每帧 `physics.evaluate(model, 1/fps)`；初始状态 = `create()` 后的快照；`physics.stabilize` 时在预热前调用 `stabilization()`。
Options 是 Framework 缺省值（gravity (0, −1)、wind (0, 0)），json 里的 `EffectiveForces` 不参与计算。

## 3. 需要特殊处理、或与「直觉」不同的地方

1. **Float32 存储**：`_parameterCaches`、`_parameterInputCaches`、模型参数都是 `Float32Array`，粒子和 rig 输出是 double。
   float64 模式（`emulate_f32=True`，缺省）在同样的位置做 float32 舍入，反向直通（梯度按未舍入的函数）。
   不做这个舍入时误差约 1e-7 量级，在混沌的链上会被放大（见 §5），所以缺省打开。
2. **X / Y 类型输出恒为 0**：`getOutputScaleTranslationX/Y` 返回 `output.translationScale`，而 Framework 从未给它赋值（构造为 (0, 0)）。
   本模型 292 个输出全是 Angle 类型，不受影响；照原样移植。
3. **目标参数索引的懒解析**：`destinationParameterIndex` 在第一次 `evaluate` 时才解析，挂在 rig 上、不在 runtime 快照里，
   `resetPhysics()` 不会恢复成未解析。所以「全新实例的第一帧且该帧没有子步（dt < 1/Fps）」时 `interpolate` 不写参数，
   而复用过的实例会写。`initial_state(resolved=True)`（缺省）对应复用过的实例；dt = 1/60、Fps = 60 时两者没有区别。
   （对拍里 `walk_dt144` 第 0 帧就是这种情况，参考数据来自复用过的实例。）
4. **Fps = 60、dt = 1/60 时**：每帧恰好一个子步，inputWeight = 1，α = 0，于是写回参数的是 `_previousRigOutputs`，
   即**物理输出比输入晚一帧**。这是 Framework R5 的行为，照原样保留。
5. **模型里不存在的参数**（本模型没有）：输入按 Framework 读到 undefined 的结果处理（贡献 = 归一化缺省值 × 符号 × 权重），输出不写。
6. **NaN 与梯度**：`atan2` 用自定义 autograd，(0, 0) 处梯度为 0；`normalize` 长度为 0 时前向给 NaN（与 JS 的 0/0 相同），
   反向用 `where` 屏蔽、不会产生 NaN；`delay = 0` 的粒子不更新速度（常量分支）。实际序列里这些退化情况不出现。
7. **不可导点**：吸附阈值（`|x| < 0.01` 置 0）、`directionToRadian` 的 ±π 折回、输入 / 输出的夹取都是分段的；
   autograd 给的是所在分支的导数（夹取段为 0，跨越吸附 / 折回的跳变不体现在梯度里）。

## 4. 并行化

- setting 之间的依赖只通过参数缓存：按「写后读」（严格在后）、「写后写」「读后写」（不早于）给 131 个 setting 分层，
  本模型是 **4 层**；每层所有 setting 先从同一份缓存读输入，全部算完再按原顺序写回（写同一参数的多个输出分轮），与逐个顺序执行等价。
- 每层一组（缺省）：链长填充到该层最长（4 层分别为 64×7、60×11、5×7、2×4，共 1151 个粒子槽，实际 431 个）；
  填充粒子是一条合法的摆（半径 1），不会产生 NaN，也不影响真实粒子（链只向下依赖）。粒子按链下标逐个推进，层内所有链同时算。
- `split_by_length=True`：每层再按链长分组（16 组，无填充）。B=64 显存 1243 → 812 MiB，但 kernel 多，compile 后慢 2～3 倍。
- `rig.compile()` 用 `torch.compile` 编译单帧函数，前向 / 反向都融合成少量 kernel（本问题受 kernel 启动数限制，提速 10～40 倍）。

## 5. 对拍

参考数据（runtime 同一份代码：`LayerEvaluator` / `ModelInstance` + Framework 的 `CubismPhysics`，包装 `stepPhysics` 记录前后参数）：

```
node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types tools/gpu/scripts/a2_ref_dump.ts
tools/.venv/bin/python -I tools/gpu/scripts/a2_compare.py [--dtype f64,f32io,f32] [--compile]
```

序列（全部从初始状态开始，含预热）：
- 场景图层 `s24.main`（帧 2278–2441，含 90 帧预热）、`s15.left`（412 帧）、`s30.full`（380 帧）、`s06.main`（359 帧）；
- 合成：`step` 阶跃 600、`sweep` 0.1→8 Hz 扫频 720、`walk` 随机游走 900、`stab_walk`（先 stabilization）601；
  另外检查固定步长内插：`walk_dt50`（dt = 1/50）、`walk_dt144`（dt = 1/144）、`jitter`（每帧随机 dt ∈ [1/240, 1/20]）各 600。

误差 = |ours − ref| / (参数 max − min)，所有 292 个物理输出参数、所有帧取最大；「四段」为按时间四等分的最大值（检查漂移）。

结果（RTX 5080，2026-11-06；`f64` = rig float64 + float32 舍入模拟；`f32io` = 输入输出 float32、rig float64；`f32` = 全 float32）：

| 序列 | 帧数 | f64 最大误差 | 所在帧 / 参数 | f32io | f32 最大误差 | 所在帧 / 参数 |
|---|---|---|---|---|---|---|
| s24_main | 164 | 0 | — | 0 | 4.7e-5 | 2317 ShineX |
| s15_left | 412 | 4.4e-16 | 1384 Param35 | 同 f64 | **3.4e-3** | 1229 RBowX4 |
| s30_full | 380 | 9.3e-10 | 2966 RWrist | 同 f64 | **4.7e-3** | 2797 LateSideHairY2 |
| s06_main | 359 | 3.6e-15 | 389 LipPhysics1 | 同 f64 | 4.9e-5 | 388 ShineX |
| step | 600 | 4.7e-10 | 59 Hat2phB | 同 f64 | **4.8e-3** | 183 RRibbonX4 |
| sweep | 720 | **3.0e-8**（全表最大） | 700 ShortPonytailYBS | 同 f64 | **1.0**（翻转） | 649 ribbonY6 |
| walk | 900 | 0 | — | 同 f64 | 1.1e-4 | 844 LJewelry3 |
| stab_walk | 601 | 3.6e-15 | 359 LateSideHairX2 | 同 f64 | **3.3e-3** | 291 RJewelry3 |
| walk_dt50 | 600 | 0 | — | 同 f64 | 7.4e-5 | 501 ShineX |
| walk_dt144 | 600 | 0 | — | 同 f64 | 4.3e-4 | 275 REyeLashX2 |
| jitter | 600 | 0 | — | 同 f64 | 1.1e-4 | 85 SideHairRBS |

- **float64：全部通过（门槛 1e-5），最大 3.0e-8**；四段分段最大值没有随时间增长的趋势（无漂移）；
  rig 输出（弧度）最大偏差 5.3e-8；非物理参数逐位原样。`--compile` 版本同样全部通过（最大 1.5e-7，融合乘加带来的舍入差）。
- **f32io：全部通过**，结果与 f64 相同（输入本来就是 float32 值）。
- **纯 float32：5 个序列未达 1e-3（未降低门槛）**。原因不是实现，而是动力学本身会放大 1e-7 级的扰动：
  在 float64 下给输入加 1e-7×范围 的随机扰动，输出偏离 s15_left 3.4e-3、stab_walk 1.3e-3、step 1.8e-4、sweep 1.0（翻转）
  ——与 float32 的误差同量级。放大来自两类不连续/混沌：MovementThreshold 吸附（|x| < 0.01 置 0，扰动决定是否吸附；
  关掉吸附后 s15_left 降到 3.7e-5），以及多粒子链（多重摆）的混沌与 ±π 折回（关掉吸附后 stab_walk 仍有 4.9e-3、sweep 仍翻转）。
  float32 的舍入本身就是 1e-7 级，任何 float32 写法都无法避免。**需要 float32 接口时用 f32io**（GPU 上与 float32 同速）。

## 6. 梯度

```
tools/.venv/bin/python -I tools/gpu/tests/test_a2_grad.py
```

float64、关闭 float32 舍入模拟（否则差分被量化淹没；autograd 对舍入是直通），起点 s30_full 第 120 帧（场景帧 2720），窗口 60 帧，
loss = 随机权重 × 物理输出参数之和；h = 1e-6 × 参数范围，另用 h/4 检查差分本身是否稳定。门槛 1e-4。

| 检查 | 结果 |
|---|---|
| 单点：AngleX、AngleZ、PositionY、ParamBreath、LeanInput、ParamEyeLOpen、ParamBrowLY，各在窗口第 0 / 15 / 29 帧扰动，loss 只统计扰动 30 帧之后（BPTT 30～59 帧） | 21/21 通过，最大相对误差 3.8e-6 |
| 方向导数（窗口内全部帧、全部不在端点上的输入，中心差分） | 2/3 通过（6.8e-8、7.9e-8）；1 个跨过吸附跳变（h 与 h/4 的差分相差一个数量级、差分 2.4e4 对 autograd 1.6e3），不计入 |
| 方向导数（只动停在 min/max 上的输入，方向向内，单侧差分） | 2/2 通过（2.1e-7、2.1e-6） |
| stabilize + 60 帧，对第 0 帧输入求导（AngleX、PositionX、ParamEyeROpen） | 3/3 通过，最大 1.8e-9 |

修过的一处：输入恰好等于范围中点时（缺省值，例如 LeanInput = 0），Framework 走 `sign = 0` 的常量分支，autograd 原本给 0；
现在前向不变（`pv·r + m` 在 pv = 0 时逐位等于 m），梯度取左右斜率平均（= 中心差分）。
端点（min / max）上 autograd 给向内的单侧导数，中心差分是两侧平均，所以端点条目单独检查。

梯度测试是在 `split_by_length=True` 时跑的；两种布局逐元素做的是同一组运算（对拍结果逐位相同），梯度不受影响。

## 7. 速度

```
tools/.venv/bin/python -I tools/gpu/scripts/a2_bench.py --mode eager,compile,compile+ckpt --dtype f64
```

300 帧前向 + 整段反向，输入为 s30_full 加小扰动，loss = 全部物理输出的加权和；RTX 5080，与其他 4 个代理共用 GPU（数值有 ±50% 波动），
取 3 次的中位数；显存为该次调用的峰值增量。

| 配置 | B=1 前向 / 反向 / 合计 | 显存 | B=64 前向 / 反向 / 合计 | 显存 |
|---|---|---|---|---|
| compile，f64（**推荐**） | 0.32 / 0.43 / **0.75 s** | 43 MiB | 0.42 / 0.86 / **1.28 s** | 1243 MiB |
| compile，f64，split_by_length | 1.30 / 0.97 / 2.28 s | 111 MiB | 1.11 / 1.06 / 2.18 s | 812 MiB |
| compile + checkpoint，f64，split | 6.4 / 4.7 / 11.1 s | 20 MiB | 5.6 / 4.3 / 9.9 s | 606 MiB |
| compile，f32，split | 0.86 / 1.08 / 1.94 s | 87 MiB | 1.25 / 1.15 / 2.40 s | 444 MiB |
| eager，f64，split | 12.7 / 22.1 / 34.7 s | 157 MiB | 12.9 / 22.1 / 35.0 s | 1249 MiB |
| eager，f64（不分组，较空闲时） | 4.7 / 7.3 / 12.1 s | 61 MiB | 5.6 / 11.0 / 16.6 s | 1839 MiB |

- 首次 `compile` 编译约 30～80 s（每种「每帧子步数」各一次；dt = 1/60 只有一种）；`TORCHINDUCTOR_CACHE_DIR` 可缓存。
- 时间几乎不随 B 变化（kernel 启动受限），f32 与 f64 速度相当；显存随 B 线性增长（主要是每帧保存的激活）。
- 非有限梯度：所有配置都是 0。

## 8. 给优化器（A5）的说明

- 推荐配置：`dtype=torch.float64`（输入输出可以是 float32），`rig.compile()`；GPU 上本问题受 kernel 启动数限制，float64 与 float32 速度相当。
- 需要逐帧把规范 a～d 的参数（[B, P]）交给 `step`，拿回物理之后的参数，再做 f（postPhysics 轨）。
  只关心物理输出时取 `out[:, rig.output_param_index]`。
- 预热：`step` warmup 帧（输入取帧 in），这部分通常不需要梯度：`with torch.no_grad()` 推进后 `st.detach()`。
- 截断 BPTT：任意时刻 `st = st.detach()`。
- 梯度的局限见 §3 第 7 条：吸附阈值与 ±π 折回处的跳变没有梯度；参数被夹在范围端点时梯度为 0。
- `step` 的 `dt` 改变时调度（每帧子步数）会变；compile 版本对每种子步数各编译一次。

> **A8 更新**：拟合器改用 `fused_rollout`（或设 `rig.auto_fused = True`）即可，见 §9.4。

## 9. 提速（A8）：fused_rollout 与 graphed_rollout

两条可选的快速路径，都与 `rollout(params_seq, state, dt)` 同语义（返回 `(最终状态, [T, B, P])`，可反向，可以接在任意状态之后、
任意时刻 `detach`），缺省行为不变。数字见 `perf.md`「物理（A8）」。

### 9.1 `rig.fused_rollout(params_seq, state=None, dt=1/60)`（推荐）

`l2d/_kernels_physics.py`：整段 T 帧的前向是**一个** Triton 核，反向也是**一个**核，每个批元素一个 CTA
（setting 并行、链内逐粒子顺序，层与层之间 `__syncthreads`）。不需要 `compile()`，第一次调用时 Triton 编译约 10～20 秒（有磁盘缓存）。

- **前向与 eager（未 compile）的 `rollout` / `step` 逐位相同**：同样的运算顺序，关闭 FMA 合并，IEEE 除法 / 开方，
  「张量 / Python 标量」按 PyTorch 的做法乘倒数，float32 舍入在同样的位置。sin / cos / atan2 用 **PyTorch 所用 CUDA 版本的 libdevice**
  （`/usr/local/cuda-<torch.version.cuda>/nvvm/libdevice`，找不到时依次试 `CUDA_HOME`、`/usr/local/cuda`）：Triton 自带的 libdevice
  版本不同，atan2 有约 0.3% 的输入差 1 ulp，链是混沌的，会被放大。找不到 CUDA 的 libdevice 时退回 Triton 自带的（`_kernels_physics.LIBDEVICE` 为 None），
  前向不再逐位相同（误差量级同 compile 版）。
- compile 版本身与 eager 差 1e-16 级（FMA 合并，混沌放大后见 §5 的 compile 行），所以 **fused 与 compile 版不逐位相同，与 eager 版逐位相同**。
- 反向是手写伴随：求导时前向顺带把反向要用的中间量记成「磁带」（每帧开始时的状态、每个子步后的粒子状态、各层开始时的 C、各层输出、
  每个 setting 的旋转 / 重力 / 空气阻力中间量、每个粒子的单位向量 / 长度 / 吸附标志），反向核不重算前向。
  公式与 eager autograd 的各个算子一致（atan2 在 (0,0)、normalize 长度 0 处梯度为 0，where / 夹取 / 吸附按所在分支，float32 舍入直通，
  输入恰在中点取两侧斜率平均），只有求和顺序不同：与 eager autograd 的相对差 ≤ 3e-14（200 帧不截断也一样）。
- **确定性**：所有累加按固定顺序（预先建好的逆表），没有原子加，同一输入两次逐位相同（前向与梯度）。
- 对输入参数、以及对传入状态的每个分量（若 `requires_grad`）都求导，所以不截断的 BPTT、跨窗口的梯度都对。
- 支持任意 dt（每帧 0～n 个子步）、`initial_state(resolved=False)`、`in_cache` 为 None 的全新状态、`emulate_f32=False`、float32 输入输出。
  只支持 CUDA、float64 rig、`split_by_length=False`；其他情况（以及 dt ≤ 0、T = 0）直接走普通 `rollout`。
- 显存：求导时快照 + 磁带每帧每个批元素约 150 KB（dt = 1/60），T=200、B=16 峰值约 0.5 GB；不求导时不记。

### 9.2 `rig.graphed_rollout(params_seq, state=None, dt=1/60)`

把整段 T 帧逐帧调用 `_frame`（`compile()` 之后是编译版）的全部内核捕获成一张 CUDA graph，求导时再捕获一张反向图，
每次调用只有一次前向重放、一次反向重放（另有几次拷贝）。**与同一 rig 的 `rollout` 前向、梯度都逐位相同**（compile 与否各自对应）。

- 图按 (T, B, 每帧的子步调度, 哪些输入求导, 是否编译) 缓存。每个图实例的中间量在自己的静态显存池里：同一个键上一次前向的反向还没跑
  又做前向（拟合器把 13 个窗口都前向完再一起反向）时，自动再捕获一个实例；所有键合计最多 `rig.graph_max_instances`（缺省 64）个，
  超出且没有空闲实例时退回普通 `rollout`（结果相同）。`rig.graph_clear()` 释放全部实例；`rig.graph_stats` 是捕获 / 重放 / 退回次数。
- 一次前向只能反向一次（compile 版的反向会就地复用保存的缓冲）。需要 `retain_graph` 多次反向时用 `rollout`；误用会报错，不会给出错误梯度。
- 第一次遇到某个键要预热 + 捕获（compile 版 T=24 约 0.3～1 秒）。未编译的 rig 也能用，但每帧上千个内核，图很大（主机内存大），不推荐。
- 比 `fused_rollout` 慢（GPU 时间还是 compile 版的那些内核），优点是与 compile 版逐位相同。

### 9.3 对原路径的改动（缺省行为不变）

- gather（`x[:, idx]`）换成 `_Index` / `_Gather`：前向就是 `x[:, idx]`，反向按预先建好的逆表固定顺序求和。原来 compile 版的反向
  在这里用 float64 原子加，下标重复时（同一参数被多个 setting 读、同一粒子被多个输出用）求和顺序随调度变，**梯度低位逐次不同**；
  现在 eager / compile 的梯度都逐次相同。前向不变：a2_compare（eager / compile）11 个序列的误差与改动前逐项相同。
- `compile()` 把 dynamo 的重编译上限放宽到 64。原来缺省 8 份：每种 (B, 是否求导, in_cache 是否为 None …) 各编一份，
  用到 3 种 B 就会超，之后**静默退回 eager**（慢约 20 倍，数值也变成 eager 的）。改动前的基准里 B=16 就是这样（`perf.md`）。
- `step` 的时间调度提成 `_advance`（同样的 Python double 运算）。

### 9.4 给拟合器（A5）的接入说明

只需在建好 rig 之后加一行（`fit/rig.py` 或 `a5_fit.py`，A8 没有改 `fit/`）：

```python
rig.phys.auto_fused = True      # Traj.physics / physics_batch 里的 ph.rollout(...) 全部自动走 fused_rollout
# phys_compile 可以去掉：fused 不用 torch.compile（留着也无妨，fused 不走编译版）
```

- 预热（no_grad、90 帧）也走融合核，不记磁带。24 帧一窗、窗间 `st.detach()` 的写法不用改；全部窗口前向完再一起反向也没问题
  （每段的磁带挂在各自的 autograd 节点上，互不覆盖）。
- 前向结果与 eager 版逐位相同，与 compile 版差 1e-16 级再经混沌放大（§5：compile 行与 eager 行的差别），所以**换路径后的拟合结果
  与之前的 compile 版不会逐位相同**，但同一路径跑两次逐位相同；`_canon_grads` 可以保留。
- 想与之前的 compile 结果逐位一致时用 `rig.phys.auto_graph = True`（保留 `phys_compile=True`），速度约为 fused 的一半。
- 物理现在几乎与 B 无关（B ≤ 60 时每个批元素一个 SM），多图层 / 多单元摞成一批是免费的。
