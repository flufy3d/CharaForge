# moc3 解析器与 PyTorch 变形器（A1）

> 授权：解析 moc3 的风险用户已知晓并决定采用。本解析器只作内部拟合工具，不进播放器，不对外分发，不发布到仓库之外。
> 最终渲染与打分仍然只认 runtime（Cubism Core + Framework）。

| 文件 | 作用 |
|---|---|
| `tools/gpu/l2d/moc3.py` | moc3 解析、校验、json 摘要、drawable 静态数据（Core 口径） |
| `tools/gpu/l2d/deform.py` | PyTorch 变形器 `L2D`：参数 [B, P] → 顶点 / 不透明度 / draw order / render order / 颜色 / 可见性，可求导 |
| `tools/gpu/scripts/a1_core_dump.mjs` | Node 里用官方 Core 逐组参数 `csmUpdateModel`，dump 参考输出 |
| `tools/gpu/scripts/a1_scene_params.ts` | 用 runtime 的 LayerEvaluator 从 `scenes/deepseek-chan/scene.json` 求出真实参数帧 |
| `tools/gpu/scripts/a1_make_refs.py` | 生成参数集（随机 / 特殊值 / 场景 / 部件不透明度）并调用 Core dump |
| `tools/gpu/scripts/a1_compare.py` | 对拍：L2D vs Core，报最大误差、p99、最差 drawable |
| `tools/gpu/scripts/a1_features.py` | 本模型没用到的特性（isQuadSource = 0、反射、变形器颜色、非仿射网格外插）：在内存里改 moc 字节造出来再对拍 |
| `tools/gpu/scripts/a1_probe.py` | 探测工具：把一个 mesh 挂到任意变形器下、顶点换成探测点，比较 Core 与 L2D 的逐点结果 |
| `tools/gpu/scripts/a1_grad_bench.py` | 梯度检查（float64 中心差分 vs autograd）、速度与显存 |
| `tools/gpu/tests/test_a1_moc3.py`、`test_a1_deform.py` | 单元测试（unittest，也可用 pytest） |

## 1. 用法

```python
import sys; sys.path.insert(0, 'tools/gpu')
import torch
from l2d import moc3
from l2d.deform import L2D

m = moc3.load('assets/models/deepseek-chan/DeepSeek/大肥鱼.moc3')   # strict：任何不认识的内容都抛 Moc3Error
model = L2D(m, device='cuda', dtype=torch.float32)                  # 或 float64；编译约 1 秒
P = torch.as_tensor(m['parameters']['defaultValues'])[None]          # [B, P]，顺序 = model.param_ids（= Core 的参数顺序）
out = model(P, part_opacity=None)                                    # part_opacity [B, Q]，顺序 = model.part_ids，缺省全 1
out.verts          # [B, V, 2] 画布像素（原点左上，y 向下），全部 drawable 按 drawable 顺序拼接；第 d 个：verts[:, vo[d]:vo[d+1]]
                   #   vo = model.vertex_offsets；out.vertices_of(model, d) 等价
out.opacity        # [B, D] 最终不透明度（keyform × 父变形器链 × 部件不透明度级联）
out.draw_order     # [B, D] int64，同 csmGetDrawableDrawOrders
out.render_order   # [B, D] int64，同 csmGetRenderOrders（drawable 在绘制序列中的位置，0 最先画）；out.draw_sequence() 是逆排列
out.multiply       # [B, D, 3] 乘算色；out.screen [B, D, 3] 屏幕色（Core 还有恒为 1 的第 4 分量，这里不输出）
out.visible        # [B, D] bool，同动态标志 csmIsVisible
out.vertex_updated # [B, D] bool，同动态标志 csmVertexPositionsDidChange（Framework 生成 clipping mask 时用）
```

- 可求导的输出：`verts`、`opacity`、`multiply`、`screen`，对 `params` 与 `part_opacity` 求导；`draw_order`、`render_order`、`visible` 是离散量。
- 参数先被夹到 [min, max]（与 Core 相同），范围外梯度为 0；离 key 不到 0.001 时吸附到 key（见第 3 节），这一小段里梯度也是 0。
- 静态数据（给光栅化）：`moc3.drawable_static(m)` 返回 Core 口径的 `uvs [V, 2]`（v 已翻转）、每个 drawable 的三角形 `indices`（绕序已反转）、
  `texture`、`flags`（bit0 加算、bit1 乘算、bit2 双面、bit3 反转遮罩）、`masks`（已去掉 −1）、`parent_part`、`vertex_offsets`，均与 Core 逐项一致。
- 给 A3 光栅化：`LayerInput.from_arrays(out.verts, out.opacity, out.multiply, out.screen, out.render_order, out.visible, to_screen, out.vertex_updated)`，口径一致。
- 显存：前向不保存逐点中间量（warp / rotation 变换是自定义 autograd Function，反向时分块重算）。每个样本约 4MB（float32 前向）/ 7MB（前向 + 反向）。
  样本之间互相独立，批太大时可以分块前向 + 反向、累加梯度，结果完全相同。
- 调试：`model.debug = {}` 之后前向会把每层变形器的世界状态（网格、仿射项、旋转参数、属性表）记进 `model.debug['levels']`。
- 摘要：`tools/.venv/bin/python -I tools/gpu/l2d/moc3.py <moc3> -o out.json [--full]`；本模型的摘要在 `.data/gpu/moc3_summary.json`（`--full` 版含 keyform 明细）。

## 2. 文件格式（只写我们解析出来的结构）

小端；所有数组在文件里按 64 字节对齐，段之间的填充必须是 0（解析器逐字节检查，非零即报「未解析的数据」）。

- **文件头**（0x00，64 字节）：`'MOC3'`、版本字节（1 = V3.00，2 = V3.03，3 = V4.00，4 = V4.02，5 = V5.00；与 Core 的 csmMocVersion 相同）、
  端序字节（1 = 大端，不支持），其余保留为 0。本模型版本 5。版本 6（V5.03，Cubism 5.3 起，有离屏 / 部件颜色等新结构）明确报错，未实现。
- **段偏移表**（0x40）：每项 u32，指向一个数组。V5.00 共 152 项；表之后到第一个段之前必须全 0（否则报「不认识的段」）。
- **计数表**（段 0，64 个 u32）：parts、deformers、warpDeformers、rotationDeformers、artMeshes、parameters、partKeyforms、
  warpDeformerKeyforms、rotationDeformerKeyforms、artMeshKeyforms、keyformPositions（float 个数）、parameterBindingIndices、keyformBindings、
  parameterBindings、keys、uvs（float 个数）、positionIndices、drawableMasks、drawOrderGroups、drawOrderGroupObjects、glue、glueInfo、glueKeyforms；
  V4.02 起：keyformColors（乘算）、keyformColors（屏幕）、blendShapeParameterBindings、blendShapeKeyformBindings、blendShapesWarpDeformers、
  blendShapesArtMeshes、blendShapeConstraintIndices、blendShapeConstraints、blendShapeConstraintValues；V5.00 起：blendShapesParts、
  blendShapesRotationDeformers、blendShapesGlue。其余计数必须为 0。
- **画布**（段 1）：pixelsPerUnit、originX、originY、width、height（f32），1 字节 flags（本模型 0，含义未知，原样报出）。

各段（括号内是元素类型；`rt` = 运行时占位，每项 8 字节、文件里为 0；`id` = 64 字节、以 0 结尾的字符串）：

| 组 | 字段 |
|---|---|
| parts | rt, ids, keyformBindingSourcesIndices, keyformSourcesBeginIndices/Counts, isVisible, isEnabled, parentPartIndices |
| deformers | rt, ids, keyformBindingSourcesIndices, isVisible, isEnabled, parentPartIndices, parentDeformerIndices, types（0 warp / 1 rotation）, specificSourcesIndices（在 warp 或 rotation 表里的下标） |
| warpDeformers | keyformBindingSourcesIndices, keyformSourcesBeginIndices/Counts, vertexCounts（=(rows+1)(cols+1)）, rows, columns；V3.03 起 isQuadSource；V4.02 起 keyformColorSourcesBeginIndices |
| rotationDeformers | keyformBindingSourcesIndices, keyformSourcesBeginIndices/Counts, baseAngles（度）；V4.02 起 keyformColorSourcesBeginIndices |
| artMeshes | rt×4, ids, keyformBindingSourcesIndices, keyformSourcesBeginIndices/Counts, isVisible, isEnabled, parentPartIndices, parentDeformerIndices, textureNos, drawableFlags（u8）, vertexCounts, uvSourcesBeginIndices, positionIndexSourcesBeginIndices/Counts, drawableMaskSourcesBeginIndices/Counts；V4.02 起 keyformColorSourcesBeginIndices |
| parameters | rt, ids, maxValues, minValues, defaultValues, isRepeat, decimalPlaces, parameterBindingSourcesBeginIndices/Counts；V4.02 起 parameterTypes（0 普通 / 1 blend shape）, blendShapeParameterBindingSourcesBeginIndices/Counts |
| parameterExtensions（V4.00） | rt, keysSourcesBeginIndices/Counts（参数自己的 key 表 = Core 的 keyValues） |
| partKeyforms | drawOrders（f32） |
| warpDeformerKeyforms | opacities, keyformPositionSourcesBeginIndices；V5.00 起 keyformMultiplyColorSourcesIndices, keyformScreenColorSourcesIndices |
| rotationDeformerKeyforms | opacities, angles（度）, originX, originY, scales, isReflectX, isReflectY；V5.00 起两组颜色索引 |
| artMeshKeyforms | opacities, drawOrders, keyformPositionSourcesBeginIndices；V5.00 起两组颜色索引 |
| keyformPositions | xys（f32，warp 网格点 / mesh 顶点的 keyform 坐标） |
| parameterBindingIndices → keyformBindings → parameterBindings → keys | keyform 绑定 = 若干「参数绑定」（参数 + key 表）；keyform 多维数组第一维变化最快 |
| uvs、positionIndices（u16）、drawableMasks | UV、三角形索引、mask 列表（−1 表示空） |
| drawOrderGroups | objectSourcesBeginIndices/Counts, objectSourcesTotalCounts, maximumDrawOrders, minimumDrawOrders |
| drawOrderGroupObjects | types（0 mesh / 1 部件）, indices, selfIndices（部件对象对应的组号） |
| glue / glueInfo / glueKeyforms | rt, ids, keyform 绑定与区间, artMeshIndicesA/B, glueInfoSourcesBeginIndices/Counts；info = weights + positionIndices（成对：A 的顶点、B 的顶点）；keyform = intensities |
| keyformColorsMultiply / Screen（V4.02） | r, g, b 各一个 f32 数组 |
| blendShapeParameterBindings（V4.02） | keysSourcesBeginIndices/Counts, baseKeyIndices |
| blendShapeKeyformBindings | parameterBindingSourcesIndices, keyformSourcesBlendShapeIndices/Counts, blendShapeConstraintIndexSourcesBeginIndices/Counts |
| blendShapesWarpDeformers / ArtMeshes（V4.02）, Parts / RotationDeformers / Glue（V5.00） | targetIndices, blendShapeKeyformBindingSourcesBeginIndices/Counts |
| blendShapeConstraintIndices / Constraints / ConstraintValues | 约束 = 参数 + 分段线性曲线（keys, weights） |

校验（`moc3.validate`）：所有索引与区间都检查范围；父变形器必须排在子之前；keyform 个数 = 各维 key 数之积；
warp 的 vertexCounts = (rows+1)(cols+1)；三角形索引不超出顶点数；glue 顶点索引不越界；blend shape 的 keyform 区间与位置数据不越界等。
本模型：0 个警告。

Core 输出与文件的口径差异（已逐项对拍）：UV 的 v 被翻转（1 − v）；每个三角形的顶点顺序被反转；mask 列表去掉 −1；
顶点 y 向上（Core 单位：x_px = originX + x·ppu，y_px = originY − y·ppu）。L2D 内部用 y 向下的单位坐标，输出直接是画布像素。

## 3. 求值语义（全部由 Core 对拍或改字节探测确认）

1. **参数**：先夹到 [min, max]（实测：超出 max 的值与 max 结果逐位相同）。repeat 参数也按夹取处理——本模型的 4 个 repeat 参数都没有
   keyform 绑定，Core 里是否循环无法验证（Framework 的 setParameterValue 才做循环，传给 Core 的值已在范围内）。
2. **keyform 插值**：每个参数绑定求段号与 t（夹到 [0, 1]，不外插）。**离某个 key 不到 0.001 时吸附到该 key**（实测 LCry = 0.9003 → 按 0.9 计算）。
   keyform 绑定对各维做多线性组合。blend shape 的参数绑定**不吸附**。
3. **key 范围外**：参数超出某绑定的 key 范围 0.001 以上（[k₀ − 0.001, k_last + 0.001] 之外）时，该对象「范围外」：
   - 变形器范围外：它和所有子孙都不被 Core 求值（顶点、不透明度、draw order、颜色都不写；新建模型时为 0），子孙不可见；
   - mesh 自身范围外：只做 keyform 插值（不透明度、颜色、draw order 是自身 keyform 的值，不乘父与部件），不变形，不可见。
   L2D 照此输出（`vertex_updated` = 父链有效且自身在范围内）。注意 Core 对这些 drawable 保留上一次的值，所以对拍时每个样本都新建 Model。
4. **blend shape**：keyform 存的是增量（base key 处为 0）。权重 = 参数在 blend shape key 上的线性插值 × 约束权重；
   **多个约束取最小值**（不是乘积，实测）。增量加到 warp 网格、mesh 顶点、rotation 的角度 / 原点 / 缩放、不透明度、颜色上（全部按 keyform 线性叠加）。
5. **warp**：子点是局部 [0,1]² 坐标（u 沿列、v 沿行；网格点行优先、列变化最快）。
   - 网格内：格子内双线性；isQuadSource = 0 时按 (1,0)–(0,1) 对角线分成两个三角形；
   - 网格外 (−2, 3)² 的带状区：虚拟网格线 −2 / 3 上的点取仿射外推 A(u, v) = c + (u − ½)·du + (v − ½)·dv
     （c = 四角平均，du = 上下两条边向量的平均，dv = 左右两条边的平均），在网格边界点与虚拟点之间**一律用三角形插值**；
   - 更远处：纯仿射 A(u, v)。
6. **rotation**：世界变换 q = o + s·R(θ)·diag(±1, ±1)·p（反射在旋转之前）。
   - 根：o、θ、s 取 keyform 值；
   - 有父时：o = 父变换(局部原点)；θ = keyform 角 + 父方向角 + baseAngle，父方向角由「局部原点向上偏一点」的点经父变换后的方向测得
     （父为 warp 偏 (0, −0.1) 局部坐标，父为 rotation 偏 (0, −10)；改成 0.05 / 0.2 时误差都上升到 100px 以上，0.1 是确定的）；
     s = keyform 缩放 × 父的总缩放（warp 的总缩放 = 它的父的总缩放，根为 1；rotation 的总缩放 = 它的世界缩放）。
   - 反射标志按插值后 > 0.5 判定（反射对拍通过；插值中间值的取法未验证，本模型反射全为 0）。
7. **不透明度与颜色**：沿变形器链级联：不透明度相乘，乘算色相乘，屏幕色 1 − (1 − a)(1 − b)；
   mesh 再乘部件不透明度（部件及其所有祖先部件的不透明度之积；没有父部件的 mesh 不受影响）。可见 = 父链有效 ∧ 自身在范围内 ∧ 不透明度 > 0。
8. **glue**：在世界坐标里，按文件顺序，对每对顶点 (A_i, B_j, w_A, w_B)：A ← A + (B − A)·w_A·强度，B ← B + (A − B)·w_B·强度（用的是同一时刻的旧值）。
9. **draw order**：mesh 的 draw order = int(插值值 + 0.001)（float32；实测 500.68 → 500、500.9990 → 501、500.9988 → 500），部件同理。
   render order：从根 draw order 组开始，组内按 draw order 稳定排序（相同时按组内对象顺序），部件对象展开成它自己的组；
   排序用的是插值得到的 draw order，不管可见与否（Core 报告的 draw order 对没求值的 drawable 是 0）。

实现（deform.py）：每个参数绑定求段号与 t → keyform 绑定的 2^M 个角点权重 → 每个 keyform 的权重 Wkf[B, NKF]（blend shape 也只是往 Wkf 里加权重）；
网格点 / 顶点 = 稀疏矩阵（CSR）× Wkf，其余属性 = index_add。变形器按深度分 23 层，第 L 步把本层 warp 的网格点、rotation 的原点与方向点、
父在上一层的 mesh 顶点按父类型分段，warp 父、rotation 父各用一个自定义 autograd Function 变换（反向只存输入、分块重算，
`WARP_CHUNK_ELEMS` 控制块大小）。没有逐 drawable 的 Python 循环；glue 按「互不共享 mesh」分成 17 波，每波一次向量化。

## 4. 对拍

```bash
# 1) 参数集 + Core 参考输出（.data/gpu/ref/<名>.{bin,json,params.npy}）
node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types tools/gpu/scripts/a1_scene_params.ts 6   # 场景帧 → scene_frames.{bin,json}
tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py random  -n 300    # 300 组：各参数在 [min, max] 均匀分布
tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py special -n 160    # 161 组：恰好在 key 上 / 端点 / 略超范围（1% 或 5e-4）/ 混合（含 key ± 1e-4、± 2e-3）/ 默认值
tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py scene             # 282 帧：47 个图层、每层 6 帧（runtime LayerEvaluator 实际求值，含物理）
tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py partop -n 100     # 100 组：随机参数 + 随机部件不透明度
# 2) 对拍
tools/.venv/bin/python -I tools/gpu/scripts/a1_compare.py .data/gpu/ref/{random,special,scene,partop} --dtype float64 --json .data/gpu/ref/a1_report_f64.json
tools/.venv/bin/python -I tools/gpu/scripts/a1_compare.py .data/gpu/ref/{random,special,scene,partop} --dtype float32 --json .data/gpu/ref/a1_report_f32.json
tools/.venv/bin/python -I tools/gpu/scripts/a1_features.py          # 本模型没用到的特性
tools/.venv/bin/python -I tools/gpu/scripts/a1_grad_bench.py grad   # 梯度
tools/.venv/bin/python -I tools/gpu/scripts/a1_grad_bench.py bench --dtype float32 --batches 1,64,512
tools/.venv/bin/python -I tools/gpu/tests/test_a1_moc3.py && tools/.venv/bin/python -I tools/gpu/tests/test_a1_deform.py
```

Core：Cubism SDK Core 6.0.1（`vendor/cubism/core/live2dcubismcore.js`），每次 update 约 8～10ms（CPU）。顶点误差只统计 Core 可见的 drawable
（不可见的 Core 不更新顶点）；其余各项统计全部 drawable。门槛：顶点 < 0.1px，不透明度 / 颜色 < 1e-4，render order、draw order、可见性、
vertex_updated 完全一致。结果（2026-11-06）：

| 参数集 | dtype | 样本 | 比较的顶点数 | 顶点 max (px) | p99 | p99.9 | 均值 | 不透明度 max | 乘算 max | 屏幕 max | render / draw order、可见性 | 最差 drawable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | f64 | 300 | 14,541,906 | 0.0434 | 0.0060 | 0.0114 | 0.00081 | 6.0e-7 | 5.4e-7 | 6.0e-8 | 全部一致 | ArtMesh0 |
| special | f64 | 161 | 6,190,776 | 0.0633 | 0.0077 | 0.0162 | 0.00103 | 3.0e-7 | 3.0e-7 | 6.0e-8 | 全部一致 | ArtMesh216 |
| scene | f64 | 282 | 11,327,804 | 0.0224 | 0.0062 | 0.0114 | 0.00073 | 6.6e-7 | 6.6e-7 | 6.0e-8 | 全部一致 | ArtMesh646 |
| partop | f64 | 100 | 4,829,343 | 0.0228 | 0.0060 | 0.0104 | 0.00081 | 4.8e-7 | 4.8e-7 | 6.0e-8 | 全部一致 | ArtMesh216 |
| random | f32 | 300 | 同上 | 0.0450 | 0.0062 | 0.0136 | 0.00069 | 4.8e-7 | 6.0e-7 | 3.0e-8 | 全部一致 | ArtMesh0 |
| special | f32 | 161 | 同上 | 0.0718 | 0.0078 | 0.0174 | 0.00074 | 3.0e-7 | 3.6e-7 | 3.0e-8 | 全部一致 | ArtMesh216 |
| scene | f32 | 282 | 同上 | 0.0243 | 0.0058 | 0.0110 | 0.00058 | 4.8e-7 | 6.0e-7 | 3.0e-8 | 全部一致 | ArtMesh216 |
| partop | f32 | 100 | 同上 | 0.0340 | 0.0065 | 0.0140 | 0.00070 | 4.8e-7 | 6.0e-7 | 3.0e-8 | 全部一致 | ArtMesh216 |

剩下的误差是 float32 舍入：我们自己的 float32 与 float64 结果之差（special 上 max 0.060px、p99 0.0078）与「float64 vs Core」（max 0.063、p99 0.0077）
同量级；最差的都是 23 层深的链（ArtMesh216 / ArtMesh0：warp 与 rotation 交替十几层）。

本模型没用到的特性（`a1_features.py`，12 组随机参数，改字节后 Core vs L2D）：

| 特性 | 顶点 max | 其余 |
|---|---|---|
| 根 warp 改成非仿射网格（网格外带状区与仿射外插） | 0.044px | 全部一致 |
| 1/3 的 warp 改成 isQuadSource = 0 | 0.041px | 全部一致 |
| 一半 rotation 的 keyform 打开 isReflectX / isReflectY | 0.011px | 全部一致 |
| 1/5 的变形器 keyform 改成非默认乘算 / 屏幕色 | 0.021px | 乘算 2.8e-7、屏幕 1.9e-7 |

梯度（`a1_grad_bench.py grad`，float64，场景帧，损失 = 顶点、不透明度、颜色的固定随机线性组合；24 个参数，避开 key ± 0.006 与端点）：
autograd 与中心差分的最大相对误差 5.7e-9。自定义 Function 另用 `torch.autograd.gradcheck` 覆盖网格内、四边与四角带状区、远处仿射、三角形模式、多块分块（测试里）。

速度与显存（RTX 5080 Laptop，float32，场景帧；同时有 4 个代理在用 GPU / CPU，数字有 ±50% 的波动）：

| B | 前向 | 前向 + 反向 | 前向显存峰值 | 前向 + 反向显存峰值 |
|---|---|---|---|---|
| 1 | ≈120 ms | ≈340 ms | 7 MB | 195 MB |
| 64 | ≈170 ms（2.6 ms/样本） | ≈450 ms | 374 MB | 619 MB |
| 512 | 370～500 ms（0.7～1.0 ms/样本） | 1.3～1.8 s（2.5～3.6 ms/样本） | 2.1 GB | 3.5 GB |

（GPU 被别的代理占用较重时测到过 B = 1 前向 175ms、B = 512 前向 + 反向 3.3s，显存不变。）

float64：B = 64 前向 285～300 ms、前向 + 反向 0.92 s、显存 0.67 / 1.17 GB；B = 128 前向 + 反向 1.44 s、2.2 GB（float64 的 B = 512 前向 + 反向约需 9GB，超出本代理 4GB 的显存额度，没跑；可分块）。
小批量时是 CPU 发射内核的开销（约 1 万次小内核）在主导，GPU 实际计算 B = 256 时约 90ms。对照：Core 一次 update 8～10ms，一个雅可比（约 170 个参数的有限差分）约 1.7s；
这里一个样本的完整梯度约 2.5ms。

## 5. 已知限制 / 未验证

- 只支持小端、版本 1～5；版本 6（V5.03）报错。本模型是 V5.00，V3.x / V4.x 的分支（V4.02 的颜色按对象连续存放、没有 parameterExtensions 等）按结构推断实现，没有样本验证。
- 本模型里为 0、无法对拍的：部件 / glue 的 blend shape（代码路径存在）、根级 rotation 变形器（是否加 baseAngle：现在只在有父时加）、
  repeat 参数的 keyform、canvas flags 字节的含义、rotation 反射在插值中间值的取法、blend shape keyform 的颜色增量（本模型全 0）、
  约束曲线是否吸附（本模型的约束 key 都是整数，与吸附无关）。
- 方向测量在 Core 里有「位移为 0 时缩小 10 倍重试」的退化处理，这里没做（只在父变换把那一小段压成 0 时才有区别）。
- key 吸附与「范围外」在 0.001 这条边界上的严格 / 非严格不等号未确定（只影响恰好落在边界上的值）。
