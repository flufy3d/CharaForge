"""
可微分的 Cubism 物理（PyTorch 移植）。

移植对象：vendor/cubism/framework/src/physics/cubismphysics.ts（Cubism SDK for Web Framework R5，
Live2D Open Software License，Copyright(c) Live2D Inc.），以及 packages/runtime 对它的用法
（instance.ts / evaluator.ts / physicsState.ts）。逐段对应关系见 tools/gpu/docs/physics.md。

与 Framework 逐行一致的地方（前向结果）：
  - 输入归一化 normalizeParameterValue（夹到参数范围、按「范围中点」分正负两段线性映射、反转、权重）；
  - 每个 setting 的 totalTranslation 旋转（包括 Framework 里 y 分量用的是「已旋转的 x」这一写法）；
  - updateParticles：重力方向、风、加速度、空气阻力（lastGravity→currentGravity 夹角 / 5 的旋转）、
    delay = delay·dt·30、半径约束、MovementThreshold 吸附、delay ≠ 0 时的速度更新、mobility；
  - 输出：X / Y / Angle 三种，Angle 的父方向（顶点 ≥ 2 取上一段、否则取 −options.gravity），反转，
    scale（X / Y 的 translationScale 在 Framework 里从未被赋值，恒为 0；Angle 用 json 的 Scale），
    夹到目标参数范围，weight < 100 时与目标当前值线性混合；
  - R5 固定步长：physics3 的 Fps → physicsDeltaTime；_currentRemainTime 累积（> 5 秒清零）；
    每个子步先把 _currentRigOutputs 拷到 _previousRigOutputs，输入在 _parameterInputCaches 与当前参数之间
    按 physicsDeltaTime / remainTime 线性插值；最后按 remainTime / physicsDeltaTime 在前后两次结果间内插写回参数；
  - setting 之间通过 _parameterCaches 传递（前面 setting 的输出是后面 setting 的输入）；
  - Framework 的 _parameterCaches / _parameterInputCaches 与模型参数都是 Float32Array：float64 模式下在
    同样的位置做 float32 舍入（直通估计：前向舍入、反向按恒等传梯度），粒子与 rig 输出保持 double；
  - stabilization()（layer.physics.stabilize 时 runtime 会调用）。

为了求导做的处理（不改变前向结果）：
  - atan2 在 (0, 0) 处的梯度置 0（自定义 autograd）；
  - 向量归一化在长度为 0 时前向给 NaN（与 JS 0/0 相同），反向不产生 NaN（分母换成 1 再用 where 屏蔽）；
  - delay = 0 的粒子不更新速度（常量分支），不做除法；
  - 链长不同的 setting 打包成矩形时，填充粒子是一条合法的摆（半径 1），不会产生 NaN，也不影响真实粒子。

并行化：setting 按数据依赖分层（读后写 / 写后读 / 写后写都保持 Framework 的顺序语义），同一层的
setting 一次算完；同一层里写同一个参数的多个输出按原顺序分成多轮写入。

用法（批量 [B, P]，P = 模型参数个数，顺序与 moc3 / Core 一致）：
    rig = PhysicsRig.from_files(physics3_json_path, model_params_json_path, device='cuda', dtype=torch.float64)
    st = rig.initial_state(B)                 # = CubismPhysics.create() 之后、未 stabilization 的状态
    st, out = rig.step(st, params, dt=1/60)   # 规范 e：物理前进一步；out 为物理之后的全部参数
    st, out = rig.stabilize(st, params)       # Cubism stabilization（physics.stabilize = true 时）
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

# Framework 常量（cubismphysics.ts 文件头）
AIR_RESISTANCE = 5.0
MAXIMUM_WEIGHT = 100.0
MOVEMENT_THRESHOLD = 0.001
MAX_DELTA_TIME = 5.0

_TYPES = {'X': 0, 'Y': 1, 'Angle': 2}


# ---------------------------------------------------------------- 求导安全的基本运算

class _SafeAtan2(torch.autograd.Function):
    """torch.atan2，(0, 0) 处梯度为 0 而不是 NaN。"""

    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, g):
        y, x = ctx.saved_tensors
        r2 = x * x + y * y
        ok = r2 > 0
        inv = torch.where(ok, g / torch.where(ok, r2, torch.ones_like(r2)), torch.zeros_like(r2))
        return x * inv, -y * inv


def atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _SafeAtan2.apply(y, x)


class _Index:
    """列下标 idx（常量）及其逆表，供确定性的 gather 使用（A8）。

    x[:, idx] 的原生反向是「按 idx 散射累加」：torch.compile 生成的是 float64 原子加，下标有重复时求和顺序随线程调度变，
    梯度低位逐次不同。这里预先建好逆表 inv[j, d] = 第 d 个满足 idx[k] = j 的 k（不足的填 n，指向补的零列），
    反向按固定顺序逐项相加，结果逐次相同。ignore 中的源列不回传梯度（只用于丢弃的填充列）。"""

    def __init__(self, idx: Sequence[int], n_src: int, device, ignore=()):
        idx = [int(i) for i in idx]
        pos = [[] for _ in range(n_src)]
        for k, j in enumerate(idx):
            if j not in ignore:
                pos[j].append(k)
        D = max([1] + [len(p) for p in pos])
        n = len(idx)
        self.idx = torch.tensor(idx, device=device, dtype=torch.long)
        self.inv = torch.tensor([[p[d] if d < len(p) else n for d in range(D)] for p in pos], device=device,
                                dtype=torch.long).reshape(n_src, D)
        self.n_src = n_src

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _Gather.apply(x, self.idx, self.inv)


class _Gather(torch.autograd.Function):
    """y = x[:, idx]；反向 gx[:, j] = Σ_d gy[:, inv[j, d]]（固定顺序，确定性）。"""

    @staticmethod
    def forward(ctx, x, idx, inv):
        ctx.save_for_backward(inv)
        return x[:, idx]

    @staticmethod
    def backward(ctx, g):
        inv, = ctx.saved_tensors
        gz = torch.cat([g, g.new_zeros(g.shape[0], 1)], 1)
        acc = gz[:, inv[:, 0]]
        for d in range(1, inv.shape[1]):
            acc = acc + gz[:, inv[:, d]]
        return acc, None, None


def direction_to_radian(fx, fy, tx, ty):
    """CubismMath.directionToRadian(from, to)：atan2(to) − atan2(from)，折回 [−π, π]。"""
    ret = atan2(ty, tx) - atan2(fy, fx)
    ret = torch.where(ret < -math.pi, ret + math.pi * 2.0, ret)
    ret = torch.where(ret > math.pi, ret - math.pi * 2.0, ret)
    return ret


def normalize(x, y):
    """CubismVector2.normalize()：除以 (x²+y²)^0.5；长度 0 时前向为 NaN（同 JS），反向不出 NaN。"""
    l2 = x * x + y * y
    ok = l2 > 0
    l = torch.sqrt(torch.where(ok, l2, torch.ones_like(l2)))
    nan = torch.full_like(l2, float('nan'))
    return torch.where(ok, x / l, nan), torch.where(ok, y / l, nan)


# ---------------------------------------------------------------- 状态

@dataclass(frozen=True)
class PhysicsState:
    """CubismPhysics 的完整跨帧状态（对应 runtime/physicsState.ts 的快照）。

    粒子按「组」存放（依赖层内再按链长分组）：px[g] 等是 [B, R_g, L_g]（第 g 组的 R_g 个 setting，链长 L_g）。
    lastPosition、force 在每次 updateParticles 开头被重写，不是状态；同一条链的 lastGravity 总是相同，按 setting 存。
    """
    px: tuple
    py: tuple
    vx: tuple
    vy: tuple
    gx: tuple  # lastGravity，[B, R_l]
    gy: tuple
    cur: torch.Tensor   # _currentRigOutputs，[B, NO]（setting、output 顺序展平）
    prev: torch.Tensor  # _previousRigOutputs，[B, NO]
    in_cache: torch.Tensor | None  # _parameterInputCaches（只存物理用到的参数，[B, Q]）；None = Framework 的 null
    remain: float  # _currentRemainTime（整批共用）
    resolved: bool  # 输出目标参数索引是否已解析（第一次子步或 stabilization 之后为 True，影响 interpolate）

    def detach(self) -> 'PhysicsState':
        d = lambda ts: tuple(t.detach() for t in ts)
        return replace(self, px=d(self.px), py=d(self.py), vx=d(self.vx), vy=d(self.vy), gx=d(self.gx), gy=d(self.gy),
                       cur=self.cur.detach(), prev=self.prev.detach(),
                       in_cache=None if self.in_cache is None else self.in_cache.detach())


@dataclass
class _Level:
    rigs: list           # setting 索引（升序）
    L: int               # 填充后的链长
    K: int               # 填充后的每个 setting 的输入个数
    # 输入：[R, K] 的槽位指向本层输入列表（NI 个，NI 处为全零填充）
    in_q: torch.Tensor   # [NI] 物理参数空间（Q）里的索引
    in_slot: torch.Tensor  # [R*K] → 0..NI（NI = 填充）
    in_type: torch.Tensor  # [R, K, 3] one-hot（X / Y / Angle），填充为 0
    n_min: torch.Tensor; n_max: torch.Tensor; n_mid: torch.Tensor
    r_pos: torch.Tensor; r_neg: torch.Tensor; m_norm: torch.Tensor
    has_pos: torch.Tensor; has_neg: torch.Tensor; in_sw: torch.Tensor  # 反转符号 × 权重
    in_const: torch.Tensor  # [R, 3]：模型里不存在的输入参数的常量贡献（见 _build）
    # 粒子：[R, L]
    acc: torch.Tensor; delay: torch.Tensor; radius: torch.Tensor; mobility: torch.Tensor
    thr: torch.Tensor    # [R]
    init_py: torch.Tensor  # [R, L]（initialPosition.y；x 恒为 0）
    # 输出（只含顶点索引合法的输出）
    out_gidx: torch.Tensor   # [NOv] 全局输出索引
    out_flat: torch.Tensor   # [NOv] r*L + v
    out_parent: torch.Tensor  # [NOv] bool：v ≥ 2
    out_type: torch.Tensor   # [NOv] 0/1/2
    out_sign: torch.Tensor   # [NOv] ±1
    rnd_items: list          # [(全局输出下标, 本组输出下标, q_dest, scale, min, max, weight)]，由 _build 合成每层的写入轮次
    out_vertex_ok: bool
    # 确定性 gather（A8）：C[:, in_q]、c[:, in_slot]（填充列不回传）、粒子 [i]、[i−1]、[父]
    g_in: '_Index' = None
    g_slot: '_Index' = None
    g_o1: '_Index' = None
    g_o0: '_Index' = None
    g_o2: '_Index' = None


class PhysicsRig:
    """由 physics3.json 与模型参数表构建的物理求值器。"""

    def __init__(self, physics: dict, param_ids: Sequence[str], pmin: Sequence[float], pmax: Sequence[float], *,
                 device='cpu', dtype=torch.float64, emulate_f32: bool = True, split_by_length: bool = False,
                 gravity=(0.0, -1.0), wind=(0.0, 0.0)):
        """
        physics:    physics3.json 的内容
        param_ids / pmin / pmax: 模型参数表（Core 的 Float32 值，顺序即参数索引）
        emulate_f32: float64 模式下在 Framework 用 Float32Array 的位置做 float32 舍入（缺省开，与原实现一致）
        split_by_length: 每个依赖层再按链长分组（没有填充，B=64 显存省约 1/3，但 kernel 多、compile 后慢 2～3 倍）；
                    缺省 False = 每层一组、链长填充到该层最长（填充粒子是合法的摆，不影响真实粒子）
        gravity / wind: CubismPhysics 的 Options（runtime 不改，缺省 (0,−1) / (0,0)；json 里的 EffectiveForces 不参与计算）
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.emulate_f32 = emulate_f32 and dtype == torch.float64
        self.split_by_length = split_by_length
        self.opt_gravity = (float(gravity[0]), float(gravity[1]))
        self.opt_wind = (float(wind[0]), float(wind[1]))
        self.param_ids = list(param_ids)
        self.P = len(self.param_ids)
        self._pmin = [float(x) for x in pmin]
        self._pmax = [float(x) for x in pmax]
        self.fps = float(physics['Meta'].get('Fps', 0.0) or 0.0)
        self._build(physics)
        # A8：CUDA graph 快速路径（graphed_rollout）。auto_graph = True 时 rollout() 自动改走 graphed_rollout（缺省关）。
        self.auto_graph = False
        # A8：Triton 融合核快速路径（fused_rollout）。auto_fused = True 时 rollout() 自动改走 fused_rollout（缺省关，优先于 auto_graph）。
        self.auto_fused = False
        self.graph_max_instances = 64   # 所有键合计最多保留的图实例数（每个实例一份私有显存池）
        self._graphs: dict = {}         # key → [_PhysGraph, ...]（按最近使用排序）
        self.graph_stats = dict(captures=0, replays=0, fallbacks=0)
        self._fused_plan = None

    # ------------------------------------------------------------ 构建

    @classmethod
    def from_files(cls, physics_path: str, model_params_path: str, **kw) -> 'PhysicsRig':
        """model_params_path：{ids, min, max} 的 JSON（a2_ref_dump.ts 输出的 model_params.json 即是）。"""
        with open(physics_path, encoding='utf-8') as f:
            phys = json.load(f)
        with open(model_params_path, encoding='utf-8') as f:
            tab = json.load(f)
        return cls(phys, tab['ids'], tab['min'], tab['max'], **kw)

    def _t(self, x, dtype=None):
        return torch.tensor(x, device=self.device, dtype=dtype or self.dtype)

    def _build(self, physics: dict):
        idx = {pid: i for i, pid in enumerate(self.param_ids)}
        settings = physics['PhysicsSettings']
        self.S = len(settings)

        # 物理参数空间 Q：所有存在于模型里的输入 / 输出参数
        q_ids = []
        for s in settings:
            for i in s['Input']:
                q_ids.append(i['Source']['Id'])
            for o in s['Output']:
                q_ids.append(o['Destination']['Id'])
        q_glob = sorted({idx[x] for x in q_ids if x in idx})
        self.q_glob = q_glob
        q_of = {g: k for k, g in enumerate(q_glob)}
        self.q_index = self._t(q_glob, torch.long)
        self.Q = len(q_glob)

        # 每个 setting 的读 / 写集合（Q 空间），用于分层
        reads, writes = [], []
        for s in settings:
            reads.append({q_of[idx[i['Source']['Id']]] for i in s['Input'] if i['Source']['Id'] in idx})
            n = len(s['Vertices'])
            writes.append({q_of[idx[o['Destination']['Id']]] for o in s['Output']
                           if o['Destination']['Id'] in idx and 1 <= o['VertexIndex'] < n})
        level = [0] * self.S
        for k in range(self.S):
            lv = 0
            for j in range(k):
                if writes[j] & reads[k]:
                    lv = max(lv, level[j] + 1)   # 写后读：k 必须在 j 写完之后读
                if writes[j] & writes[k]:
                    lv = max(lv, level[j])       # 写后写：同层内按轮次保持顺序
                if reads[j] & writes[k]:
                    lv = max(lv, level[j])       # 读后写：k 的写不能早于 j 的读（同层时读在前）
            level[k] = lv
        self.setting_level = level
        nlev = max(level) + 1 if level else 0

        # 全局输出编号（setting、output 顺序展平）
        gbase, g = [], 0
        for s in settings:
            gbase.append(g)
            g += len(s['Output'])
        self.NO = g

        # 每层再按链长分组（减少填充）：self.levels 是所有组（按层排列），self.stages[l] = (该层的组下标, 写入轮次)。
        # 同一层的所有组先从同一份缓存读输入，全部算完后再按原顺序写回，与逐个 setting 顺序执行等价。
        self.levels: list[_Level] = []
        self.stages = []
        for lv in range(nlev):
            rigs = [k for k in range(self.S) if level[k] == lv]
            by_len: dict[int, list] = {}
            for k in rigs:
                by_len.setdefault(len(settings[k]['Vertices']) if self.split_by_length else 0, []).append(k)
            gids, items, off = [], [], 0
            for n in sorted(by_len):
                grp = self._build_level(settings, by_len[n], idx, q_of, gbase)
                gids.append(len(self.levels))
                self.levels.append(grp)
                items += [(g, loc + off, *rest) for (g, loc, *rest) in grp.rnd_items]
                off += len(grp.out_gidx)
            items.sort(key=lambda x: x[0])   # 全局输出顺序（同一参数的多次写入按原顺序分轮）
            seen: dict[int, int] = {}
            tagged = []
            for it in items:
                r = seen.get(it[2], 0)
                seen[it[2]] = r + 1
                tagged.append((r, *it))
            rounds = []
            for r in range(max([x[0] for x in tagged], default=-1) + 1):
                its = [x[1:] for x in tagged if x[0] == r]
                rounds.append(self._round_tensors([x[1] for x in its], [x[2] for x in its], [(None, None, None, *x[3:]) for x in its],
                                                  off, self.Q))
            self.stages.append((gids, rounds))

        # interpolate：所有目标参数存在的输出（全局顺序），按「同一参数之前出现的次数」分轮
        seen: dict[int, int] = {}
        ip = []
        for k, s in enumerate(settings):
            for j, o in enumerate(s['Output']):
                pid = o['Destination']['Id']
                if pid not in idx:
                    continue  # 目标参数不存在：Framework 写到越界位置，无效果
                p = idx[pid]
                r = seen.get(p, 0)
                seen[p] = r + 1
                ip.append((r, gbase[k] + j, p, self._scale(o), self._pmin[p], self._pmax[p], o['Weight'] / MAXIMUM_WEIGHT))
        self.output_params = sorted({x[2] for x in ip})          # 物理写入的参数（全局索引）
        self.input_params = sorted({idx[i['Source']['Id']] for s in settings for i in s['Input'] if i['Source']['Id'] in idx})
        self.output_param_index = self._t(self.output_params, torch.long)
        self._g_q = _Index(q_glob, self.P, self.device)
        self._g_outp = _Index(self.output_params, self.P, self.device)
        # interpolate 在「物理输出参数」的紧凑空间里写，最后一次性写回 [B, P]
        o_of = {p: k for k, p in enumerate(self.output_params)}
        self.interp_rounds = []
        for r in range(max([x[0] for x in ip], default=-1) + 1):
            items = [x for x in ip if x[0] == r]
            self.interp_rounds.append(self._round_tensors([x[1] for x in items], [o_of[x[2]] for x in items], items,
                                                          self.NO, len(self.output_params)))
        self.has_const = any(bool((lv.in_const != 0).any()) for lv in self.levels)
        ok = [lv.out_gidx for lv in self.levels if lv.out_vertex_ok]
        self._out_gidx_all = torch.cat(ok) if ok else None
        # 每个子步的 delay = delay·physicsDeltaTime·30（Fps > 0 时 physicsDeltaTime 固定，预先算好）
        self._delays_fixed = [self._delays(lv, 1.0 / self.fps) for lv in self.levels] if self.fps > 0.0 else None
        self._sched_cache: dict = {}
        self._frame_c = None

    @staticmethod
    def _scale(o) -> float:
        # getOutputScaleTranslationX/Y 返回 output.translationScale.x/y：Framework 从不赋值，恒为 0
        return float(o['Scale']) if o['Type'] == 'Angle' else 0.0

    def _round_tensors(self, src, dst, items, n_src, n_dst):
        return dict(
            src=self._t(src, torch.long), dst=self._t(dst, torch.long),
            g_src=_Index(src, n_src, self.device), g_dst=_Index(dst, n_dst, self.device),
            scale=self._t([x[3] for x in items]), vmin=self._t([x[4] for x in items]), vmax=self._t([x[5] for x in items]),
            w=self._t([x[6] for x in items]), full=self._t([x[6] >= 1.0 for x in items], torch.bool),
        )

    def _build_level(self, settings, rigs, idx, q_of, gbase) -> _Level:
        R = len(rigs)
        L = max(len(settings[k]['Vertices']) for k in rigs)
        K = max(len(settings[k]['Input']) for k in rigs)
        # ---- 输入
        in_q, in_c = [], []
        slot = [[-1] * K for _ in range(R)]
        typ = [[[0.0] * 3 for _ in range(K)] for _ in range(R)]
        const = [[0.0] * 3 for _ in range(R)]
        for r, k in enumerate(rigs):
            s = settings[k]
            nrm = s['Normalization']
            for j, i in enumerate(s['Input']):
                t = _TYPES[i['Type']]
                n = nrm['Angle'] if t == 2 else nrm['Position']
                nmin, nmax, ndef = float(n['Minimum']), float(n['Maximum']), float(n['Default'])
                w = float(i['Weight']) / MAXIMUM_WEIGHT
                sgn = 1.0 if i['Reflect'] else -1.0
                pid = i['Source']['Id']
                if pid not in idx:
                    # 参数不存在：Framework 读到 undefined，归一化里 sign(NaN) = 0 → 结果 = 归一化缺省值
                    const[r][t] += (ndef * sgn) * w
                    continue
                p = idx[pid]
                a, b = self._pmin[p], self._pmax[p]
                vmax = b if b > a else a          # CubismMath.max(parameterMaximum, parameterMinimum)
                vmin = a if b > a else b          # CubismMath.min(parameterMaximum, parameterMinimum)
                mn_norm = nmax if nmin > nmax else nmin
                mx_norm = nmin if nmin > nmax else nmax
                mid = vmin + abs(vmax - vmin) / 2.0   # getDefaultValue(minValue, maxValue)
                pl_pos, pl_neg = vmax - mid, vmin - mid
                in_c.append(dict(min=vmin, max=vmax, mid=mid,
                                 r_pos=(mx_norm - ndef) / pl_pos if pl_pos != 0.0 else 0.0,
                                 r_neg=(mn_norm - ndef) / pl_neg if pl_neg != 0.0 else 0.0,
                                 has_pos=pl_pos != 0.0, has_neg=pl_neg != 0.0, m=ndef, sw=(sgn, w)))
                slot[r][j] = len(in_q)
                in_q.append(q_of[p])
                typ[r][j][t] = 1.0
        NI = len(in_q)
        slot_flat = [NI if x < 0 else x for row in slot for x in row]
        # ---- 粒子
        acc = [[1.0] * L for _ in range(R)]
        dly = [[1.0] * L for _ in range(R)]
        rad = [[1.0] * L for _ in range(R)]
        mob = [[1.0] * L for _ in range(R)]
        thr = []
        for r, k in enumerate(rigs):
            s = settings[k]
            for v, vt in enumerate(s['Vertices']):
                acc[r][v] = float(vt['Acceleration'])
                dly[r][v] = float(vt['Delay'])
                rad[r][v] = float(vt['Radius'])
                mob[r][v] = float(vt['Mobility'])
            # 填充粒子：半径 1 的普通粒子（不影响真实粒子）
            thr.append(MOVEMENT_THRESHOLD * float(s['Normalization']['Position']['Maximum']))
        init_py = []
        for r in range(R):
            y, row = 0.0, [0.0]
            for v in range(1, L):
                y = y + rad[r][v]   # initialize()：initialPosition = 上一个 + (0, radius)
                row.append(y)
            init_py.append(row)
        # ---- 输出
        o_g, o_flat, o_par, o_typ, o_sgn = [], [], [], [], []
        rnd_items: list = []
        for r, k in enumerate(rigs):
            s = settings[k]
            n = len(s['Vertices'])
            for j, o in enumerate(s['Output']):
                v = int(o['VertexIndex'])
                if v < 1 or v >= n:
                    continue  # Framework：particleIndex 越界的输出跳过（不更新 rig 输出、不写参数）
                loc = len(o_g)
                o_g.append(gbase[k] + j)
                o_flat.append(r * L + v)
                o_par.append(v >= 2)
                o_typ.append(_TYPES[o['Type']])
                o_sgn.append(-1.0 if o['Reflect'] else 1.0)
                pid = o['Destination']['Id']
                if pid not in idx:
                    continue
                p = idx[pid]
                rnd_items.append((gbase[k] + j, loc, q_of[p], self._scale(o), self._pmin[p], self._pmax[p],
                                  o['Weight'] / MAXIMUM_WEIGHT))
        t = self._t
        d = self.device
        g_out = {}
        if o_g:
            g_out = dict(g_o1=_Index(o_flat, R * L, d), g_o0=_Index([i - 1 for i in o_flat], R * L, d),
                         g_o2=_Index([i - 2 if p else i - 1 for i, p in zip(o_flat, o_par)], R * L, d))
        return _Level(
            g_in=_Index(in_q, self.Q, d), g_slot=_Index(slot_flat, NI + 1, d, ignore=(NI,)), **g_out,
            rigs=rigs, L=L, K=K,
            in_q=t(in_q, torch.long), in_slot=t(slot_flat, torch.long), in_type=t(typ),
            n_min=t([c['min'] for c in in_c] + [0.0]), n_max=t([c['max'] for c in in_c] + [0.0]),
            n_mid=t([c['mid'] for c in in_c] + [0.0]),
            r_pos=t([c['r_pos'] for c in in_c] + [0.0]), r_neg=t([c['r_neg'] for c in in_c] + [0.0]),
            m_norm=t([c['m'] for c in in_c] + [0.0]),
            has_pos=t([c['has_pos'] for c in in_c] + [False], torch.bool),
            has_neg=t([c['has_neg'] for c in in_c] + [False], torch.bool),
            in_sw=t([c['sw'] for c in in_c] + [(0.0, 0.0)]),
            in_const=t(const),
            acc=t(acc), delay=t(dly), radius=t(rad), mobility=t(mob), thr=t(thr), init_py=t(init_py),
            out_gidx=t(o_g, torch.long), out_flat=t(o_flat, torch.long), out_parent=t(o_par, torch.bool),
            out_type=t(o_typ, torch.long), out_sign=t(o_sgn), rnd_items=rnd_items, out_vertex_ok=len(o_g) > 0,
        )

    # ------------------------------------------------------------ 公共接口

    def initial_state(self, B: int, resolved: bool = True) -> PhysicsState:
        """CubismPhysics.create() 之后的状态（runtime 的 resetPhysics()；不做 stabilization）。

        resolved：Framework 在第一次 evaluate 时才解析输出的目标参数索引（destinationParameterIndex），
        它挂在 rig 上、不在 runtime 的快照里，resetPhysics() 不会把它恢复成未解析。runtime 的实例会被多个图层复用，
        所以缺省 True（与已经求值过的实例一致）；False = 全新的 CubismPhysics：在第一次子步之前 interpolate 不写参数。
        只在某一帧没有任何子步时有区别（dt < 1/Fps 的第一帧）；dt = 1/60、Fps = 60 时两者完全相同。
        """
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=self.dtype)
        px, py, vx, vy, gx, gy = [], [], [], [], [], []
        for lv in self.levels:
            R, L = len(lv.rigs), lv.L
            px.append(z(B, R, L))
            py.append(lv.init_py.expand(B, R, L).clone())
            vx.append(z(B, R, L))
            vy.append(z(B, R, L))
            gx.append(z(B, R))
            gy.append(torch.ones(B, R, device=self.device, dtype=self.dtype))  # lastGravity = (0, −1)·(1, −1) = (0, 1)
        return PhysicsState(tuple(px), tuple(py), tuple(vx), tuple(vy), tuple(gx), tuple(gy),
                            z(B, self.NO), z(B, self.NO), None, 0.0, resolved)

    def step(self, state: PhysicsState, params: torch.Tensor, dt: float = 1.0 / 60.0):
        """CubismPhysics.evaluate(model, dt)。params：[B, P] 物理之前的参数；返回 (新状态, 物理之后的参数 [B, P])。"""
        if not (dt > 0.0):
            return state, params
        io = params.dtype
        ws, alpha, remain, resolved, phys_dt = self._advance(state.remain, state.resolved, dt)
        wt, at = self._sched(ws, alpha)
        fn = self._frame_c or self._frame
        res = fn(params.to(self.dtype), state.in_cache, state.cur, state.prev,
                 list(state.px), list(state.py), list(state.vx), list(state.vy), list(state.gx), list(state.gy),
                 wt, at, len(ws), resolved, phys_dt)
        out, in_cache, cur, prev, px, py, vx, vy, gx, gy = res
        st = PhysicsState(tuple(px), tuple(py), tuple(vx), tuple(vy), tuple(gx), tuple(gy), cur, prev, in_cache, remain, resolved)
        return st, out.to(io)

    def compile(self, **kw) -> 'PhysicsRig':
        """用 torch.compile 编译单帧函数（前向与反向都融合成少量 kernel）。kw 传给 torch.compile。"""
        kw.setdefault('dynamic', False)
        # 每种 (B, 是否求导, in_cache 是否为 None, 子步数, resolved) 各编译一份；dynamo 缺省最多 8 份，超出后静默退回 eager
        # （慢约 20 倍，数值也变成 eager 的）。拟合器常同时用 B = 1 和 B = 图层数，很容易超，这里放宽（A8）。
        import torch._dynamo.config as dcfg
        for name in ('recompile_limit', 'cache_size_limit'):
            if hasattr(dcfg, name) and getattr(dcfg, name) < 64:
                setattr(dcfg, name, 64)
        self._frame_c = torch.compile(self._frame, **kw)
        return self

    def _advance(self, remain: float, resolved: bool, dt: float):
        """一帧的时间调度（只依赖 dt 与 remain，整批相同，用 Python double 算，与 JS 逐位一致）。
        返回 (各子步的 inputWeight, α, 新 remain, 新 resolved, physicsDeltaTime)。"""
        remain = remain + dt
        if remain > MAX_DELTA_TIME:
            remain = 0.0
        phys_dt = 1.0 / self.fps if self.fps > 0.0 else dt
        ws = []
        while remain >= phys_dt:
            ws.append(phys_dt / remain)   # inputWeight
            remain -= phys_dt
        alpha = remain / phys_dt
        resolved = resolved or len(ws) > 0
        return ws, alpha, remain, resolved, phys_dt

    def _sched(self, ws, alpha):
        key = (tuple(ws), alpha)
        v = self._sched_cache.get(key)
        if v is None:
            v = (self._t(ws if ws else [0.0]), self._t(alpha))
            if len(self._sched_cache) < 1024:
                self._sched_cache[key] = v
        return v

    def _frame(self, params, in_cache, cur, prev, px, py, vx, vy, gx, gy, wt, at, n_sub: int, resolved: bool, phys_dt: float):
        """evaluate() 的一帧（纯函数）：n_sub 个子步 + interpolate。wt：[max(n_sub,1)] 各子步的 inputWeight；at：α。"""
        params = self._r(params)
        pq = self._g_q(params)
        if in_cache is None:
            in_cache = pq   # _parameterInputCaches 为 null：用当前参数初始化
        delays = self._delays_fixed or [self._delays(lv, phys_dt) for lv in self.levels]
        for k in range(n_sub):
            prev = cur
            w = wt[k]
            C = self._r(in_cache * (1.0 - w) + pq * w)
            in_cache = C
            vals = []
            for gids, rounds in self.stages:
                sv = []
                for li in gids:   # 同一层的组都从同一份 C 读输入
                    lv = self.levels[li]
                    tx, ty, ang = self._inputs(lv, C)
                    (px[li], py[li], vx[li], vy[li], gx[li], gy[li]), ov = self._update_particles(
                        lv, tx, ty, ang, px[li], py[li], vx[li], vy[li], gx[li], gy[li], delays[li])
                    sv.append(ov)
                C = self._apply_rounds(rounds, C, torch.cat(sv, 1))
                vals += sv
            cur = self._scatter_outputs(cur, vals)
        out = params
        if resolved:
            out = self._interpolate(params, prev, cur, at)
        return out, in_cache, cur, prev, px, py, vx, vy, gx, gy

    def stabilize(self, state: PhysicsState, params: torch.Tensor):
        """CubismPhysics.stabilization(model)：用当前参数把各条链置于静止平衡态。返回 (新状态, 写入物理输出后的参数)。"""
        io = params.dtype
        params = self._r(params.to(self.dtype))
        pq = self._g_q(params)
        C = pq
        px, py, vx, vy, gx, gy = map(list, (state.px, state.py, state.vx, state.vy, state.gx, state.gy))
        vals = []
        for gids, rounds in self.stages:
            sv = []
            for li in gids:
                lv = self.levels[li]
                tx, ty, ang = self._inputs(lv, C)
                (px[li], py[li], vx[li], vy[li], gx[li], gy[li]), ov = self._stab_particles(
                    lv, tx, ty, ang, px[li], py[li], vx[li], vy[li])
                sv.append(ov)
            C = self._apply_rounds(rounds, C, torch.cat(sv, 1))
            vals += sv
        cur = self._scatter_outputs(state.cur, vals)
        out = params.index_copy(1, self.q_index, C)
        st = PhysicsState(tuple(px), tuple(py), tuple(vx), tuple(vy), tuple(gx), tuple(gy), cur, cur, pq, state.remain, True)
        return st, out.to(io)

    def rollout(self, params_seq: torch.Tensor, state: PhysicsState | None = None, dt: float = 1.0 / 60.0,
                stabilize_first: bool = False, checkpoint: bool = False):
        """逐帧推进整段序列。params_seq：[T, B, P]。返回 (最终状态, [T, B, P] 物理之后的参数)。可整段反向传播（BPTT）。

        checkpoint：逐帧做激活检查点（反向时重算该帧前向），显存只剩每帧的状态，代价约多一次前向。
        auto_fused / auto_graph（缺省都关）为 True 且 checkpoint=False 时改走 fused_rollout / graphed_rollout（A8，同语义）。
        """
        T, B, _ = params_seq.shape
        st = state if state is not None else self.initial_state(B)
        if stabilize_first:
            st, _ = self.stabilize(st, params_seq[0])
        if self.auto_fused and not checkpoint:
            return self.fused_rollout(params_seq, st, dt)
        if self.auto_graph and not checkpoint:
            return self.graphed_rollout(params_seq, st, dt)
        outs = []
        for t in range(T):
            if checkpoint and torch.is_grad_enabled():
                st, o = self._step_ckpt(st, params_seq[t], dt)
            else:
                st, o = self.step(st, params_seq[t], dt)
            outs.append(o)
        return st, torch.stack(outs)

    def _step_ckpt(self, st: PhysicsState, params, dt):
        from torch.utils.checkpoint import checkpoint as ckpt
        n = len(self.levels)
        flat = [*st.px, *st.py, *st.vx, *st.vy, *st.gx, *st.gy, st.cur, st.prev]
        has_ic = st.in_cache is not None
        if has_ic:
            flat.append(st.in_cache)
        meta = {}

        def f(p, *ts):
            s = PhysicsState(*(tuple(ts[i * n:(i + 1) * n]) for i in range(6)), ts[6 * n], ts[6 * n + 1],
                             ts[6 * n + 2] if has_ic else None, st.remain, st.resolved)
            s2, o = self.step(s, p, dt)
            meta['remain'], meta['resolved'] = s2.remain, s2.resolved
            return (o, *s2.px, *s2.py, *s2.vx, *s2.vy, *s2.gx, *s2.gy, s2.cur, s2.prev, s2.in_cache)

        r = ckpt(f, params, *flat, use_reentrant=False)
        o, ts = r[0], r[1:]
        s2 = PhysicsState(*(tuple(ts[i * n:(i + 1) * n]) for i in range(6)), ts[6 * n], ts[6 * n + 1], ts[6 * n + 2],
                          meta['remain'], meta['resolved'])
        return s2, o

    # ------------------------------------------------------------ 可选快速路径：CUDA graph（A8）

    def graphed_rollout(self, params_seq: torch.Tensor, state: PhysicsState | None = None, dt: float = 1.0 / 60.0):
        """与 rollout(params_seq, state, dt) 同语义、结果逐位相同，但把整段 T 帧的前向（求导时连同反向）捕获成 CUDA graph 重放。

        - 逐帧调用的仍是 _frame（compile() 之后是编译版），只是 T 帧的全部内核一次发射：没有逐帧的 Python / dynamo 开销，
          前向与反向都没有 CPU 同步。所以前向结果与同一 rig 的 rollout 逐位相同（compile 与否各自对应）；
          梯度也逐位相同（gather 的反向是确定性的，见 _Index），同一输入跑两次逐位相同。
        - 图按 (T, B, 每帧的子步调度, 哪些输入求导) 缓存；第一次遇到某个键要预热 + 捕获（T=24 约 0.5～1 秒，
          另外未编译过的子步数还要先编译），之后每次调用只有约 60 次内核发射。
        - 求导时，每个图实例的中间量在它自己的静态显存里：同一个键上一次前向的反向还没跑，又做前向（例如拟合器把 13 个
          24 帧窗口都前向完再一起反向），会自动再捕获一个实例，最多 graph_max_instances 个（所有键合计）；
          超出且没有空闲实例时退回普通 rollout（结果相同，只是慢）。同一段的反向只能跑一次之后、该实例被下一次前向复用前
          再跑（retain_graph 后在复用之后再反向会报错，不会给出错误的梯度）。
        - state 为 None 时用 initial_state(B)；state.in_cache 为 None（第一帧）时在图外用第一帧参数初始化，与 _frame 相同。
        - 输出与新状态都是新张量（从图的静态缓冲复制出来），可以放心保留；输出 dtype 同 params_seq。
        - 非 CUDA 设备、dt ≤ 0、T = 0 时直接调用普通 rollout。
        """
        T, B, P = params_seq.shape
        st = state if state is not None else self.initial_state(B)
        if T == 0 or not (dt > 0.0) or self.device.type != 'cuda' or not params_seq.is_cuda:
            return self._rollout_plain(params_seq, st, dt)
        io = params_seq.dtype
        x = params_seq.to(self.dtype)
        sched, remain, resolved, phys_dt = self._schedule_T(st, T, dt)
        ic = st.in_cache
        if ic is None:
            ic = self._g_q(self._r(x[0]))   # 与 _frame 里 in_cache 为 None 时相同
        ins = [x, *st.px, *st.py, *st.vx, *st.vy, *st.gx, *st.gy, st.cur, st.prev, ic]
        grad = torch.is_grad_enabled() and any(t.requires_grad for t in ins)
        mask = tuple(t.requires_grad for t in ins) if grad else None
        key = (T, B, x.dtype, tuple(sched), phys_dt, grad, mask, self._frame_c is not None)
        g = self._graph_get(key, ins)
        if g is None:
            self.graph_stats['fallbacks'] += 1
            return self._rollout_plain(params_seq, st, dt)
        self.graph_stats['replays'] += 1
        res = g.run(ins)
        n = len(self.levels)
        ts = res[1:]
        st2 = PhysicsState(*(tuple(ts[i * n:(i + 1) * n]) for i in range(6)), ts[6 * n], ts[6 * n + 1], ts[6 * n + 2],
                           remain, resolved)
        return st2, res[0].to(io)

    def _schedule_T(self, st: PhysicsState, T: int, dt: float):
        """T 帧的调度：[(各子步 inputWeight, α, resolved)]、最终 remain / resolved、physicsDeltaTime。"""
        remain, resolved = st.remain, st.resolved
        sched = []
        phys_dt = 1.0 / self.fps if self.fps > 0.0 else dt
        for _ in range(T):
            ws, alpha, remain, resolved, phys_dt = self._advance(remain, resolved, dt)
            sched.append((tuple(ws), alpha, resolved))
        return sched, remain, resolved, phys_dt

    def fused_rollout(self, params_seq: torch.Tensor, state: PhysicsState | None = None, dt: float = 1.0 / 60.0):
        """与 rollout(params_seq, state, dt) 同语义，用 Triton 融合核（l2d/_kernels_physics.py）：整段 T 帧前向一个核、
        反向一个核，每个批元素一个 CTA，链内逐粒子顺序、链间并行。

        - 前向与 eager（未 compile）的 rollout / step 逐位相同（同样的运算顺序、IEEE 除法 / 开方、同一套 sin / cos / atan2、
          关闭 FMA 合并）。注意 compile() 之后的原路径因为 FMA 合并与 eager 本来就差 1e-16 级（混沌放大后见 physics.md §5），
          所以 fused 与 compile 版不是逐位相同，与 eager 版逐位相同；
        - 反向是手写伴随，与 autograd 的公式一致，只有浮点求和顺序不同（相对误差 1e-13 级，见 test_a8_fused.py）；
          没有原子加，同一输入两次逐位相同；对输入参数与初始状态（若它们 requires_grad）都求导；
        - 不依赖 compile()，没有编译 / 捕获的等待（Triton 核第一次调用时编译一次，约 10～20 秒，可被 Triton 缓存）；
        - 只支持 CUDA、float64 rig、split_by_length=False；其他情况、dt ≤ 0、T = 0 时直接调用普通 rollout；
        - state 为 None 时用 initial_state(B)；state.in_cache 为 None 时在核外用第一帧参数初始化（与 _frame 相同）。
        显存：求导时每帧保存一份状态快照（B × 约 47 KB / 帧），反向核另需每个批元素约 120 KB 的工作区。
        """
        T, B, P = params_seq.shape
        st = state if state is not None else self.initial_state(B)
        if (T == 0 or not (dt > 0.0) or self.device.type != 'cuda' or not params_seq.is_cuda or self.dtype != torch.float64
                or self.split_by_length):
            return self._rollout_plain(params_seq, st, dt)
        from . import _kernels_physics as K
        if self._fused_plan is None:
            self._fused_plan = K.Plan(self)
        plan = self._fused_plan
        io = params_seq.dtype
        x = params_seq.to(self.dtype)
        sched, remain, resolved, phys_dt = self._schedule_T(st, T, dt)
        ic = st.in_cache
        if ic is None:
            ic = self._g_q(self._r(x[0]))   # 与 _frame 里 in_cache 为 None 时相同
        S0 = plan.pack(st, ic)
        out, S = K.FusedRollout.apply(plan, plan.schedule(sched), phys_dt, x, S0)
        (px, py, vx, vy, gx, gy), cur, prev, ic2 = plan.unpack(S)
        return PhysicsState(px, py, vx, vy, gx, gy, cur, prev, ic2, remain, resolved), out.to(io)

    def _rollout_plain(self, params_seq, st, dt):
        outs = []
        for t in range(params_seq.shape[0]):
            st, o = self.step(st, params_seq[t], dt)
            outs.append(o)
        return st, torch.stack(outs) if outs else params_seq.clone()

    def graph_clear(self):
        """丢掉所有缓存的图实例（释放它们的私有显存池；之后可 torch.cuda.empty_cache()）。"""
        self._graphs.clear()

    def _graph_get(self, key, ins):
        lst = self._graphs.pop(key, None)
        if lst is None:
            lst = []
        self._graphs[key] = lst   # 移到末尾（最近使用）
        for g in lst:
            if g.free():
                return g
        total = sum(len(v) for v in self._graphs.values())
        while total >= self.graph_max_instances:
            # 从最久未用的键里丢一个空闲实例
            victim = None
            for k, v in self._graphs.items():
                for i, gg in enumerate(v):
                    if gg.free():
                        victim = (k, i)
                        break
                if victim:
                    break
            if victim is None:
                return None
            k, i = victim
            self._graphs[k].pop(i)
            if not self._graphs[k] and k != key:
                del self._graphs[k]
            total -= 1
        g = _PhysGraph(self, key, ins, warmup=0 if lst else 2)
        self.graph_stats['captures'] += 1
        lst.append(g)
        return g

    # ------------------------------------------------------------ 内部

    def _r(self, x):
        """Float32Array 存储：float64 模式下前向舍入到 float32，反向直通。"""
        if not self.emulate_f32:
            return x
        return x + (x.float().double() - x).detach()

    def _inputs(self, lv: _Level, C):
        """各 setting 的 totalTranslation（已旋转）与 totalAngle。[B, R]。"""
        B = C.shape[0]
        R, K = len(lv.rigs), lv.K
        v = lv.g_in(C)
        v = torch.cat([v, v.new_zeros(B, 1)], 1)
        # normalizeParameterValue
        v = torch.where(lv.n_max < v, lv.n_max, v)
        v = torch.where(lv.n_min > v, lv.n_min, v)
        pv = v - lv.n_mid
        rp = torch.where(lv.has_pos, pv * lv.r_pos + lv.m_norm, torch.zeros_like(pv))
        rn = torch.where(lv.has_neg, pv * lv.r_neg + lv.m_norm, torch.zeros_like(pv))
        # pv = 0（输入恰好在范围中点，常见于缺省值）：Framework 走 sign = 0 分支，结果 = middleNorm。
        # 函数在这里连续、左右斜率为 r_neg / r_pos；前向仍给 middleNorm（pv·r + m 在 pv = 0 时逐位等于 m），
        # 梯度取两侧斜率的平均（与中心差分一致），而不是常量分支的 0。
        r_mid = torch.where(lv.has_pos, lv.r_pos, 0.0) + torch.where(lv.has_neg, lv.r_neg, 0.0)
        r_mid = r_mid / torch.where(lv.has_pos & lv.has_neg, 2.0, 1.0)
        r0 = pv * r_mid + lv.m_norm
        res = torch.where(pv > 0, rp, torch.where(pv < 0, rn, r0))
        c = (res * lv.in_sw[:, 0]) * lv.in_sw[:, 1]          # isInverted ? result : −result，再 × weight
        c = lv.g_slot(c).view(B, R, K, 1) * lv.in_type     # [B, R, K, 3]
        tot = c[:, :, 0]
        for k in range(1, K):                                  # 与 Framework 相同的顺序逐个累加
            tot = tot + c[:, :, k]
        if self.has_const:
            tot = tot + lv.in_const
        tx, ty, ang = tot.unbind(-1)
        ra = (-ang / 180.0) * math.pi
        cr, sr = torch.cos(ra), torch.sin(ra)
        tx = tx * cr - ty * sr
        ty = tx * sr + ty * cr   # Framework 原样：这里用的是已经旋转过的 x
        return tx, ty, ang

    def _gravity(self, ang):
        tr = (ang / 180.0) * math.pi
        return normalize(torch.sin(tr), torch.cos(tr))

    def _delays(self, lv: _Level, dt: float):
        """delay = strand[i].delay · deltaTimeSeconds · 30；返回 (delay, delay ≠ 0, 用作除数的 delay)。"""
        dl = (lv.delay * dt) * 30.0
        nz = dl != 0.0
        return dl, nz, torch.where(nz, dl, torch.ones_like(dl))

    def _update_particles(self, lv: _Level, tx, ty, ang, px, py, vx, vy, lgx, lgy, delays):
        """updateParticles（cubismphysics.ts）。tx/ty/ang：[B, R]；px…：[B, R, L]。"""
        cgx, cgy = self._gravity(ang)
        wx, wy = self.opt_wind
        rad = direction_to_radian(lgx, lgy, cgx, cgy) / AIR_RESISTANCE
        cr, sr = torch.cos(rad), torch.sin(rad)
        dl, dnz, dsafe = delays
        PX, PY, VX, VY = px.unbind(2), py.unbind(2), vx.unbind(2), vy.unbind(2)
        nx_, ny_, nvx, nvy = [tx], [ty], [VX[0]], [VY[0]]
        thr = lv.thr
        for i in range(1, lv.L):
            a, d = lv.acc[:, i], dl[:, i]
            fx = cgx * a + wx
            fy = cgy * a + wy
            lpx, lpy = PX[i], PY[i]
            ox, oy = nx_[i - 1], ny_[i - 1]
            dx = lpx - ox
            dy = lpy - oy
            dx = cr * dx - dy * sr
            dy = sr * dx + dy * cr   # Framework 原样：用已旋转的 x
            qx = ox + dx
            qy = oy + dy
            qx = (qx + VX[i] * d) + (fx * d) * d
            qy = (qy + VY[i] * d) + (fy * d) * d
            ex, ey = normalize(qx - ox, qy - oy)
            r = lv.radius[:, i]
            qx = ox + ex * r
            qy = oy + ey * r
            qx = torch.where(torch.abs(qx) < thr, torch.zeros_like(qx), qx)
            nz, ds, mob = dnz[:, i], dsafe[:, i], lv.mobility[:, i]
            nvx.append(torch.where(nz, ((qx - lpx) / ds) * mob, VX[i]))
            nvy.append(torch.where(nz, ((qy - lpy) / ds) * mob, VY[i]))
            nx_.append(qx)
            ny_.append(qy)
        npx, npy = torch.stack(nx_, 2), torch.stack(ny_, 2)
        new = (npx, npy, torch.stack(nvx, 2), torch.stack(nvy, 2), cgx, cgy)
        return new, self._outputs(lv, npx, npy)

    def _stab_particles(self, lv: _Level, tx, ty, ang, px, py, vx, vy):
        """updateParticlesForStabilization。"""
        cgx, cgy = self._gravity(ang)
        wx, wy = self.opt_wind
        nx_, ny_ = [tx], [ty]
        VX, VY = vx.unbind(2), vy.unbind(2)
        nvx, nvy = [VX[0]], [VY[0]]
        for i in range(1, lv.L):
            a = lv.acc[:, i]
            fx, fy = normalize(cgx * a + wx, cgy * a + wy)
            r = lv.radius[:, i]
            qx = nx_[i - 1] + fx * r
            qy = ny_[i - 1] + fy * r
            qx = torch.where(torch.abs(qx) < lv.thr, torch.zeros_like(qx), qx)
            nx_.append(qx)
            ny_.append(qy)
            nvx.append(torch.zeros_like(qx))
            nvy.append(torch.zeros_like(qy))
        npx, npy = torch.stack(nx_, 2), torch.stack(ny_, 2)
        new = (npx, npy, torch.stack(nvx, 2), torch.stack(nvy, 2), cgx, cgy)
        return new, self._outputs(lv, npx, npy)

    def _outputs(self, lv: _Level, npx, npy):
        """getOutputTranslationX / Y / getOutputAngle：本层合法输出的原始值 [B, NOv]。"""
        if not lv.out_vertex_ok:
            return npx.new_zeros(npx.shape[0], 0)
        B = npx.shape[0]
        fx, fy = npx.reshape(B, -1), npy.reshape(B, -1)
        x1, y1 = lv.g_o1(fx), lv.g_o1(fy)
        x0, y0 = lv.g_o0(fx), lv.g_o0(fy)
        tx, ty = x1 - x0, y1 - y0
        gx_, gy_ = -self.opt_gravity[0], -self.opt_gravity[1]
        pxp = torch.where(lv.out_parent, x0 - lv.g_o2(fx), torch.full_like(x0, gx_))   # g_o2：v < 2 时这一项不用（换成 −gravity）
        pyp = torch.where(lv.out_parent, y0 - lv.g_o2(fy), torch.full_like(y0, gy_))
        angle = direction_to_radian(pxp, pyp, tx, ty)
        t = lv.out_type
        val = torch.where(t == 0, tx, torch.where(t == 1, ty, angle))
        return val * lv.out_sign

    def _apply_rounds(self, rounds, target, vals):
        """updateOutputParameterValue：× scale、夹到目标参数范围、按 weight 与目标当前值混合，写回（float32）。"""
        for rd in rounds:
            v = rd['g_src'](vals) * rd['scale']
            v = torch.where(v < rd['vmin'], rd['vmin'], torch.where(v > rd['vmax'], rd['vmax'], v))
            base = rd['g_dst'](target)
            v = torch.where(rd['full'], v, base * (1.0 - rd['w']) + v * rd['w'])
            target = target.index_copy(1, rd['dst'], self._r(v))
        return target

    def _scatter_outputs(self, cur, vals):
        """本子步各层的合法输出写进 _currentRigOutputs（越界顶点的输出保持原值）。"""
        v = [x for x, lv in zip(vals, self.levels) if lv.out_vertex_ok]
        if not v:
            return cur
        return cur.index_copy(1, self._out_gidx_all, torch.cat(v, 1))

    def _interpolate(self, params, prev, cur, alpha):
        """interpolate(model, weight)：prev·(1−α) + cur·α 写入参数（在物理输出参数的紧凑空间里按轮写，最后写回）。"""
        vals = prev * (1.0 - alpha) + cur * alpha
        tgt = self._apply_rounds(self.interp_rounds, self._g_outp(params), vals)
        return params.index_copy(1, self.output_param_index, tgt)


# ---------------------------------------------------------------- CUDA graph 快速路径（A8）

class _PhysGraph:
    """graphed_rollout 的一个图实例：静态输入 → T 帧前向（求导时还有一张反向图）→ 静态输出。

    输入 / 输出的排列：[params_seq, px×n, py×n, vx×n, vy×n, gx×n, gy×n, cur, prev, in_cache]（输出第 0 个是 [T, B, P] 的结果）。"""

    def __init__(self, rig: PhysicsRig, key, ins, warmup: int = 2):
        T, B, dtype, sched, phys_dt, grad, mask, _ = key
        self.grad = grad
        self.gen = 0            # 每次前向重放 +1（反向时核对，防止用被覆盖的中间量）
        self.bwd_gen = -1       # 最近一次反向对应的 gen（一次前向只能反向一次）
        self.busy = False       # 求导实例：前向之后、反向之前为 True
        self._node = None       # 上一次前向的 autograd 节点（弱引用；节点没了说明这次前向的图已被丢弃）
        # 调度张量（常量，图里按地址引用，这里持有引用）
        self.sched_t = [rig._sched(list(ws), alpha) for ws, alpha, _ in sched]
        n_frames = [(len(ws), resolved) for ws, _, resolved in sched]
        fn = rig._frame_c or rig._frame
        n = len(rig.levels)

        def f(*a):
            x, s = a[0], a[1:]
            px, py, vx, vy, gx, gy = (list(s[i * n:(i + 1) * n]) for i in range(6))
            cur, prev, ic = s[6 * n], s[6 * n + 1], s[6 * n + 2]
            outs = []
            for t, ((ns, resolved), (wt, at)) in enumerate(zip(n_frames, self.sched_t)):
                o, ic, cur, prev, px, py, vx, vy, gx, gy = fn(x[t], ic, cur, prev, px, py, vx, vy, gx, gy, wt, at, ns, resolved, phys_dt)
                outs.append(o)
            res = [torch.stack(outs), *px, *py, *vx, *vy, *gx, *gy, cur, prev, ic]
            # 没有子步的帧会把输入原样传出：与静态输入共用存储的输出要复制一份（否则重放时互相覆盖）
            ptrs = {t.untyped_storage().data_ptr() for t in a}
            return [r.clone() if r.untyped_storage().data_ptr() in ptrs else r for r in res]

        self.s_in = [t.detach().clone() for t in ins]
        if grad:
            for t, m in zip(self.s_in, mask):
                t.requires_grad_(m)
        self.in_rg = [i for i, t in enumerate(self.s_in) if t.requires_grad]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        cs = torch.cuda.current_stream()
        if warmup:
            s = torch.cuda.Stream()
            s.wait_stream(cs)
            with torch.cuda.stream(s), ctx:
                for _ in range(warmup):
                    o = f(*self.s_in)
                    if grad:
                        oo = [t for t in o if t.requires_grad]
                        torch.autograd.grad(oo, [self.s_in[i] for i in self.in_rg], [torch.ones_like(t) for t in oo],
                                            allow_unused=True)
                    del o
            cs.wait_stream(s)
        self.pool = torch.cuda.graph_pool_handle()
        self.fwd = torch.cuda.CUDAGraph()
        with ctx, torch.cuda.graph(self.fwd, pool=self.pool):
            self.s_out = f(*self.s_in)
        self.out_rg = [i for i, t in enumerate(self.s_out) if t.requires_grad] if grad else []
        if grad:
            self.s_gout = [torch.zeros_like(self.s_out[i]) for i in self.out_rg]
            self.bwd = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.bwd, pool=self.pool):
                # 与 make_graphed_callables 相同，不保留图：保存的中间量在捕获后归还给本实例的私有池，
                # 前向重放时重新写入（编译版的反向还会就地复用这些缓冲，所以一次前向只能反向一次）
                gi = torch.autograd.grad([self.s_out[i] for i in self.out_rg], [self.s_in[i] for i in self.in_rg],
                                         self.s_gout, allow_unused=True)
            self.s_gin = list(gi)

    def free(self) -> bool:
        if not self.busy:
            return True
        return self._node is not None and self._node() is None

    def run(self, ins):
        if not self.grad:
            torch._foreach_copy_(self.s_in, [t.detach() for t in ins])
            self.fwd.replay()
            self.gen += 1
            return _clone_all(self.s_out)
        outs = _PhysGraphFn.apply(self, *ins)
        node = next((o.grad_fn for o in outs if o.grad_fn is not None), None)
        if node is not None:
            import weakref
            self._node = weakref.ref(node)
            self.busy = True
        return outs


def _clone_all(ts):
    out = [torch.empty_like(t) for t in ts]
    torch._foreach_copy_(out, ts)
    return out


class _PhysGraphFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: _PhysGraph, *ins):
        torch._foreach_copy_(g.s_in, list(ins))
        g.fwd.replay()
        g.gen += 1
        ctx.g, ctx.gen = g, g.gen
        outs = _clone_all(g.s_out)
        nd = [o for i, o in enumerate(outs) if i not in g.out_rg]
        if nd:
            ctx.mark_non_differentiable(*nd)
        return tuple(outs)

    @staticmethod
    def backward(ctx, *gouts):
        g = ctx.g
        if g.gen != ctx.gen or g.bwd_gen == ctx.gen:
            raise RuntimeError('graphed_rollout：这一段的中间量已被覆盖（同一图实例又做了前向，或这一段已经反向过一次），'
                               '不能再反向；需要 retain_graph 多次反向时请用 rollout')
        g.bwd_gen = ctx.gen
        src = [gouts[i] if gouts[i] is not None else torch.zeros_like(g.s_gout[k]) for k, i in enumerate(g.out_rg)]
        torch._foreach_copy_(g.s_gout, src)
        g.bwd.replay()
        g.busy = False
        grads = [None] * len(g.s_in)
        live = [(i, t) for i, t in zip(g.in_rg, g.s_gin) if t is not None]
        cl = _clone_all([t for _, t in live])
        for (i, _), c in zip(live, cl):
            grads[i] = c
        return (None, *grads)
