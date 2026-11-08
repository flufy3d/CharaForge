"""整段轨迹的参数化与可微分前向（一个模型图层；多图层由 optimize.Fitter 按 z 序合成）。

未知量（全部是叶张量，按组给 Adam 学习率）：
  inputs [Nn, Din]   面捕输入，每个新画面帧（节点）一个值，节点之间线性插值（区间外保持端点）——同 traj2.inputSeries；
  tf     [Nk, 4]     图层变换关键帧 (x, y, ln scale, rotation°)，关键帧之间线性插值；anchor 固定为参考姿态两眼虹膜中点；
  post   [Nn, Q]     postPhysics add 偏移（姿态输出 6 个 + 物理写入的全部参数，各一个自由度），节点间线性插值；
  consts             LeanInput（规划没固定时）、呼吸周期/相位、blur σ、景深倍率 dofK、颜色滤镜（规划要求时）；resample 因子离散搜索。
规划给定、不拟合的逐帧量：开关斜坡（切换帧另做离散估计）、翻牌 scaleX、clip、不透明度、景深强度曲线。
前向（规范 a～g，与导出的图层逐项同语义）：
  inputs →（插值）逐帧输入 → VTS 映射 + 平滑（vts.py）→ 逐帧 prePhysics set（夹到范围、float32）
  → 预热 warmup 帧（输入取帧 in）→ A2 物理（dt = 1/60，整段 BPTT）→ postPhysics add（夹取）
  → 只在要显示的帧 g（节点）上：A1 变形 → A3 光栅化 → 滤镜 → 合成。
"""
from __future__ import annotations

import math

import numpy as np
import torch

from l2d import composite as C

from .plan import LayerPlan, switch_value
from .render import raster_layer, compose_one
from .vts import INPUT_DIMS, Vts

WARMUP = 90
SHAPE_SWITCHES = ("Clothes2",)  # 改变外形的开关（体型 / Q 版），见 Traj.sw_gain
POSE_INPUTS = ("FaceAngleX", "FaceAngleY", "FaceAngleZ", "BodyX", "BodyY", "BodyZ")  # 小实例姿态先验（mag_pose）管的面捕输入
POSE_OUT = ["ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"]
# 次级物理按发束分组（同 tools/fit/src/model.ts 的 HAIR_GROUPS）
HAIR_GROUPS = [
    ("hFX", ["FrontHairX1", "FrontHairX2", "FrontHairX3", "FrontHairX4"]),
    ("hFY", ["FrontHairY1", "FrontHairY2", "FrontHairY3", "FrontHairY4"]),
    ("hSX", ["SideHairX1", "SideHairX2", "SideHairX3", "SideHairX4"]),
    ("hSY", ["SideHairY1", "SideHairY2", "SideHairY3", "SideHairY4"]),
    ("hLX", ["LateHairX1", "LateHairX2", "LateHairX3", "LateHairX4", "LateHairX5"]),
    ("hLY", ["LateHairY1", "LateHairY2", "LateHairY3", "LateHairY4", "LateHairY5"]),
    ("hLSX", ["LateSideHairX1", "LateSideHairX2", "LateSideHairX3", "LateSideHairX4", "LateSideHairX5"]),
    ("hLSY", ["LateSideHairY1", "LateSideHairY2", "LateSideHairY3", "LateSideHairY4", "LateSideHairY5"]),
    ("hEarL", ["LEar1", "LEar2", "LEar3"]),
    ("hEarR", ["REar1", "REar2", "REar3"]),
    ("hSlvL", ["LSleeveX1", "LSleeveX2", "LSleeveX3"]),
    ("hSlvR", ["RSleeveX1", "RSleeveX2", "RSleeveX3", "RSleeveX4"]),
    ("hChX", ["ChestX1", "ChestX2", "ChestX3"]),
    ("hChY", ["ChestY1", "ChestY2", "ChestY3"]),
    ("hLace", ["LaceRX", "LaceRX2", "LaceRX3", "LaceLX1", "LaceLX2", "LaceLX3"]),
    ("hBow", ["RBowX1", "RBowX2", "RBowX3", "RBowX4", "LBowX1", "LowX2", "LBowX3", "LBowX4", "RBowX5", "RBowX6", "RBowX7", "RBowX8"]),
    ("hPony", ["ShortPonytailX1", "ShortPonytailX2"]),
    ("hRib", ["LRibbonX1", "LRibbonX2", "LRibbonX3", "LRibbonX4", "RRibbonX1", "RRibbonX2", "RRibbonX3", "RRibbonX4"]),
]


def interp_matrix(xs: list[int], frames: list[int]) -> torch.Tensor:
    """线性插值矩阵 [len(frames), len(xs)]（区间外保持端点）。"""
    M = np.zeros((len(frames), len(xs)))
    for r, f in enumerate(frames):
        if f <= xs[0]:
            M[r, 0] = 1
        elif f >= xs[-1]:
            M[r, -1] = 1
        else:
            i = int(np.searchsorted(xs, f, side="right")) - 1
            w = (f - xs[i]) / (xs[i + 1] - xs[i])
            M[r, i], M[r, i + 1] = 1 - w, w
    return torch.tensor(M)


def second_diff(x: torch.Tensor, t: list[int]) -> torch.Tensor:
    """非均匀节点上的「斜率变化」：(x_{i+1}−x_i)/h_i − (x_i−x_{i−1})/h_{i−1}，再乘平均间距（量纲同 x）。x [N, D] → [N−2, D]。"""
    if x.shape[0] < 3:
        return x[:0]
    tt = torch.tensor(t, dtype=x.dtype, device=x.device)
    h = (tt[1:] - tt[:-1])[:, None]
    s = (x[1:] - x[:-1]) / h
    hm = 0.5 * (h[1:] + h[:-1])
    return (s[1:] - s[:-1]) * hm


class Traj:
    POSE_SD = 8.0  # 小实例姿态先验的尺度（面捕输入单位 / 姿态参数单位，都是「度」量级）

    def __init__(self, rig, plan: LayerPlan, vts: Vts, smooth_c: float = 0.15, fit_inputs: list[str] | None = None, device="cuda"):
        self.rig, self.plan, self.vts, self.c = rig, plan, vts, smooth_c
        self.dev = torch.device(device)
        self.frames = plan.frames
        self.Tf = len(self.frames)
        self.start = plan.fin - WARMUP
        self.nodes, self.keys = plan.nodes, plan.keys
        self.inputs_fit = [k for k in (fit_inputs or vts.inputs) if k in vts.inputs]
        dd = dict(device=self.dev, dtype=torch.float64)
        self.Wn = interp_matrix(self.nodes, self.frames).to(**dd)
        self.Wk = interp_matrix(self.keys, self.frames).to(**dd)
        # ---- 未知量 ----
        neu = torch.tensor([vts.neutral.get(k, 0.0) for k in self.inputs_fit], **dd)
        self.in_neutral = neu
        self.in_lo = torch.tensor([INPUT_DIMS[k]["lo"] for k in self.inputs_fit], **dd)
        self.in_hi = torch.tensor([INPUT_DIMS[k]["hi"] for k in self.inputs_fit], **dd)
        self.in_tau = torch.tensor([INPUT_DIMS[k]["tau"] for k in self.inputs_fit], **dd)
        # 叶张量在归一化坐标里（Adam 每组一个学习率）：物理量 = z × 单位
        self.in_unit = self.in_tau
        self.z_in = (neu / self.in_unit)[None].repeat(len(self.nodes), 1).clone().requires_grad_(True)
        self.pose_cols = [i for i, k in enumerate(self.inputs_fit) if k in POSE_INPUTS]
        # 变换先验（粗配准成功时由 register_small 设置）：中心 [Nk, 4]、单位 [4]、逐关键帧权重 [Nk]
        self.tf_center, self.tf_sd, self.tf_w = None, None, None
        self.vmax_k = None  # 逐关键帧区间的速度上限 [Nk−1, 4]（set_camera_ranges），缺省各区间相同
        # 可见框中心（逐关键帧，720p；shots 的 bbox 裁到画面内，没有时用 head 框）：给 lever() 算力臂；全段都没有框时为 None。
        # 不能先用 head 框：腿部特写的 head 框在画面外、就在锚点附近，算出来的力臂反而很短
        import sys as _sys
        from .rig import REPO as _REPO
        _sys.path.insert(0, str(_REPO / "tools/qa"))
        import metric as _metric
        cs = []
        for k in self.keys:
            from .plan import _on_screen
            r = _on_screen(_metric._interp_skip_null(plan.track, k, "bbox")) or _on_screen(_metric._interp_skip_null(plan.track, k, "head"))
            cs.append([r[0] + r[2] / 2, r[1] + r[3] / 2] if r else [float("nan"), float("nan")])
        self.vis_c = torch.tensor(cs, **dd) if any(not math.isnan(c[0]) for c in cs) else None
        self.tf_unit = torch.tensor([1.0, 1.0, 0.01, 0.5], **dd)  # 像素、像素、ln scale、度
        self.z_tf = torch.zeros(len(self.keys), 4, **dd).requires_grad_(True)
        # 体型开关（Clothes2 切到 Q 版等改变外形的开关）连带的尺度变化：每个这样的开关一个 ln s 增益，按该开关自己的斜坡加到变换上
        # （tf_at）。原视频切 Q 版时往往同时放大角色（s38.side：Q 版的头与切换前的头差不多大，而模型的 Q 版头小一圈），
        # 变换关键帧 24 帧一个、线性插值，跟不上切换的几帧；增益随开关一起平移，开关估计换候选时尺度变化跟着走。
        self.sw_gain_idx = [i for i, sw in enumerate(plan.switches) if sw["param"] in SHAPE_SWITCHES and sw["to"] != sw["from"]]
        self.z_swg = torch.zeros(len(self.sw_gain_idx), **dd).requires_grad_(True)
        self._init_sw_gain()
        # postPhysics 偏移：姿态输出（每个参数一个自由度）+ 次级物理分组
        # 姿态输出 + 物理写入的全部参数，各一个自由度（同 fit4 逐帧精修的「全部可见参数 postPhysics add」，这里带时间正则联合求解）
        outs = [rig.param_ids[i] for i in rig.phys.output_params]
        groups = [(p, [p]) for p in POSE_OUT] + [(p, [p]) for p in outs if p not in POSE_OUT]
        self.post_groups = [(g, ps) for g, ps in groups if ps]
        self.post_scale = torch.tensor([float(rig.pmax[rig.idx(g)] - rig.pmin[rig.idx(g)]) / 2 for g, _ in self.post_groups], **dd)
        self.post_unit = 0.05 * self.post_scale
        self.z_post = torch.zeros(len(self.nodes), len(self.post_groups), **dd).requires_grad_(True)
        pid, gid = [], []
        for j, (_, ps) in enumerate(self.post_groups):
            for p in ps:
                pid.append(rig.idx(p))
                gid.append(j)
        self.post_pid = torch.tensor(pid, device=self.dev)
        self.post_gid = torch.tensor(gid, device=self.dev)
        self.pose_post_cols = [j for j, (g, _) in enumerate(self.post_groups) if g in POSE_OUT]
        # 前倾：规划固定（小实例为 0）时不拟合
        self.lean_free = "lean" not in plan.fixed
        self.consts = {
            "lean": torch.tensor(float(plan.fixed.get("lean", 0.0)), **dd).requires_grad_(True),
            "dofK": torch.tensor(1.0, **dd).requires_grad_(True),
            "breathPeriod": torch.tensor(3.5, **dd).requires_grad_(True),
            "breathPhase": torch.tensor(0.0, **dd).requires_grad_(True),
            "blur": torch.tensor(1.0, **dd).requires_grad_(True),
            "brightness": torch.tensor(0.0, **dd).requires_grad_(True),
            "contrast": torch.tensor(1.0, **dd).requires_grad_(True),
            "saturation": torch.tensor(1.0, **dd).requires_grad_(True),
            "multiply": torch.ones(3, **dd).requires_grad_(True),
        }
        self.resample = 1.0
        self.clothes2 = float(plan.fixed.get("clothes2", plan.constants.get("Clothes2", 0.0)))
        self.has_dof = any(v is not None for v in plan.dof)
        # 小实例的先验加权：眼距（shots 眼睛采样的中位数，720p）< 90px 时 small_k = (90 / 眼距)²，最多 16
        ed = float(np.median([np.hypot(*(e[1] - e[0])) for _, e, _ in plan.eye_init])) if plan.eye_init else 200.0
        self.small_k = float(min(16.0, max(1.0, (90.0 / max(ed, 1.0)) ** 2)))
        self.tiny_k = max(0.0, self.small_k - 4.0)  # 姿态与变换先验（mag_pose、mag_tf）的权重，只对眼距 < 45px 的实例有分量
        # 锚点：显示帧 f 的画面 = 帧 g 的求值
        self.anchor_frames = [plan.hold(f) for f, _, _ in plan.anchors]
        self.anchor_xy = torch.tensor(np.stack([e for _, e, _ in plan.anchors]), **dd) if plan.anchors else None
        self.anchor_w = torch.tensor([c for _, _, c in plan.anchors], **dd) if plan.anchors else None
        # anchor：参考姿态（模型缺省参数）下两眼虹膜网格质心的中点（画布像素）
        with torch.no_grad():
            out = rig.deform(rig.tdef[None])
            self.iris0 = rig.iris_centers(out.verts)[0].double()  # [2, 2]
        self.anchorX, self.anchorY = [float(v) for v in self.iris0.mean(0)]
        self.eye_d0 = float((self.iris0[1] - self.iris0[0]).norm())

    @property
    def inputs(self) -> torch.Tensor:
        return self.z_in * self.in_unit

    @property
    def tf(self) -> torch.Tensor:
        return self.z_tf * self.tf_unit

    @property
    def post(self) -> torch.Tensor:
        return self.z_post * self.post_unit

    def eye_dist(self, param: str, values: list[float]) -> list[float]:
        """缺省参数下把 param 设成各值时的画布眼距（虹膜中心距离）。"""
        r = self.rig
        with torch.no_grad():
            P = r.tdef[None].repeat(len(values), 1).clone()
            P[:, r.idx(param)] = torch.tensor(values, device=self.dev, dtype=P.dtype)
            ic = r.iris_centers(r.deform(P).verts).double()
            return (ic[:, 1] - ic[:, 0]).norm(dim=-1).tolist()

    def _init_sw_gain(self) -> None:
        """体型开关尺度增益的初值：让画面上的眼距在切换前后连续（shots 的眼睛锚点在切换前后基本连续）
        = ln(切换前 / 切换后 画布眼距)。本模型 Clothes2 0 → 1：眼距 339 → 299，初值 ln 1.13。之后由光流 / SSIM 阶段拟合。"""
        with torch.no_grad():
            for k, i in enumerate(self.sw_gain_idx):
                sw = self.plan.switches[i]
                d0, d1 = self.eye_dist(sw["param"], [float(sw["from"]), float(sw["to"])])
                self.z_swg[k] = math.log(d0 / d1) / float(self.tf_unit[2])

    def set_body(self, c2: float) -> None:
        """换体型常量 Clothes2（粗配准的体型多假设用）：常量、以它为起点的 Clothes2 开关的 from、开关的尺度增益初值一起改。"""
        old = self.clothes2
        self.clothes2 = float(c2)
        for sw in self.plan.switches:
            if sw["param"] == "Clothes2" and abs(float(sw["from"]) - old) < 1e-6:
                sw["from"] = float(c2)
        self.sw_gain_idx = [i for i, sw in enumerate(self.plan.switches) if sw["param"] in SHAPE_SWITCHES and sw["to"] != sw["from"]]
        with torch.no_grad():
            self.z_swg = torch.zeros(len(self.sw_gain_idx), device=self.dev, dtype=torch.float64).requires_grad_(True)
        self._init_sw_gain()

    VMAX = (30.0, 30.0, 0.02, 0.8)  # 速度上限（每帧）：x, y px、ln s、度
    # 镜头运动效果区间内放宽的分量：pan 只放宽平移，zoom / scale 放宽平移与缩放；旋转都不放宽（这些效果不转画面）。
    # 上一版三种效果一律放宽平移、缩放、旋转（旋转 3°/帧），s13（pan、腿部特写、锚点在画面外约 2500px）的变换因此来回晃，
    # 新画面帧上脚部跳 10～20px（闪烁尖峰 5 帧）
    CAMERA_VMAX = {"pan": (90.0, 90.0, None, None), "zoom": (90.0, 90.0, 0.12, None), "scale": (90.0, 90.0, 0.12, None)}

    def set_camera_ranges(self, ranges: list[tuple[int, int, str]]) -> None:
        """shots 的镜头运动效果（[start, end), kind）：与之重叠的关键帧区间按 CAMERA_VMAX 放宽速度上限。"""
        if not ranges or len(self.keys) < 2:
            return
        rows = []
        for a, b in zip(self.keys[:-1], self.keys[1:]):
            r = list(self.VMAX)
            for s0, e, kind in ranges:
                if a < e and b > s0:
                    r = [max(x, y) if y is not None else x for x, y in zip(r, self.CAMERA_VMAX[kind])]
            rows.append(r)
        self.vmax_k = torch.tensor(rows, device=self.dev, dtype=torch.float64)

    def lever(self) -> torch.Tensor:
        """逐关键帧的力臂 [Nk]（720p 像素）：shots 可见框（head 或 bbox）中心到图层锚点（两眼中点）的距离，不求导。
        旋转、缩放对画面的影响 = 角度 / ln s × 力臂；锚点在画面外很远时（腿部特写），零点几度就是几十像素。"""
        if self.vis_c is None:
            return torch.zeros(len(self.keys), device=self.dev, dtype=torch.float64)
        with torch.no_grad():
            return (self.vis_c - self.tf[:, :2]).norm(dim=-1).nan_to_num(0.0)

    @property
    def sw_gain(self) -> torch.Tensor:
        return self.z_swg * self.tf_unit[2]

    def shape_ramp(self, gs: list[int]) -> torch.Tensor:
        """体型开关的斜坡（0 → 1，与 switch_value 同形）[len(gs), 开关数]。"""
        rows = []
        for i in self.sw_gain_idx:
            sw = self.plan.switches[i]
            rows.append([(switch_value(sw, g) - sw["from"]) / (sw["to"] - sw["from"]) for g in gs])
        return torch.tensor(rows, device=self.dev, dtype=torch.float64).T.reshape(len(gs), len(self.sw_gain_idx))

    def set_tf(self, tf: torch.Tensor) -> None:
        with torch.no_grad():
            self.z_tf.copy_(tf / self.tf_unit)

    # ------------------------------------------------------------------ 参数组
    def param_groups(self, lr: dict) -> list[dict]:
        g = []
        add = lambda ps, key: g.append({"params": ps, "lr": lr[key], "name": key}) if lr.get(key, 0) > 0 else None
        add([self.z_tf] + ([self.z_swg] if self.sw_gain_idx else []), "tf")
        add([self.z_in], "inputs")
        add([self.z_post], "post")
        if self.lean_free:
            add([self.consts["lean"]], "lean")
        add([self.consts["breathPeriod"], self.consts["breathPhase"]], "breath")
        add([self.consts["blur"]] + ([self.consts["dofK"]] if self.has_dof else []), "blur")
        if self.plan.color_filter:
            add([self.consts[k] for k in ("brightness", "contrast", "saturation", "multiply")], "color")
        return g

    # ------------------------------------------------------------------ 前向
    def input_series(self) -> dict[str, torch.Tensor]:
        X = self.Wn @ self.inputs  # [Tf, Din]
        return {k: X[:, i] for i, k in enumerate(self.inputs_fit)}

    def track_values(self) -> dict[str, torch.Tensor]:
        """prePhysics 参数轨（导出顺序：常量、LeanInput、Clothes2、开关、VTS 输出）→ 每个参数最后一次写入的逐帧值 [Tf]（float64，未夹取）。"""
        dd = dict(device=self.dev, dtype=torch.float64)
        out: dict[str, torch.Tensor] = {}
        for k, v in self.plan.constants.items():
            if k not in ("Clothes2", "LeanInput"):
                out[k] = torch.full((self.Tf,), float(v), **dd)
        out["LeanInput"] = self.consts["lean"].expand(self.Tf)
        out["Clothes2"] = torch.full((self.Tf,), self.clothes2, **dd)
        for sw in self.plan.switches:
            out[sw["param"]] = torch.tensor([switch_value(sw, g) for g in self.frames], **dd)
        vo = self.vts.forward(self.input_series(), self.Tf, self.plan.fin, self.c, self.consts["breathPeriod"], self.consts["breathPhase"], device=self.dev)
        out.update(vo)
        return {k: v for k, v in out.items() if k in self.rig.pindex}

    def pre_params(self) -> torch.Tensor:
        """[Tf, P] float32：规范 a～d（set 语义 = 夹到 [min, max]，存 float32）。"""
        r = self.rig
        tv = self.track_values()
        ids = list(tv)
        idx = torch.tensor([r.idx(k) for k in ids], device=self.dev)
        cols = torch.stack([tv[k] for k in ids], 1).float()
        cols = cols.clamp(r.tmin[idx], r.tmax[idx])
        base = r.tdef[None].expand(self.Tf, -1)
        return base.index_copy(1, idx, cols)

    def physics(self, pre: torch.Tensor) -> torch.Tensor:
        """预热 + 物理。返回 [WARMUP + Tf, P]（帧 start..fout−1）物理之后的参数（float32）。
        前向与 runtime 逐帧相同（float64 物理）；梯度做两处截断（A2 建议：物理链混沌，次级物理的长程梯度噪声很大）：
          - 预热帧在 no_grad 里跑完、状态 detach（预热输入 = 帧 in 的输入，它对帧 in 本身的梯度仍经正式段传回）；
          - 正式段每 self.bptt 帧把状态 detach 一次（截断 BPTT）。姿态链（AngleX → ParamAngleX 等）只滞后 1 帧、低通 5～15 帧，
            窗口 24 帧足够；后发、缎带等次级物理另有 postPhysics 低维偏移兜底。"""
        ph = self.rig.phys
        pre64 = pre.double()[:, None]
        with torch.no_grad():
            st, warm = ph.rollout(pre64[:1].expand(WARMUP, -1, -1).contiguous())
        st = st.detach()
        outs = [warm[:, 0]]
        K = getattr(self, "bptt", 24)
        for a in range(0, self.Tf, K):
            st, o = ph.rollout(pre64[a:a + K], state=st)
            outs.append(o[:, 0])
            st = st.detach()
        return torch.cat(outs, 0).float()

    def post_offsets(self, gs: list[int]) -> torch.Tensor:
        """节点偏移 → 帧 gs 上的逐参数 add 值 [len(gs), Q_params]（float64）。"""
        M = interp_matrix(self.nodes, gs).to(self.dev, torch.float64)
        return (M @ self.post)[:, self.post_gid]

    def final_params(self, after: torch.Tensor, gs: list[int]) -> torch.Tensor:
        """物理之后（全序列）→ 帧 gs 的最终参数（加 postPhysics add 偏移并夹取）。"""
        r = self.rig
        sel = after[[g - self.start for g in gs]]
        add = self.post_offsets(gs).float()
        cur = sel[:, self.post_pid]
        new = (cur + add).clamp(r.tmin[self.post_pid], r.tmax[self.post_pid])
        return sel.index_copy(1, self.post_pid, new)

    def tf_at(self, gs: list[int]) -> torch.Tensor:
        """[len(gs), 4]（x, y, ln s, rot）。"""
        M = interp_matrix(self.keys, gs).to(self.dev, torch.float64)
        out = M @ self.tf
        if self.sw_gain_idx:
            z = torch.zeros(len(gs), device=self.dev, dtype=torch.float64)
            out = out + torch.stack([z, z, self.shape_ramp(gs) @ self.sw_gain, z], 1)
        return out

    def to_screen(self, tf: torch.Tensor, s: float = 1.0, gs: list[int] | None = None) -> torch.Tensor:
        """[N, 4] → [N, 2, 3] 画布像素 → 合成像素（规范第 2 条：R·diag(scale·scaleX, scale)；× 分辨率比例 s）。
        gs 给出时按规划的翻牌曲线取 scaleX。"""
        x, y, ls, rot = tf.unbind(1)
        sc = torch.exp(ls)
        r = rot * (math.pi / 180.0)
        c, sn = torch.cos(r) * sc, torch.sin(r) * sc
        sx = torch.ones_like(sc)
        if gs is not None and self.plan.scale_x is not None:
            sx = torch.tensor([self.plan.at("scale_x", g) for g in gs], device=sc.device, dtype=sc.dtype)
        a, b, cc, d = c * sx, sn * sx, -sn, c
        e = x - (a * self.anchorX + cc * self.anchorY)
        f = y - (b * self.anchorX + d * self.anchorY)
        m = torch.stack([torch.stack([a, cc, e], -1), torch.stack([b, d, f], -1)], -2)
        return m * s

    def sigma_at(self, g: int) -> torch.Tensor:
        """帧 g 的 blur σ（720p 像素）：景深 / 模糊段内 dofK × 强度，段外常量 blur（同 traj2 dofAt）。"""
        k = self.consts
        v = self.plan.at("dof", g)
        return (k["dofK"] * v if v is not None else k["blur"]).clamp_min(0.0)

    def filters_at(self, g: int, s: float, resample: float | None = None) -> list[dict]:
        """滤镜（顺序同导出：color → resample → blur），屏幕像素量按 s 缩放。"""
        k = self.consts
        fl = []
        if self.plan.color_filter:
            fl.append({"type": "color", "brightness": k["brightness"], "contrast": k["contrast"], "saturation": k["saturation"],
                       "multiply": k["multiply"], "offset": (0.0, 0.0, 0.0)})
        # resample 因子是相对合成分辨率的：降分辨率渲染时等效因子 = f / s（≥ 1 即不处理）
        fl.append({"type": "resample", "factor": min(1.0, (self.resample if resample is None else resample) / s), "method": "bilinear"})
        fl.append({"type": "blur", "sigma": self.sigma_at(g) * s})
        return fl

    def out_at(self, g: int, s: float) -> dict:
        """输出 pass 的参数：clip（屏幕像素 × s）、不透明度、混合方式。"""
        c = self.plan.clip_at(g)
        if c is not None:
            c = {kk: (v * s if kk != "feather" else max(1.0, v) * s) for kk, v in c.items()}
        op = self.plan.at("opacity", g)
        return {"clip": c, "opacity": 1.0 if op is None else float(op), "blendMode": "normal"}

    def layer_src(self, params: torch.Tensor, ts: torch.Tensor, gs: list[int], s: float, W: int, H: int, aa=False, resample=None, quantize=True):
        """图层缓冲 → 逐帧滤镜（σ 可能逐帧不同）。返回 (滤镜之后的缓冲 [B, H, W, 4], A1 输出)。
        quantize（缺省开，同 runtime 的 RGBA8）：A3 的光栅化在真实帧上不是逐位确定的（同一输入两次渲染约 1e-4 的差异、上千个像素），
        取 8 位之后只剩个位数像素不同，光流目标与 SSIM 基本可复现。"""
        buf, out = raster_layer(self.rig, params, ts.float(), W, H, aa=aa, quantize=quantize)
        src = torch.cat([C.apply_filters(buf[i:i + 1], self.filters_at(g, s, resample), quantize=quantize) for i, g in enumerate(gs)])
        return src, out

    # ------------------------------------------------------------------ 正则
    def reg_terms(self) -> dict[str, torch.Tensor]:
        sd_in = second_diff(self.inputs, self.nodes) / self.in_tau
        # 变换的二阶差分：平移以 60px 计；旋转、缩放以 6°、0.12 计，但力臂长时按画面像素收紧到「力臂处 60px」
        # （min(6°, 60px / 力臂)、min(0.12, 60px / 力臂)）。力臂 < 约 500px 的常见情况不变
        L = self.lever()
        Lm = L[1:-1] if len(self.keys) > 2 else L[:0]
        den = torch.stack([torch.full_like(Lm, 60.0), torch.full_like(Lm, 60.0),
                           torch.minimum(torch.full_like(Lm, 0.12), 60.0 / Lm.clamp_min(1.0)),
                           torch.minimum(torch.full_like(Lm, 6.0), 60.0 / (Lm.clamp_min(1.0) * math.pi / 180))], 1)
        sd_tf = second_diff(self.tf, self.keys) / den
        sd_po = second_diff(self.post, self.nodes) / (0.05 * self.post_scale)
        # 速度上限（每帧）：x, y 30px、ln s 0.02、rot 0.8°；超出部分平方惩罚。
        # 与 shots 的镜头运动效果（zoom / pan / scale）重叠的关键帧区间放宽（set_camera_ranges），
        # 例如 s04 254～268 窗口从小到大弹出（约 ×3），0.02 / 帧的上限跟不上
        kt = torch.tensor(self.keys, device=self.dev, dtype=torch.float64)[:, None]
        v = (self.tf[1:] - self.tf[:-1]) / (kt[1:] - kt[:-1])
        vmax = self.vmax_k if self.vmax_k is not None else torch.tensor(self.VMAX, device=self.dev, dtype=torch.float64)[None].expand(len(self.keys) - 1, 4)
        # 旋转、缩放的速度上限同样按力臂换算成画面像素，不超过平移的上限（力臂 1500px 以上才起作用）
        Lv = (0.5 * (L[1:] + L[:-1])).clamp_min(1.0)[:, None]
        vmax = torch.cat([vmax[:, :2], torch.minimum(vmax[:, 2:3], vmax[:, :1] / Lv), torch.minimum(vmax[:, 3:4], vmax[:, :1] / (Lv * math.pi / 180))], 1)
        over = ((v.abs() - vmax).clamp_min(0) / vmax)
        return {
            "sd_inputs": (sd_in ** 2).sum(1).mean() if sd_in.numel() else self.inputs.sum() * 0,
            "sd_tf": (sd_tf ** 2).sum(1).mean() if sd_tf.numel() else self.tf.sum() * 0,
            "sd_post": (sd_po ** 2).mean() if sd_po.numel() else self.post.sum() * 0,
            "mag_post": ((self.post / (0.15 * self.post_scale)) ** 2).mean() * self.small_k,
            # 面捕输入离中性值的幅度先验（以 INPUT_DIMS 的 lo..hi 半宽为单位）。只对小实例有分量（small_k > 1）：
            # 眼距几十像素的实例，光流 / SSIM 对表情、头部角度几乎没有约束，会漂到范围端点（s38.side 曾出现 ParamAngleY = 30，
            # 脸在头发里缩成一小块）。大实例 small_k = 1，这一项为 0。
            "mag_inputs": (((self.inputs - self.in_neutral) / (0.5 * (self.in_hi - self.in_lo))) ** 2).mean() * (self.small_k - 1.0),
            # 小实例的姿态先验：头部与身体的面捕输入（FaceAngleX / Y / Z、BodyX / Y / Z）离中性值的偏差，以及 6 个姿态输出
            # （ParamAngle*、ParamBodyAngle*）的 postPhysics 偏移，都以 POSE_SD 计。mag_inputs / mag_post 是十几维、上百维的均值，
            # 落到单个姿态量上太弱：s38.side 曾漂到 FaceAngleX ≈ −28；只收住面捕输入时，漂移又转到 ParamAngleY 偏移（−21）。
            # 眼距几十像素时画面对姿态几乎没有约束，收在中性值附近。权重 tiny_k = max(0, small_k − 4)：眼距 30px 时 5，
            # 眼距 ≥ 45px 时为 0（s21.side 眼距 50px，姿态看得清、锚点可靠，加了反而变差）。
            "mag_pose": ((((self.inputs[:, self.pose_cols] - self.in_neutral[self.pose_cols]) / self.POSE_SD) ** 2).mean()
                         + ((self.post[:, self.pose_post_cols] / self.POSE_SD) ** 2).mean()) * self.tiny_k,
            # 小实例的变换先验：粗配准成功的关键帧上，变换 (x, y, ln s, 旋转) 离配准结果的偏差（tf_sd 为单位，tf_w 为逐关键帧权重）。
            # 光流 / SSIM 在眼距几十像素的实例上对尺度的约束很弱（s38.side 不加时拟合之后眼距 23px，配准约 30px），shots 的眼睛锚点
            # 又偏几十像素，所以配准成功时用它取代锚点项（见 Fitter.register_small）。
            "mag_tf": (((self.tf - self.tf_center) / self.tf_sd) ** 2 * self.tf_w[:, None]).sum() / self.tf_w.sum().clamp_min(1.0) * self.tiny_k
            if self.tf_center is not None else self.tf.sum() * 0,
            "speed": (over ** 2).sum(),
        }

    def project(self) -> None:
        """每步之后把未知量投影回可行域。"""
        with torch.no_grad():
            self.z_in.copy_(torch.maximum(torch.minimum(self.inputs, self.in_hi), self.in_lo) / self.in_unit)
            self.z_post.copy_(torch.maximum(torch.minimum(self.post, self.post_scale), -self.post_scale) / self.post_unit)
            k = self.consts
            k["lean"].clamp_(-30, 0)
            k["dofK"].clamp_(0.3, 2.5)
            k["breathPeriod"].clamp_(2, 6)
            k["blur"].clamp_(0, 2)  # 上限 2px@720：几何没对齐时 SSIM 偏爱模糊，再大会把表情抹掉（目测帧看不清）
            k["brightness"].clamp_(-0.3, 0.3)
            k["contrast"].clamp_(0.6, 1.4)
            k["saturation"].clamp_(0.5, 1.5)
            k["multiply"].clamp_(0.7, 1.3)

    def anchor_pred(self, gs: list[int], after: torch.Tensor) -> torch.Tensor:
        """帧 gs 的两眼虹膜网格质心（720p 合成像素）[N, 2, 2]。"""
        P = self.final_params(after, gs)
        out = self.rig.deform(P)
        c = self.rig.iris_centers(out.verts).double()  # 画布像素
        M = self.to_screen(self.tf_at(gs), 1.0, gs)
        return torch.einsum("nij,nkj->nki", M[:, :, :2], c) + M[:, None, :, 2]

    # ------------------------------------------------------------------ 快照（多假设）
    def leaves(self) -> list[torch.Tensor]:
        return [self.z_in, self.z_tf, self.z_post, *self.consts.values(), self.z_swg]

    def snapshot(self) -> dict:
        return {"leaves": [x.detach().clone() for x in self.leaves()], "resample": self.resample,
                "switches": [dict(sw) for sw in self.plan.switches]}

    def restore(self, snap: dict) -> None:
        with torch.no_grad():
            for x, v in zip(self.leaves(), snap["leaves"]):
                x.copy_(v)
        self.resample = snap["resample"]
        for sw, v in zip(self.plan.switches, snap["switches"]):
            sw.update(v)


def physics_batch(trajs: list["Traj"], bptt: int = 24) -> list[torch.Tensor]:
    """多个图层的物理一起推进（批量 B = 图层数；A2 的 rollout 对 B 是批量的，耗时与单图层几乎相同）。
    每个图层的序列从它自己的 in − warmup 开始（索引对齐到各自的起点），短的在末尾用最后一行补齐。
    前向逐帧与 runtime 相同；梯度截断同 Traj.physics（预热 no_grad、正式段每 bptt 帧 detach）。返回各图层 [WARMUP + Tf, P]。"""
    ph = trajs[0].rig.phys
    pres = [t.pre_params().double() for t in trajs]
    T = max(p.shape[0] for p in pres)
    seq = torch.stack([torch.cat([p, p[-1:].expand(T - p.shape[0], -1)], 0) if p.shape[0] < T else p for p in pres], 1)  # [T, B, P]
    with torch.no_grad():
        st, warm = ph.rollout(seq[:1].expand(WARMUP, -1, -1).contiguous())
    st = st.detach()
    outs = [warm]
    for a in range(0, T, bptt):
        st, o = ph.rollout(seq[a:a + bptt], state=st)
        outs.append(o)
        st = st.detach()
    out = torch.cat(outs, 0).float()
    return [out[:WARMUP + p.shape[0], b] for b, p in enumerate(pres)]
