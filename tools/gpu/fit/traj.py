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
        self.tf_unit = torch.tensor([1.0, 1.0, 0.01, 0.5], **dd)  # 像素、像素、ln scale、度
        self.z_tf = torch.zeros(len(self.keys), 4, **dd).requires_grad_(True)
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

    def set_tf(self, tf: torch.Tensor) -> None:
        with torch.no_grad():
            self.z_tf.copy_(tf / self.tf_unit)

    # ------------------------------------------------------------------ 参数组
    def param_groups(self, lr: dict) -> list[dict]:
        g = []
        add = lambda ps, key: g.append({"params": ps, "lr": lr[key], "name": key}) if lr.get(key, 0) > 0 else None
        add([self.z_tf], "tf")
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
        return M @ self.tf

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
        sd_tf = second_diff(self.tf, self.keys) / torch.tensor([60.0, 60.0, 0.12, 6.0], device=self.dev, dtype=torch.float64)
        sd_po = second_diff(self.post, self.nodes) / (0.05 * self.post_scale)
        # 速度上限（每帧）：x, y 30px、ln s 0.02、rot 0.8°；超出部分平方惩罚
        kt = torch.tensor(self.keys, device=self.dev, dtype=torch.float64)[:, None]
        v = (self.tf[1:] - self.tf[:-1]) / (kt[1:] - kt[:-1])
        vmax = torch.tensor([30.0, 30.0, 0.02, 0.8], device=self.dev, dtype=torch.float64)
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
        return [self.z_in, self.z_tf, self.z_post, *self.consts.values()]

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
