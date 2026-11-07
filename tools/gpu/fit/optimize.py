"""拟合单元（一个或多个图层）的分阶段、由粗到细的联合优化。

显示帧 f 上，每个在 f（与其保持帧）有效的图层 L 用帧 g_L = hold_L(f) 求值；各图层滤镜之后按数组顺序（z 从小到大）
经 clip × 不透明度合成（与 runtime 相同），再合成到参考帧上与参考帧 f 比较。

阶段：
  1. 初值：各图层变换由 shots 眼睛给出（reanchor：按当前姿态的实际虹膜位置求相似变换，关键帧加权最小二乘）；
  2. 前倾多假设（规划没固定前倾的图层）：粗尺度光度项比较 → 取前两名各跑一段短的光流粗对齐，按 SSIM 选定；
  3. 光流对应阶段 F1 / F2 / F3：外层渲染「合成图」并求合成图 → 参考帧的 DIS 光流；每个图层的顶点只在「自己在最前面」处取目标
     （自己的输出 α > 0.5 且上方图层的合成 α < 0.5，即按遮挡关系把同一张光流分给各图层）；内层只跑 VTS + 物理 + 变形，
     顶点走向目标（Cauchy）+ 锚点 + 时间正则。长镜头每次外层只渲染一个轮换的帧子集（≤ max_frames 帧）；
  4. 开关切换帧的离散估计（F2 之后）：规划给的 ±8 帧窗口内逐个候选整段前向，按窗口内的 SSIM 取最好；
  5. 外观常量：1280×720 上缓存图层缓冲，只对滤镜求导（blur、dofK、颜色），resample 离散搜索，数据项 = metric 口径 SSIM@360。
多图层为什么在合成图上联合求光流、按遮挡分配：参考帧只有一张，各图层在画面上互相遮挡；单独拿一个图层的渲染去对参考帧求光流，
被别的实例盖住或与别的实例重叠的地方会给出错误的对应。合成图与参考帧的结构一致，光流可信，再按「谁在最前」把目标归属到图层。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from l2d import composite as C

from .loss import _sample, blur_l1, composite_on_ref, dis_flows, masked_ssim, metric_mask, ssim_map
from .traj import Traj, interp_matrix, physics_batch


def _ssim_chunked(comp, ref, excl, s, mask=None, ch: int = 16):
    """逐块（ch 帧）算 masked_ssim：SSIM 的 5 路高斯滤波中间量对上百帧的批会占好几 GB。"""
    out = []
    for a in range(0, comp.shape[0], ch):
        S, _ = masked_ssim(comp[a:a + ch], ref[a:a + ch], excl[a:a + ch], s, mask=None if mask is None else mask[a:a + ch])
        out.append(S)
    return torch.cat(out)


def _canon_grads(opt) -> None:
    """梯度规整（可复现）：A1 的 Triton 反向核与 A2 的物理反向用原子加累加，float64 下低位随求和顺序变化（约 1e-16 相对），
    Adam 把它放大成不同的步长，几十步之后光流目标的阈值判断就会分叉。这里把每个梯度保留 24 位有效数字，
    并把小于该张量最大值 1e-10 的分量置 0（这些是数值抵消剩下的噪声）。对优化本身没有可察觉的影响。"""
    for g in opt.param_groups:
        for p in g["params"]:
            if p.grad is None:
                continue
            x = p.grad
            m, e = torch.frexp(x)
            x = torch.ldexp(torch.round(m * 2.0 ** 24) / 2.0 ** 24, e)
            amax = x.abs().max()
            p.grad = torch.where(x.abs() < 1e-10 * amax, torch.zeros_like(x), x)


@dataclass
class Stage:
    name: str
    W: int
    H: int
    lr: dict
    outer: int = 10
    inner: int = 8
    flow_c: float = 8.0          # Cauchy 尺度（720p 像素）
    w_anchor: float = 0.0
    reg: dict = field(default_factory=dict)
    chunk: int = 8
    lr_final: float = 0.3
    max_frames: int = 40         # 每次外层最多渲染的显示帧数（长镜头轮换子集）


DEFAULT_REG = {"sd_inputs": 0.02, "sd_tf": 0.02, "sd_post": 0.3, "mag_post": 0.05, "mag_inputs": 0.05, "mag_pose": 0.05, "mag_tf": 0.05, "speed": 1.0}


class Fitter:
    def __init__(self, trajs: list[Traj], unit, refs, log=print, effect_weight: float = 0.5):
        self.T, self.u, self.refs, self.log = trajs, unit, refs, log
        self.dev = trajs[0].dev
        self.wf = {f: (effect_weight if unit.cls[f] == "effect" else 1.0) for f in unit.frames}
        # 渲染帧：至少是一个图层的节点（新画面）的显示帧
        self.F_all = sorted({f for t in trajs for f in t.nodes if f in self.wf})
        self.timing: dict[str, float] = {}
        self.history: list[dict] = []
        # 镜头运动效果（zoom / pan / scale）的区间：这些区间里变换允许快速变化（Traj.set_camera_ranges）
        cam = [(int(e["start"]), int(e["end"])) for e in unit.effects if e.get("kind") in ("zoom", "pan", "scale")]
        for t in trajs:
            t.set_camera_ranges(cam)

    def _tick(self, k, t0):
        torch.cuda.synchronize()
        self.timing[k] = self.timing.get(k, 0.0) + time.time() - t0

    def active(self, t: Traj, f: int) -> bool:
        return t.plan.active(f) and t.plan.active(t.plan.hold(f))

    # ------------------------------------------------------------------ 合成渲染（不求导）
    def render(self, frames: list[int], s: float, W: int, H: int, chunk: int = 8, resample=None, afters=None, reuse=None):
        """显示帧 frames 的合成图 [N, H, W, 4]，以及每个图层的 (g 列表, 屏幕顶点, 有效标志, 输出 α) 供光流分配。
        reuse：{图层序号: 上一次 render 的该图层结果}，这些图层不重画（开关候选搜索时只重画被搜索的图层）。"""
        afters = afters or physics_batch(self.T)
        per = []
        for li, (t, after) in enumerate(zip(self.T, afters)):
            if reuse and li in reuse:
                per.append(reuse[li])
                continue
            idx = [i for i, f in enumerate(frames) if self.active(t, f)]
            gs = [t.plan.hold(frames[i]) for i in idx]
            uniq = sorted(set(gs))
            srcs, verts, valid = {}, {}, {}
            if uniq:
                P = t.final_params(after, uniq)
                TS = t.to_screen(t.tf_at(uniq), s, uniq)
                for a in range(0, len(uniq), chunk):
                    cg = uniq[a:a + chunk]
                    src, dout = t.layer_src(P[a:a + chunk], TS[a:a + chunk], cg, s, W, H, resample=resample)
                    M = TS[a:a + chunk].float()
                    vs = torch.einsum("bij,bvj->bvi", M[:, :, :2], dout.verts.float()) + M[:, None, :, 2]
                    ok = (dout.visible & (dout.opacity > 0.05))[:, t.rig.raster.geom.vtx_draw]
                    for j, g in enumerate(cg):
                        srcs[g], verts[g], valid[g] = src[j], vs[j], ok[j]
            per.append({"idx": idx, "gs": gs, "src": srcs, "verts": verts, "valid": valid})
        comp = torch.zeros(len(frames), H, W, 4, device=self.dev)
        outa = [dict() for _ in self.T]   # 图层 → {帧序号: 输出 α [H, W]}
        for li, (t, d) in enumerate(zip(self.T, per)):
            for i, g in zip(d["idx"], d["gs"]):
                o = t.out_at(g, s)
                a = torch.full((1, H, W, 1), o["opacity"], device=self.dev)
                if o["clip"]:
                    a = a * C.clip_coverage(o["clip"], 1, H, W, comp)
                src = d["src"][g][None] * a
                outa[li][i] = src[0, ..., 3]
                comp[i] = C.blend(comp[i:i + 1], src, o["blendMode"])[0]
        self._last_per = per
        return comp, per, outa

    # ------------------------------------------------------------------ 初值
    def reanchor(self, ti: list[int] | None = None, lam: float = 0.05) -> None:
        """变换关键帧 ← shots 眼睛：每个眼睛采样帧上，用该帧实际的虹膜中心（含物理与偏移）与 shots 两眼求相似变换（两点精确），
        再对关键帧做带二阶差分正则的加权最小二乘。前倾、身体等姿态改变后虹膜在画布上的位置会变，刚体初值要跟着重算。"""
        with torch.no_grad():
            afters = physics_batch(self.T)
        for k, (t, after) in enumerate(zip(self.T, afters)):
            if ti is not None and k not in ti:
                continue
            p = t.plan
            if not p.eye_init:
                continue
            fs = [f for f, _, _ in p.eye_init]
            gs = [p.hold(f) for f in fs]
            with torch.no_grad():
                P = t.final_params(after, gs)
                c = t.rig.iris_centers(t.rig.deform(P).verts).double().cpu().numpy()
            rows = []
            for (f, e, _), cc in zip(p.eye_init, c):
                de, dc = e[1] - e[0], cc[1] - cc[0]
                sx = p.at("scale_x", p.hold(f)) or 1.0
                sc = np.hypot(*de) / max(1e-6, np.hypot(dc[0] * sx, dc[1]))
                rot = math.atan2(de[1], de[0]) - math.atan2(dc[1], dc[0] * sx)
                R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
                xy = e.mean(0) - sc * R @ (np.array([sx, 1.0]) * (cc.mean(0) - np.array([t.anchorX, t.anchorY])))
                rows.append([xy[0], xy[1], math.log(sc), math.degrees(rot)])
            Y = np.array(rows)
            if t.sw_gain_idx:  # 体型开关的尺度增益不属于关键帧（tf_at 另加），这里扣掉
                with torch.no_grad():
                    Y[:, 2] -= (t.shape_ramp(gs) @ t.sw_gain).cpu().numpy()
            A = interp_matrix(t.keys, gs).numpy()
            w = np.array([max(0.3, c_) for _, _, c_ in p.eye_init])
            n = len(t.keys)
            D = np.zeros((max(0, n - 2), n))
            for i in range(n - 2):
                D[i, i:i + 3] = [1, -2, 1]
            M = A.T @ (w[:, None] * A) + lam * w.sum() / max(1, n) * D.T @ D + 1e-6 * np.eye(n)
            X = np.linalg.solve(M, A.T @ (w[:, None] * Y))
            t.set_tf(torch.tensor(X, device=t.dev, dtype=torch.float64))

    # ------------------------------------------------------------------ 数据项评估（不求导）
    def score(self, frames: list[int], W=640, H=360, kind="ssim", sig=(16.0, 8.0), resample=None, reuse=None, wmask=None,
              bg_grey: bool = False) -> float:
        """frames 上的数据项（越小越好）：kind = ssim → Σ w (1 − SSIM)；blur → 粗尺度光度项。
        bg_grey：渲染合成到中灰底（而不是参考帧）、遮罩只用 wmask（不随渲染的 α 变）。比较「画出来的范围不同」的候选时用：
        合成到参考帧上、遮罩随 α 变时，没画到的地方等于参考帧本身，画得越少越占便宜（见 estimate_switches）。"""
        s = W / 1280.0
        with torch.no_grad():
            comp, _, outa = self.render(frames, s, W, H, resample=resample, reuse=reuse)
            self._last_outa = outa
            if bg_grey:
                comp = torch.cat([comp[..., :3] + 0.5 * (1 - comp[..., 3:4]), torch.ones_like(comp[..., 3:4])], -1)
            ref, excl = self.refs.at(W, H, frames)
            ref, excl = torch.tensor(ref, device=self.dev), torch.tensor(excl, device=self.dev)
            w = torch.tensor([self.wf[f] for f in frames], device=self.dev)
            if kind == "ssim":
                m = None
                if wmask is not None:  # 只在给定区域（如某个图层的覆盖范围）里算 metric 口径的 SSIM
                    m = metric_mask(comp[..., 3], excl, s) * wmask
                S = _ssim_chunked(comp, ref, excl, s, m)
                return float(((1 - S) * w).sum() / w.sum())
            v = sum(blur_l1(comp, ref, excl, g, s) for g in sig) / len(sig)
            return float((v * w).sum() / w.sum() / 20.0)

    # ------------------------------------------------------------------ 光流目标
    def flow_targets(self, st: Stage, frames: list[int]) -> dict:
        s = st.W / 1280.0
        t0 = time.time()
        with torch.no_grad():
            comp, per, outa = self.render(frames, s, st.W, st.H, st.chunk)
            ref, excl = self.refs.at(st.W, st.H, frames)
            ref, excl = torch.tensor(ref, device=self.dev), torch.tensor(excl, device=self.dev)
            S = _ssim_chunked(comp, ref, excl, s)
            flows = [x for a in range(0, len(frames), 16) for x in dis_flows(composite_on_ref(comp[a:a + 16], ref[a:a + 16]), ref[a:a + 16])]
            tg = [dict() for _ in self.T]
            nL = len(self.T)
            for i, f in enumerate(frames):
                fw, bw = (torch.tensor(x, device=self.dev) for x in flows[i])
                # 上方图层的合成 α（从最上层往下累积）
                above = torch.zeros(st.H, st.W, device=self.dev)
                for li in reversed(range(nL)):
                    d = per[li]
                    if i in outa[li]:
                        g = d["gs"][d["idx"].index(i)]
                        p = d["verts"][g]
                        f1 = _sample(fw, p)
                        f2 = _sample(bw, p + f1)
                        a_own = _sample(outa[li][i][..., None], p)[:, 0]
                        a_up = _sample(above[..., None], p)[:, 0]
                        ok = d["valid"][g] & ((f1 + f2).norm(dim=-1) < 1.5) & (a_own > 0.5) & (a_up < 0.5) \
                            & (p[:, 0] > 0) & (p[:, 1] > 0) & (p[:, 0] < st.W) & (p[:, 1] < st.H)
                        vid = ok.nonzero()[:, 0]
                        tg[li][f] = (g, vid, ((p + f1)[vid] / s).double())
                        above = above + outa[li][i] * (1 - above)
        self._tick("render", t0)
        return {"targets": tg, "ssim": {f: float(v) for f, v in zip(frames, S)}}

    def subsets(self, st: Stage) -> list[list[int]]:
        k = max(1, math.ceil(len(self.F_all) / st.max_frames))
        return [self.F_all[j::k] for j in range(k)]

    def run_flow_stage(self, st: Stage, outer: int | None = None) -> float:
        """外层渲染 + 光流定目标；内层 inner 步 Adam 只跑物理 + 变形。返回最后一次外层的渲染 SSIM 均值。"""
        # 长单元（渲染帧多于 max_frames，按子集轮换）外层次数按子集数放大（最多 3 倍），保证每帧被访问的次数相近
        outer = (st.outer * min(3, len(self.subsets(st)))) if outer is None else outer
        groups = [g for t in self.T for g in t.param_groups(st.lr)]
        opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
        base = [g["lr"] for g in opt.param_groups]
        regw = {**DEFAULT_REG, **st.reg}
        subs = self.subsets(st)
        n = outer * st.inner
        t_stage = time.time()
        ms = float("nan")
        for o in range(outer):
            frames = subs[o % len(subs)]
            ft = self.flow_targets(st, frames)
            ms = float(np.mean(list(ft["ssim"].values())))
            # 本次外层的目标在内层各步不变：预先整理成每个图层的 (行, 顶点, 目标, 权重) 张量；
            # 内层所有图层（含锚点帧）一次变形（B = 各图层帧数之和），没有逐帧的 GPU 同步
            plan_l = []
            w_sum = {}   # 单元 → 有效点权重之和（多单元批处理时各单元分别归一，见下）
            for li, t in enumerate(self.T):
                tgts = ft["targets"][li]
                uniq = sorted({g for g, _, _ in tgts.values()})
                rows, cols, tg_, wp = [], [], [], []
                for f, (g, vid, tgt) in tgts.items():
                    if vid.numel() == 0:
                        continue
                    rows.append(torch.full((vid.numel(),), uniq.index(g), device=self.dev, dtype=torch.long))
                    cols.append(vid)
                    tg_.append(tgt)
                    wp.append(torch.full((vid.numel(),), self.wf[f], device=self.dev, dtype=torch.float64))
                    w_sum[t.plan.unit] = w_sum.get(t.plan.unit, 0.0) + self.wf[f] * vid.numel()
                anc = t.anchor_frames if (st.w_anchor > 0 and t.anchor_xy is not None) else []
                plan_l.append({"uniq": uniq, "anc": anc,
                               "rows": torch.cat(rows) if rows else None, "cols": torch.cat(cols) if cols else None,
                               "tgt": torch.cat(tg_) if tg_ else None, "w": torch.cat(wp) if wp else None})
            # 数据项按单元归一：每个单元的数据项除以它自己的有效点数，再对单元求和。上一版除以整批的点数，
            # 一批里小单元的数据项被摊薄 5～10 倍，而各图层的正则、锚点项不变，相当于正则加重了 5～10 倍
            # （全片批处理 s23 普通帧 0.436，单独跑 0.688）。单个单元时与原来完全相同。
            self._wnorm = {un: max(v, 1.0) for un, v in w_sum.items()}
            for k in range(st.inner):
                prog = (o * st.inner + k) / max(1, n - 1)
                kk = 0.5 * (1 + math.cos(math.pi * prog))
                for g_, b in zip(opt.param_groups, base):
                    g_["lr"] = b * (st.lr_final + (1 - st.lr_final) * kk)
                opt.zero_grad(set_to_none=True)
                t0 = time.time()
                afters = physics_batch(self.T)
                tot = torch.zeros((), device=self.dev, dtype=torch.float64)
                data = torch.zeros((), device=self.dev, dtype=torch.float64)
                data_u = {}
                terms = {}
                Ps, sizes = [], []
                for t, after, pl in zip(self.T, afters, plan_l):
                    gs_all = pl["uniq"] + pl["anc"]
                    sizes.append(len(gs_all))
                    if gs_all:
                        Ps.append(t.final_params(after, gs_all))
                verts = t.rig.deform(torch.cat(Ps)).verts if Ps else None
                off = 0
                for li, (t, pl, nB) in enumerate(zip(self.T, plan_l, sizes)):
                    if nB == 0:
                        continue
                    V = verts[off:off + nB]
                    off += nB
                    nu = len(pl["uniq"])
                    if pl["rows"] is not None:
                        M = t.to_screen(t.tf_at(pl["uniq"]), 1.0, pl["uniq"])
                        p = V[pl["rows"], pl["cols"]].to(M.dtype)
                        Mr = M[pl["rows"]]
                        vs = torch.einsum("nij,nj->ni", Mr[:, :, :2], p) + Mr[:, :, 2]
                        dd = (vs - pl["tgt"]).norm(dim=-1)
                        v_ = (torch.log1p((dd / st.flow_c) ** 2) * pl["w"]).sum()
                        data_u[t.plan.unit] = data_u.get(t.plan.unit, 0.0) + v_
                    if pl["anc"]:
                        c = t.rig.iris_centers(V[nu:]).double()
                        Ma = t.to_screen(t.tf_at(pl["anc"]), 1.0, pl["anc"])
                        pr = torch.einsum("nij,nkj->nki", Ma[:, :, :2], c) + Ma[:, None, :, 2]
                        d = (pr - t.anchor_xy).norm(dim=-1)
                        ed = (t.anchor_xy[:, 1] - t.anchor_xy[:, 0]).norm(dim=-1)
                        cc = torch.clamp(0.15 * ed, min=10.0)[:, None]
                        la = (torch.log1p((d / cc) ** 2) * t.anchor_w[:, None]).sum() / (2 * t.anchor_w.sum())
                        tot = tot + st.w_anchor * la
                        terms[f"anchor_px[{t.plan.id}]"] = d.mean()
                for t in self.T:
                    for kname, v in t.reg_terms().items():
                        tot = tot + regw.get(kname, 0.0) * v
                        terms[kname] = terms.get(kname, 0.0) + v.detach()
                # 数据项按有效点数加权平均（归一的分母在外层开始时固定，内层各步的目标一致）
                for un, v_ in data_u.items():
                    data = data + v_ / self._wnorm[un]
                tot = tot + data
                tot.backward()
                _canon_grads(opt)
                opt.step()
                for t in self.T:
                    t.project()
                self._tick("inner", t0)
                terms["flow"] = data.detach() / max(1, len(data_u))
            terms = {k_: float(v) for k_, v in terms.items()}
            self.history.append({"stage": st.name, "outer": o, "ssim": ms, **terms,
                                 "lean": [float(t.consts["lean"]) for t in self.T], "sec": time.time() - t_stage})
            self.log(f"[{st.name}] 外层 {o:2d}（{len(frames)} 帧）SSIM(渲染时) {ms:.4f} flow {terms['flow']:.3f} "
                     + " ".join(f"{k} {v:.3g}" for k, v in terms.items() if k != "flow")
                     + f" lean {[round(float(t.consts['lean']), 2) for t in self.T]} ({time.time() - t_stage:.0f}s)")
        return ms

    # ------------------------------------------------------------------ SSIM 精修（与验收同口径，直接对数据项求导）
    def run_ssim_stage(self, st: Stage, iters: int, w_ssim: float = 1.0) -> float:
        """光流阶段之后的精修：数据项直接是 metric 口径的 SSIM@360（640×360 渲染，合成到参考帧上），对全部几何未知量求导。
        光流目标的精度有限（参考帧偏软、大片纯色区、clip 边缘），残差停在 1～2px 时 SSIM 仍有可观的提升空间；
        接入 A7 的快速光栅化之后带边缘梯度的前向 + 反向约 0.1 s / 帧，这一段才划算。
        一步：整段物理（保留图）→ 各图层帧的最终参数与 to_screen 截成叶子 → 分块渲染、合成、算 SSIM、逐块反向 →
        用叶子上的梯度把整段图反向一次（加锚点与时间正则）→ Adam。帧按子集轮换。"""
        groups = [g for t in self.T for g in t.param_groups(st.lr)]
        opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
        base = [g["lr"] for g in opt.param_groups]
        regw = {**DEFAULT_REG, **st.reg}
        subs = self.subsets(st)
        s_ = st.W / 1280.0
        t_stage = time.time()
        last = float("nan")
        for it in range(iters):
            frames = subs[it % len(subs)]
            kk = 0.5 * (1 + math.cos(math.pi * it / max(1, iters - 1)))
            for g_, b in zip(opt.param_groups, base):
                g_["lr"] = b * (st.lr_final + (1 - st.lr_final) * kk)
            opt.zero_grad(set_to_none=True)
            t0 = time.time()
            afters = physics_batch(self.T)
            leaves = []
            for t, after in zip(self.T, afters):
                act = [f for f in frames if self.active(t, f)]
                uniq = sorted({t.plan.hold(f) for f in act})
                if not uniq:
                    leaves.append(None)
                    continue
                P = t.final_params(after, uniq)
                TS = t.to_screen(t.tf_at(uniq), s_, uniq)
                leaves.append({"uniq": uniq, "P": P, "TS": TS, "Pl": P.detach().requires_grad_(True), "TSl": TS.detach().requires_grad_(True)})
            ref, excl = self.refs.at(st.W, st.H, frames)
            ref, excl = torch.tensor(ref, device=self.dev), torch.tensor(excl, device=self.dev)
            wsum = sum(self.wf[f] for f in frames)
            # 按单元归一（同 run_flow_stage）：帧 f 的权重除以它所在单元在本子集里的权重之和
            wu = {}
            for f in frames:
                wu[self.unit_of(f)] = wu.get(self.unit_of(f), 0.0) + self.wf[f]
            ssum = 0.0
            for a in range(0, len(frames), st.chunk):
                cf = frames[a:a + st.chunk]
                comp = torch.zeros(len(cf), st.H, st.W, 4, device=self.dev)
                for t, lv in zip(self.T, leaves):
                    if lv is None:
                        continue
                    rows = [i for i, f in enumerate(cf) if self.active(t, f)]
                    if not rows:
                        continue
                    gs = [t.plan.hold(cf[i]) for i in rows]
                    ug = sorted(set(gs))
                    ix = [lv["uniq"].index(g) for g in ug]
                    src, _ = t.layer_src(lv["Pl"][ix], lv["TSl"][ix], ug, s_, st.W, st.H, aa=True, quantize=False)
                    for i, g in zip(rows, gs):
                        o = t.out_at(g, s_)
                        al = torch.full((1, st.H, st.W, 1), o["opacity"], device=self.dev)
                        if o["clip"]:
                            al = al * C.clip_coverage(o["clip"], 1, st.H, st.W, comp)
                        layer = src[ug.index(g)][None] * al
                        comp = torch.cat([comp[:i], C.blend(comp[i:i + 1], layer, o["blendMode"]), comp[i + 1:]])
                S, _ = masked_ssim(comp, ref[a:a + len(cf)], excl[a:a + len(cf)], s_)
                w = torch.tensor([self.wf[f] for f in cf], device=self.dev, dtype=S.dtype)
                wn = torch.tensor([wu[self.unit_of(f)] for f in cf], device=self.dev, dtype=S.dtype)
                lc = w_ssim * ((1 - S) * w / wn).sum()
                lc.backward()
                ssum += float((S.detach() * w).sum())
            tot = torch.zeros((), device=self.dev, dtype=torch.float64)
            for t, lv, after in zip(self.T, leaves, afters):
                if lv is not None and lv["Pl"].grad is not None:
                    tot = tot + (lv["P"] * lv["Pl"].grad).sum() + (lv["TS"] * lv["TSl"].grad).sum()
                if st.w_anchor > 0 and t.anchor_xy is not None:
                    pr = t.anchor_pred(t.anchor_frames, after)
                    d = (pr - t.anchor_xy).norm(dim=-1)
                    ed = (t.anchor_xy[:, 1] - t.anchor_xy[:, 0]).norm(dim=-1)
                    cc = torch.clamp(0.15 * ed, min=10.0)[:, None]
                    tot = tot + st.w_anchor * (torch.log1p((d / cc) ** 2) * t.anchor_w[:, None]).sum() / (2 * t.anchor_w.sum())
                for kname, v in t.reg_terms().items():
                    tot = tot + regw.get(kname, 0.0) * v
            tot.backward()
            _canon_grads(opt)
            opt.step()
            for t in self.T:
                t.project()
            self._tick("ssim", t0)
            last = ssum / wsum
            self.history.append({"stage": st.name, "it": it, "ssim": last, "sec": time.time() - t_stage})
            if it % 3 == 0 or it == iters - 1:
                self.log(f"[{st.name}] {it:3d}（{len(frames)} 帧）SSIM {last:.4f}（{time.time() - t_stage:.0f}s）")
        return last

    _wnorm: dict = {}

    def unit_of(self, f: int) -> str:
        """显示帧 f 所在的拟合单元（多单元批处理时区分；单个单元时就是它自己）。"""
        for uu in (self.u.units or [self.u]):
            if uu.start <= f < uu.end:
                return uu.unit
        return self.u.unit
    switch_mask = "intersect"  # 开关估计的数据区：intersect（两种状态都画到的地方）/ union_grey（并集，合成到中灰底）
    lean_hyp = 3  # 前倾试拟合的假设个数

    # ------------------------------------------------------------------ 小实例的尺度多假设
    def bbox_mask(self, t: Traj, frames: list[int], W: int, H: int) -> torch.Tensor:
        """shots 轨迹的 bbox（metric.py 的插值规则）→ [N, H, W] 遮罩（当前分辨率）。"""
        import sys
        from .rig import REPO
        sys.path.insert(0, str(REPO / "tools/qa"))
        import metric
        ks = [x for x in t.plan.track if "bbox" in x]
        out = torch.zeros(len(frames), H, W, device=self.dev)
        sx, sy = W / 1280.0, H / 720.0
        for i, f in enumerate(frames):
            r = metric._interp_rects(ks, f, "bbox") if ks else None
            if r is None:
                continue
            x, y, w, h = r
            x0, x1 = max(0, int(x * sx)), min(W, int((x + w) * sx) + 1)
            y0, y1 = max(0, int(y * sy)), min(H, int((y + h) * sy) + 1)
            out[i, y0:y1, x0:x1] = 1
        return out

    # 头部模板：以两眼中点为原点、眼距 d 为单位的裁切范围（x ±2.2d，y −2.6d..+1.6d：头发顶、两耳、下巴与领口）
    HEAD_BOX = (2.2, 2.6, 1.6)

    def _head_match(self, ref: np.ndarray, src: np.ndarray, off: tuple, eyes: np.ndarray, anc: tuple, R: float):
        """一个关键帧的头部模板匹配（720p 像素）。ref：参考帧 [720, 1280, 3]；src：该图层单独渲染的窗口（预乘 RGBA [h, w, 4]，
        左上角在屏幕 off）；eyes：渲染里两眼虹膜的屏幕位置 [2, 2]；anc：图层锚点（变换的 x, y）的屏幕位置；R：允许的最大位移（像素）。
        模板 = 头部框（HEAD_BOX）内、合成到中灰底的颜色，掩码 = α > 0.3；参考帧在「模板框外扩 R」的区域内做零均值归一化互相关
        （TM_CCOEFF_NORMED，三通道）。返回 (分数, 新锚点 x, 新锚点 y)；模板太小或出画时返回 None。"""
        import cv2
        H, W = ref.shape[:2]
        ox, oy = off
        (ex, ey), d = eyes.mean(0), float(np.hypot(*(eyes[1] - eyes[0])))
        kx, ku, kd = self.HEAD_BOX
        x0, x1 = max(0, ox, int(ex - kx * d)), min(W, ox + src.shape[1], int(ex + kx * d) + 1)
        y0, y1 = max(0, oy, int(ey - ku * d)), min(H, oy + src.shape[0], int(ey + kd * d) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        s_ = src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        a = s_[..., 3]
        m = (a > 0.3).astype(np.float32)
        if m.sum() < 50:
            return None
        tpl = (s_[..., :3] * 255 + 128 * (1 - a[..., None])).astype(np.float32)
        sx0, sy0 = max(0, int(x0 - R)), max(0, int(y0 - R))
        sx1, sy1 = min(W, int(x1 + R)), min(H, int(y1 + R))
        if sx1 - sx0 <= x1 - x0 or sy1 - sy0 <= y1 - y0:
            return None
        r = cv2.matchTemplate(ref[sy0:sy1, sx0:sx1].astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED, mask=np.repeat(m[..., None], 3, 2))
        r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
        _, mv, _, ml = cv2.minMaxLoc(r)
        return float(mv), sx0 + ml[0] + (anc[0] - x0), sy0 + ml[1] + (anc[1] - y0)

    def _render_keys(self, t: Traj, gs: list[int], tf: torch.Tensor, chunk: int = 16):
        """图层 t 在帧 gs（变换 tf [len(gs), 4]）单独渲染，只渲染每帧头部框附近的窗口（720p 像素；整帧渲染慢一个量级）。
        返回 (窗口 [N, h, w, 4] 预乘 RGBA numpy, 窗口左上角 [N, 2], 两眼屏幕位置 [N, 2, 2], 锚点屏幕位置 [N, 2])。"""
        kx, ku, kd = self.HEAD_BOX
        with torch.no_grad():
            after = physics_batch([t])[0]
            P = t.final_params(after, gs)
            TS = t.to_screen(tf, 1.0, gs)
            ic = t.rig.iris_centers(t.rig.deform(P).verts).double()
            eyes = torch.einsum("nij,nkj->nki", TS[:, :, :2], ic) + TS[:, None, :, 2]
            em = eyes.mean(1)
            dmax = float((eyes[:, 1] - eyes[:, 0]).norm(dim=-1).max())
            w, h = int(2 * kx * dmax) + 16, int((ku + kd) * dmax) + 16
            off = torch.stack([torch.round(em[:, 0] - w / 2), torch.round(em[:, 1] - ku * dmax - 8)], 1)
            TSw = TS.clone()
            TSw[:, :, 2] -= off
            srcs = [t.layer_src(P[a:a + chunk], TSw[a:a + chunk], gs[a:a + chunk], 1.0, w, h)[0].cpu().numpy() for a in range(0, len(gs), chunk)]
        return np.concatenate(srcs), off.long().cpu().numpy(), eyes.cpu().numpy(), tf[:, :2].cpu().numpy()

    def _body_render(self, t: Traj, gs: list[int], tfk: torch.Tensor, sc: float, cur: list, min_score: float) -> dict:
        """体型假设的整身渲染：在头部匹配给出的位置与尺度上整帧渲染该图层，只取头部匹配成功、且在体型开关之前（斜坡 < 0.5）
        的关键帧。返回 {关键帧序号: (合成到中灰底的 RGB uint8 [H, W, 3], α > 0.3 [H, W])}。"""
        pre = np.ones(len(gs), bool)
        if t.sw_gain_idx:
            with torch.no_grad():
                pre = (t.shape_ramp(gs).max(1).values < 0.5).cpu().numpy()
        js = [j for j in range(len(gs)) if pre[j] and cur[j] is not None and cur[j][0] >= min_score]
        out = {}
        if not js:
            return out
        with torch.no_grad():
            tf = tfk[js].clone()
            tf[:, 2] += math.log(sc)
            for i, j in enumerate(js):
                tf[i, 0], tf[i, 1] = cur[j][1], cur[j][2]
            g = [gs[j] for j in js]
            after = physics_batch([t])[0]
            P = t.final_params(after, g)
            TS = t.to_screen(tf, 1.0, g)
            for a in range(0, len(g), 8):
                src, _ = t.layer_src(P[a:a + 8], TS[a:a + 8], g[a:a + 8], 1.0, 1280, 720)
                for k in range(src.shape[0]):
                    s = src[k]
                    img = (s[..., :3] * 255 + 128 * (1 - s[..., 3:4])).clamp(0, 255).byte().cpu().numpy()
                    out[js[a + k]] = (img, (s[..., 3] > 0.3).cpu().numpy())
        return out

    @staticmethod
    def _body_scores(renders: dict, refs: np.ndarray) -> dict:
        """各体型假设的整身得分：每个关键帧在「所有假设的轮廓之并」这同一个区域里，各假设（合成到中灰底）与参考帧算零均值
        归一化相关，再对所有假设都有的关键帧取平均。区域按假设各自的轮廓取时，轮廓小的（Q 版）只在头部附近比较，容易占便宜
        （s15.left：体型 1 以 0.358 略胜实际的 0.4 的 0.341）。"""
        keys = set.intersection(*[set(r) for r in renders.values()]) if renders else set()
        sc = {c: [] for c in renders}
        for j in sorted(keys):
            m = np.zeros_like(next(iter(renders.values()))[j][1])
            for r in renders.values():
                m |= r[j][1]
            if m.sum() < 200:
                continue
            y = refs[j][m].astype(np.float64)
            y -= y.mean(0)
            for c, r in renders.items():
                x = r[j][0][m].astype(np.float64)
                x -= x.mean(0)
                sc[c].append(float((x * y).sum() / max(1e-9, np.sqrt((x * x).sum() * (y * y).sum()))))
        return {c: (float(np.mean(v)) if v else -1.0) for c, v in sc.items()}

    def register_small(self, max_eye: float = 90.0, scales=(0.5, 0.57, 0.65, 0.75, 0.87, 1.0, 1.15, 1.3), fine=(0.92, 1.0, 1.08),
                       rots=(-12.0, -6.0, 6.0, 12.0), rot_margin: float = 0.02, min_score: float = 0.35,
                       retry_scales=(0.87, 1.0, 1.15, 1.3), bodies=(0.0, 0.4, 1.0), body_margin: float = 0.03, body_min: float = 0.15,
                       reliable_conf: float = 0.6) -> None:
        """小实例粗配准（尺度 + 位置 + 屏幕内倾斜，通用）：眼距 < max_eye（720p）的实例，shots 的眼睛锚点常有几成的尺度误差和
        几十像素的位置误差（s21.side 眼距标 50、实际约 35；s38.side 位置偏 30～50px），超出光流的捕获范围。做法：
          1. 尺度：每个变换关键帧（取最近的节点帧）单独渲染该图层的**头部模板**（两眼中点为原点，x ±2.2d、y −2.6d..+1.6d，
             d = 渲染眼距；掩码 α > 0.3，颜色合成到中灰底），在参考帧上做零均值归一化互相关（_head_match），
             位移限制在 R = max(40, 2.5 × shots 眼距) 以内；尺度在全图层上按匹配分之和投票，再在 ×fine 上微调（同样投票）。
             体型开关之后的关键帧照常参加（渲染里带着开关的尺度增益 Traj.sw_gain 的初值）。
          2. 屏幕内倾斜：全部关键帧同时加 rots 里的一个旋转，各关键帧分别取最好的（比不转好不到 rot_margin 的不转）。
          3. 匹配分低于 min_score 的关键帧：搜索半径加倍、逐关键帧试 retry_scales 再匹配一次（shots 的线性插值眼睛在动作快处
             会偏出 R，s38.side 3368 偏约 80px）。写回变换关键帧的位置、ln s、旋转；仍然失败的（转场空白、特效遮挡）只改尺度。
          0. 体型多假设（bodies，Clothes2 常量）：每个候选各做一遍第 1 步，再在头部匹配的位置上比较整身模板（_body_render / _body_scores，所有假设用同一个区域），
             比规划值好 body_margin 以上才换（Traj.set_body 连带改 Clothes2 开关的起点与尺度增益）。
          4. 很小的实例（Traj.tiny_k > 0，眼距 < 45px）、成功的关键帧过半时，配准结果作为变换先验（Traj.reg_terms 的 mag_tf），并去掉与配准矛盾的 shots 眼睛锚点：
             光流 / SSIM 在眼距几十像素的实例上对尺度、位置约束很弱，而锚点本身可能偏几十像素（s38.side）。
        头部角度（FaceAngleX / Y）也试过一起搜：匹配分对它几乎没有分辨力（常落在网格端点、相邻关键帧正负交替），
        当作先验中心反而把尺度带偏，所以不搜；小实例的姿态由 mag_pose 先验收在中性值附近（见 traj.reg_terms）。
        上一版（整身模板、梯度幅值、TM_CCORR_NORMED、640×360）有两处偏向「越小越好」：整身模板在真实尺度下常常伸出画面
        （s38.side 的腿与尾巴），matchTemplate 要求模板完全落在搜索区内，大尺度根本放不到正确位置；非零均值的相关对小模板偏高。
        s38.side 因此选了 ×0.55（实际约 ×0.9），s15.left 也是 ×0.55（实际约 ×1.0）。
        大实例不做（锚点可靠，fit4 的经验是粗配准在大实例上有害）。"""
        for t in self.T:
            if not t.plan.eye_init:
                continue
            ed = float(np.median([np.hypot(*(e[1] - e[0])) for _, e, _ in t.plan.eye_init]))
            if ed >= max_eye:
                continue
            # 眼睛锚点可靠（一半以上的眼睛采样置信度 ≥ reliable_conf，即人工 / 虹膜精修过）时不配准：锚点本身就定了位置和尺度，
            # 模板匹配反而可能错（全片批处理 s08.bowl：锚点置信度 0.7～0.8，配准投出 ×0.46、体型 1，钢盆缩成一小只）
            conf = [c for _, _, c in t.plan.eye_init]
            if np.mean([c >= reliable_conf for c in conf]) >= 0.5:
                self.log(f"[粗配准 {t.plan.id}] 眼距 {ed:.0f}px：眼睛锚点可靠（{np.mean([c >= reliable_conf for c in conf]):.0%} 的采样置信度 ≥ {reliable_conf}），不配准")
                continue
            t0 = time.time()
            gs = [min(t.nodes, key=lambda n: abs(n - k)) for k in t.keys]
            refs, _ = self.refs.at(1280, 720, gs)
            R = max(40.0, 2.5 * ed)
            tf0 = t.tf.detach().clone()
            tfk = t.tf_at(gs).detach()   # 关键帧上的变换（gs 与关键帧可能差一两帧，位置按 gs 求、写回关键帧）
            nk = len(gs)
            nev = [0]

            def evaluate(sc: float, rot=None, rad: float = R):
                nev[0] += 1
                tf = tfk.clone()
                tf[:, 2] += math.log(sc)
                if rot is not None:
                    tf[:, 3] += rot
                src, off, eyes, anc = self._render_keys(t, gs, tf)
                return [self._head_match(refs[j], src[j], off[j], eyes[j], anc[j], rad) for j in range(nk)]

            def total(r):
                return sum(v[0] for v in r if v is not None)
            def vote():
                """1) 尺度投票 + 微调 → (投票表, 粗选, 微调表, 尺度, 逐关键帧匹配)。"""
                res = {sc: evaluate(sc) for sc in scales}
                tot = {sc: total(r) for sc, r in res.items()}
                s1 = max(tot, key=tot.get)
                res_f = {fk: (res[s1] if fk == 1.0 else evaluate(s1 * fk)) for fk in fine}
                tot_f = {fk: total(r) for fk, r in res_f.items()}
                return tot, s1, tot_f, s1 * max(tot_f, key=tot_f.get), list(res_f[max(tot_f, key=tot_f.get)])

            # 0) 体型多假设（Clothes2 常量）：每个候选各做一次尺度投票，再在头部匹配给出的位置上比较整身模板（切换前的关键帧）。
            #    规划的体型来自 shots 的 variant（缺省 0），小实例常常标错：s38.side 切 Q 版之前实际是体型 1（0.4），
            #    按 0 拟合时身体大一圈，只能靠扭头部角度、缩放去凑。比规划值好 body_margin 以上才换。
            body = {}
            if "Clothes2" in t.rig.pindex:
                plan_c2 = t.clothes2
                votes, rend = {}, {}
                for c2 in [plan_c2] + [c for c in bodies if abs(c - plan_c2) > 1e-6]:
                    t.set_body(c2)
                    tfk = t.tf_at(gs).detach()
                    votes[c2] = vote()
                    rend[c2] = self._body_render(t, gs, tfk, votes[c2][3], votes[c2][4], min_score)
                bs = self._body_scores(rend, refs) if len(rend) > 1 else {plan_c2: 0.0}
                del rend
                body = {c: (bs[c], votes[c]) for c in votes}
                best_c2 = max(body, key=lambda c: body[c][0])
                # 换体型要有证据：比规划值好 body_margin 以上，而且自己的整身相关不低于 body_min
                # （s08.bowl 三个假设都在 0 附近：−0.009 / 0.037 / 0.070，是噪声）
                if body[best_c2][0] < body[plan_c2][0] + body_margin or body[best_c2][0] < body_min:
                    best_c2 = plan_c2
                t.set_body(best_c2)
                tfk = t.tf_at(gs).detach()
                tot, s1, tot_f, best_s, cur = body[best_c2][1]
            else:
                tot, s1, tot_f, best_s, cur = vote()
            # 2) 屏幕内倾斜
            rot = np.zeros(nk)
            base = list(cur)
            for v in rots:
                r = evaluate(best_s, torch.full((nk,), v, device=t.dev, dtype=torch.float64))
                for j in range(nk):
                    if r[j] is not None and (cur[j] is None or r[j][0] > cur[j][0]) and (base[j] is None or r[j][0] >= base[j][0] + rot_margin):
                        cur[j], rot[j] = r[j], v
            ok = [cur[j] is not None and cur[j][0] >= min_score for j in range(nk)]
            # 2b) 失败的关键帧：搜索半径加倍、逐关键帧再试几个尺度（shots 的线性插值眼睛在动作快的地方会偏到 R 之外，
            #     s38.side 3368 偏约 80px；体型开关之后的尺度也可能与全图层投票的不同）
            kfac = np.ones(nk)
            bad = [j for j in range(nk) if not ok[j]]
            if bad:
                for fk in retry_scales:
                    r = evaluate(best_s * fk, rad=2 * R)
                    for j in bad:
                        if r[j] is not None and r[j][0] >= min_score and (not ok[j] or r[j][0] > cur[j][0]):
                            cur[j], kfac[j], ok[j], rot[j] = r[j], fk, True, 0.0
            # 3) 写回
            with torch.no_grad():
                tf = tf0.clone()
                tf[:, 2] += math.log(best_s)
                for j in range(nk):
                    if ok[j]:
                        tf[j, 0], tf[j, 1] = cur[j][1], cur[j][2]
                        tf[j, 3] += rot[j]
                        tf[j, 2] += math.log(kfac[j])
                t.set_tf(tf)
                # 4) 只对很小的实例（tiny_k > 0）：变换先验（单位：位置 max(6, 0.5 × 配准后眼距) px、ln s 0.08、旋转 8°；
                #    失败的关键帧不加），并去掉与配准结果矛盾的 shots 眼睛锚点（离配准位置超过 max(10, 0.6 × 配准后眼距)；
                #    s38.side 的锚点偏 30～80px，去掉 5 个）。眼距 50px 的 s21.side 加这两项反而变差（锚点 20px 内、姿态看得清）
                prior = 2 * sum(ok) >= nk and t.tiny_k > 0
                n_drop = 0
                if prior:
                    sp = max(6.0, 0.5 * ed * best_s)
                    t.tf_center = t.tf.detach().clone()
                    t.tf_sd = torch.tensor([sp, sp, 0.08, 8.0], device=t.dev, dtype=torch.float64)
                    t.tf_w = torch.tensor([1.0 if o else 0.0 for o in ok], device=t.dev, dtype=torch.float64)
                    if t.anchor_xy is not None:
                        dist = (t.anchor_xy.mean(1) - t.tf_at(t.anchor_frames)[:, :2]).norm(dim=-1)
                        keep = (dist <= max(10.0, 0.6 * ed * best_s)).cpu().numpy()
                        n_drop = int((~keep).sum())
                        if keep.any():
                            ki = np.nonzero(keep)[0].tolist()
                            t.anchor_frames = [t.anchor_frames[i] for i in ki]
                            t.anchor_xy, t.anchor_w = t.anchor_xy[ki], t.anchor_w[ki]
                        else:
                            t.anchor_xy = None
            shift = [float(np.hypot(cur[j][1] - float(tfk[j, 0]), cur[j][2] - float(tfk[j, 1]))) for j in range(nk) if ok[j]]
            self.log(f"[粗配准 {t.plan.id}] 眼距 {ed:.0f}px：" + (("体型（整身匹配）" + " ".join(f"{c:g}:{v[0]:.3f}" for c, v in body.items())
                     + f" → Clothes2 {t.clothes2:g}；") if body else "") + "尺度投票 " + " ".join(f"×{k:.2f}:{v:.2f}" for k, v in tot.items())
                     + f" → ×{s1:.2f}，微调 " + " ".join(f"×{k:.2f}:{v:.2f}" for k, v in tot_f.items()) + f" → ×{best_s:.3f}；"
                     + f"{len(shift)}/{nk} 个关键帧改位置（中位移动 {np.median(shift) if shift else 0:.0f}px@720）；倾斜 / 匹配分 "
                     + " ".join(f"{g}:" + (f"{rot[j]:+.0f}°/{cur[j][0]:.2f}" + (f"（重试 ×{kfac[j]:.2f}）" if j in bad else "") if ok[j] else "—") for j, g in enumerate(gs))
                     + (f"；加变换先验，去掉 {n_drop} 个与配准矛盾的锚点" if prior else "；不加变换先验、锚点照旧")
                     + f"（{nev[0]} 次渲染，{time.time() - t0:.1f}s）")

    def search_scale(self, max_eye: float = 90.0, factors=(0.6, 0.7, 0.85, 1.0, 1.15)) -> None:
        """眼距小于 max_eye（720p）的小实例：shots 的眼睛锚点在小尺寸时常有几成的尺度误差（例如 s21.side），
        光流的捕获范围（几十像素）拉不回整体缩放。这里把该图层全部关键帧的 ln s 乘上几个因子（两眼中点不动），
        在该实例 shots bbox 内比较粗尺度光度项（σ 8 / 4px），取最好的。大实例不做（锚点可靠，且 fit4 的经验是粗配准在大实例上有害）。"""
        frames = self.F_all[::max(1, len(self.F_all) // 24)]
        for t in self.T:
            if not t.plan.eye_init:
                continue
            ed = float(np.median([np.hypot(*(e[1] - e[0])) for _, e, _ in t.plan.eye_init]))
            fr = [f for f in frames if self.active(t, f)]
            if ed >= max_eye or not fr:
                continue
            wm = self.bbox_mask(t, fr, 320, 180)
            base = t.z_tf.detach().clone()
            res = []
            for k in factors:
                with torch.no_grad():
                    tf = base * t.tf_unit
                    tf[:, 2] = tf[:, 2] + math.log(k)
                    t.set_tf(tf)
                res.append((self.score_masked(fr, wm), k))
                self.log(f"[尺度 {t.plan.id}] 眼距 {ed:.0f}px，×{k:.2f}：粗尺度光度（bbox 内）{res[-1][0]:.4f}")
            best = min(res)[1]
            with torch.no_grad():
                tf = base * t.tf_unit
                tf[:, 2] = tf[:, 2] + math.log(best)
                t.set_tf(tf)
            self.log(f"[尺度 {t.plan.id}] 选定 ×{best:.2f}")

    def score_masked(self, frames, wm, W=320, H=180, sig=(8.0, 4.0)) -> float:
        """粗尺度光度项，只在 wm 内（如某实例的 shots bbox）平均。"""
        s = W / 1280.0
        with torch.no_grad():
            comp, _, _ = self.render(frames, s, W, H)
            ref, excl = self.refs.at(W, H, frames)
            ref, excl = torch.tensor(ref, device=self.dev), torch.tensor(excl, device=self.dev)
            # 合成到中灰底上（不是参考帧上）：合成到参考帧时「画得越少、越像参考」，缩小会被错误地奖励
            cs = comp[..., :3] * 255.0 + 128.0 * (1.0 - comp[..., 3:4])
            v = 0.0
            for g in sig:
                from .loss import gauss_kernel, gfilter
                k = gauss_kernel(max(0.5, g * s), device=self.dev)
                a = gfilter(cs.permute(0, 3, 1, 2), k)
                b = gfilter(ref.permute(0, 3, 1, 2), k)
                d = (a - b).abs().mean(1) * wm * (excl > 0.5)
                v += float(d.sum() / (wm * (excl > 0.5)).sum().clamp_min(1))
            return v / len(sig) / 20.0

    # ------------------------------------------------------------------ 前倾多假设
    def search_lean(self, st_trial: Stage, values=(0.0, -10.0, -20.0, -25.0, -30.0), trial_outer: int = 4) -> None:
        """规划没固定前倾的图层（planAuto：每个单元至多一个）：先用粗尺度光度项（σ 16 / 8px，在 ±16px 内单调）比较各值
        （每个值先 reanchor），取前两名；再做两次试拟合——第 k 次把每个这样的图层设为它自己的第 k 名，整批一起跑
        trial_outer 次光流粗对齐——最后每个单元各自取 SSIM 更好的那一次（单元之间互不影响，所以可以一起试）。
        单一的光度项比较在 −20 与 −30 之间常常打平，直接取第一名会落进局部极小（s24 上 0.735 vs 0.79+）。"""
        free = [i for i, t in enumerate(self.T) if t.lean_free]
        if not free:
            return
        units = sorted({self.T[i].plan.unit for i in free})
        ufr = {un: [f for f in self.F_all if any(self.active(t, f) for t in self.T if t.plan.unit == un)] for un in units}
        ufr = {un: fr[::max(1, len(fr) // 24)] for un, fr in ufr.items()}
        ranks = {}
        for i in free:
            t = self.T[i]
            res = []
            for v in values:
                with torch.no_grad():
                    t.consts["lean"].fill_(v)
                self.reanchor([i])
                sc = self.score(ufr[t.plan.unit], 320, 180, kind="blur")
                res.append((sc, v))
                self.log(f"[lean {t.plan.id}] {v:6.1f}：粗尺度光度 {sc:.4f}")
            # 试拟合的假设：光度排名第一，再加两个与已选值相差 ≥ 10 的（各取光度最好的），保证假设分散——
            # 光度项在 clip / 近景镜头里分不清前倾（s25 实测：光度偏好 0 / −10，正确的是约 −26）
            order = [v for _, v in sorted(res)]
            pick = [order[0]]
            for v in order[1:]:
                if len(pick) < self.lean_hyp and all(abs(v - q) >= 10 for q in pick):
                    pick.append(v)
            ranks[i] = pick
        snap0 = [x.snapshot() for x in self.T]
        trials = []
        for k in range(max(len(r) for r in ranks.values()) if trial_outer > 0 else 1):
            for x, sn in zip(self.T, snap0):
                x.restore(sn)
            for i in free:
                with torch.no_grad():
                    self.T[i].consts["lean"].fill_(ranks[i][min(k, len(ranks[i]) - 1)])
            self.reanchor(free)
            if trial_outer > 0:
                self.run_flow_stage(st_trial, trial_outer)
            q = {un: self.score(ufr[un], 640, 360, kind="ssim") for un in units}
            self.log(f"[lean] 试拟合 {k}：" + "；".join(f"{self.T[i].plan.id} {ranks[i][min(k, len(ranks[i]) - 1)]:.0f}→{float(self.T[i].consts['lean']):.2f}（1−SSIM {q[self.T[i].plan.unit]:.4f}）" for i in free))
            trials.append((q, [x.snapshot() for x in self.T]))
        for j, x in enumerate(self.T):
            un = x.plan.unit
            if un in units:
                kbest = min(range(len(trials)), key=lambda k: trials[k][0][un])
            else:
                kbest = 0
            x.restore(trials[kbest][1][j])
        for i in free:
            self.log(f"[lean {self.T[i].plan.id}] 选定 lean {float(self.T[i].consts['lean']):.2f}")

    # ------------------------------------------------------------------ 开关切换帧
    def estimate_switches(self, radius: int = 6, fades=(0, 3, 6, 9, 12, 18, 30), margin: float = 0.001, extra=()) -> list[dict]:
        """开关切换（用户决定由拟合器估计）：shots 标注的切换帧 est 与渐变时长 fade 只作初值。
        同一图层上标注在同一时刻（±1 帧）的切换编成一组一起估计：它们是同一个动作（如 s26 2530 的 key5 / key7 / key12，
        或 Movement* 连同配套的手臂参数），单独移动其中一个常常看不出差别（key5 的部件在 key12 = 1 时被藏起来），
        逐个估计会得到平坦的得分。组内各参数一起平移起点（保持标注的相对时刻），fade 各自保持标注值；
        只有一个参数的组才另外搜 fade（如 s38 的 Clothes2）。
        坐标搜索：在 [est − radius, est + radius] 内逐帧搜平移与（单参数组）在 fades 里搜渐变时长交替进行，直到不再变化。
        每个候选整段前向（切换参数可能经物理影响后续帧），数据项 = 受影响显示帧上、该图层固定数据区（切换前后两种外形都画到的地方，
        见下面 switch_mask）里的 SSIM@360 口径。
        最好的候选比标注只好不到 margin 时保留标注值（画面上分辨不出，或者主要变化落在排除区里，如 s26 2530 的白色笔刷）。"""
        out = []
        for li, t in enumerate(self.T):
            sws = sorted(t.plan.switches, key=lambda x: x["est"])
            groups = []
            for sw in sws:
                if groups and abs(sw["est"] - groups[-1][-1]["est"]) <= 1:
                    groups[-1].append(sw)
                else:
                    groups.append([sw])
            for grp in groups:
                e0 = [x["est"] for x in grp]
                f0 = [x["fade"] for x in grp]
                single = len(grp) == 1
                ref_e = e0[0]
                lo, hi = ref_e - radius, ref_e + radius
                frames = [f for f in self.F_all if lo <= f < hi + max(max(fades), max(f0)) + 1 and self.active(t, f)]
                if not frames:
                    continue
                cache = {}
                self.score(frames)
                reuse = {k: v for k, v in enumerate(self._last_per) if k != li}  # 其他图层不随候选变化，只画一次
                def setp(dc, fd):
                    for x, e, fa in zip(grp, e0, f0):
                        x["est"] = e + dc
                        x["fade"] = fd if single else fa
                # 数据区（不随候选变）：switch_mask = intersect（缺省）时取「整窗都是切换前状态」与「整窗都是切换后状态」两种渲染里
                # 该图层输出 α 都 > 0.5 的地方（再腐蚀 1px@360），在这里任何候选都画了东西，比较的只是画的内容；
                # union_grey 时取两者之并（α > 0.1，外扩 6px），候选合成到中灰底。
                # 上一版用「标注状态的覆盖 × 候选自己的 α」并合成到参考帧：切换改变图层外形时（s38.side 的 Q 版小一圈），没画到的地方
                # 等于参考帧本身，越早切成小的 Q 版越占便宜（估计 3345/fade 0）；并集 + 中灰底也不中立——中灰对白底、白围裙这类
                # 平坦区的 SSIM 不低，同样偏向少画（估计顶到窗口下界 3340/fade 0）。交集口径没有这两种偏向。
                with torch.no_grad():
                    st_a = []
                    for dc in (10 ** 6, -10 ** 6):
                        setp(dc, f0[0])
                        self.score(frames, reuse=reuse)
                        st_a.append(torch.stack([self._last_outa[li].get(i, torch.zeros(360, 640, device=self.dev)) for i in range(len(frames))]))
                    if self.switch_mask == "intersect":
                        wm = -F.max_pool2d(-(torch.minimum(*st_a) > 0.5).float()[:, None], 3, 1, 1)[:, 0]
                    else:
                        wm = F.max_pool2d((torch.maximum(*st_a) > 0.1).float()[:, None], 13, 1, 6)[:, 0]
                    setp(0, f0[0])

                def ev(dc, fd):
                    if (dc, fd) not in cache:
                        setp(dc, fd)
                        cache[(dc, fd)] = self.score(frames, reuse=reuse, wmask=wm, bg_grey=self.switch_mask != "intersect")
                    return cache[(dc, fd)]
                fa0 = f0[0] if single else None
                shifts = [d for d in range(-radius, radius + 1) if t.plan.fin <= ref_e + d < t.plan.fout]
                # 坐标搜索：起点（整窗）与 fade 交替，直到不再变化（最多 4 轮；候选有缓存）。
                # 上一版是「起点 → fade → 起点 ±2」一轮，长 fade 会把起点拖早（s38.side：3346/18，而网格最小在 3347/6）
                d_best, f_best = 0, fa0
                for _ in range(4):
                    d_new = min(shifts, key=lambda d: ev(d, f_best))
                    f_new = min(sorted(set(fades) | {fa0}), key=lambda fd: ev(d_new, fd)) if single else None
                    if (d_new, f_new) == (d_best, f_best):
                        break
                    d_best, f_best = d_new, f_new
                for dc, fd in extra:  # 诊断：额外评估的候选（只进 scores，不参与选择）
                    ev(dc, fd if single else None)
                if ev(d_best, f_best) > ev(0, fa0) - margin:
                    d_best, f_best = 0, fa0
                setp(d_best, f_best)
                names = "+".join(x["param"] for x in grp)
                self.log(f"[开关 {t.plan.id}] {names}：标注 {ref_e}" + (f"/fade {fa0}" if single else f"（fade {f0}）")
                         + f"（1−SSIM {ev(0, fa0):.4f}） → 估计 {ref_e + d_best}" + (f"/fade {f_best}" if single else "")
                         + f"（{ev(d_best, f_best):.4f}；{len(frames)} 帧，{len(cache)} 个候选；得分极差 {max(cache.values()) - min(cache.values()):.4f}）")
                for x, e, fa in zip(grp, e0, f0):
                    out.append({"layer": t.plan.id, "param": x["param"], "group": names, "shots": [e, fa], "est": [x["est"], x["fade"]],
                                "scores": {f"{ref_e + d}/{fd}": v for (d, fd), v in cache.items()}})
        return out

    # ------------------------------------------------------------------ 外观常量
    def fit_appearance(self, n_frames: int = 8, iters: int = 20, resamples=(1.0, 0.7)) -> dict:
        """blur σ、dofK、颜色滤镜、resample。几何已定：1280×720 上把选中显示帧的各图层缓冲（滤镜之前）渲染一次缓存，
        之后只对滤镜求导。数据项 = metric 口径 SSIM@360（合成到参考帧 → 2×2 区域平均 = INTER_AREA → 遮罩内 SSIM）。
        resample 在 A3 里不可导，各图层共用一个值逐个优化后取最好。帧：优先普通帧，均匀取；有景深段的图层保证取到段内帧。"""
        u = self.u
        normal = [f for f in self.F_all if u.cls[f] == "normal"]
        pool = normal if len(normal) >= n_frames else self.F_all
        frames = [pool[i] for i in np.linspace(0, len(pool) - 1, min(n_frames, len(pool))).astype(int)]
        for t in self.T:
            if t.has_dof:
                dof_fr = [f for f in self.F_all if self.active(t, f) and t.plan.at("dof", t.plan.hold(f)) is not None]
                frames += [dof_fr[i] for i in np.linspace(0, len(dof_fr) - 1, min(4, len(dof_fr))).astype(int)] if dof_fr else []
        frames = sorted(set(frames))
        t0 = time.time()
        with torch.no_grad():
            afters = physics_batch(self.T)
            bufs = []
            for t, after in zip(self.T, afters):
                d = {}
                act = [f for f in frames if self.active(t, f)]
                gs = sorted({t.plan.hold(f) for f in act})
                if gs:
                    P = t.final_params(after, gs)
                    TS = t.to_screen(t.tf_at(gs), 1.0, gs)
                    from .render import raster_layer
                    for a in range(0, len(gs), 2):
                        b, _ = raster_layer(t.rig, P[a:a + 2], TS[a:a + 2].float(), 1280, 720, aa=False, quantize=True)
                        for j, g in enumerate(gs[a:a + 2]):
                            d[g] = b[j]
                bufs.append(d)
            ref, excl = self.refs.at(1280, 720, frames)
            R = torch.tensor(ref, device=self.dev)
            E = torch.tensor(excl, device=self.dev)
            R360 = F.avg_pool2d(R.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        self._tick("render", t0)

        def comp_all(rs):
            out = []
            for i, f in enumerate(frames):
                comp = torch.zeros(1, 720, 1280, 4, device=self.dev)
                for t, d in zip(self.T, bufs):
                    if not self.active(t, f):
                        continue
                    g = t.plan.hold(f)
                    src = C.apply_filters(d[g][None], t.filters_at(g, 1.0, rs), quantize=False)
                    o = t.out_at(g, 1.0)
                    a = o["opacity"] * (C.clip_coverage(o["clip"], 1, 720, 1280, src) if o["clip"] else 1.0)
                    comp = C.blend(comp, src * a, "normal")
                out.append(comp)
            return torch.cat(out)

        with torch.no_grad():
            m = metric_mask(comp_all(1.0)[..., 3], E, 1.0)
            m360 = F.avg_pool2d(m[:, None], 2)[:, 0] > 0.5
        keys = ["blur", "dofK", "brightness", "contrast", "saturation", "multiply"]
        # 滤镜的卷积反向（cuDNN）缺省不是逐次确定的；这一段很便宜，打开确定性算子（几何阶段不需要，见 _canon_grads）
        det0 = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        init = [{n: t.consts[n].detach().clone() for n in keys} for t in self.T]
        best = None
        t0 = time.time()
        for rs in resamples:
            with torch.no_grad():
                for t, ini in zip(self.T, init):
                    for n in keys:
                        t.consts[n].copy_(ini[n])
            params = [{"params": [t.consts["blur"]] + ([t.consts["dofK"]] if t.has_dof else []), "lr": 0.08} for t in self.T]
            params += [{"params": [t.consts[n] for n in keys[2:]], "lr": 0.01} for t in self.T if t.plan.color_filter]
            opt = torch.optim.Adam(params)
            for it in range(iters):
                opt.zero_grad(set_to_none=True)
                img = comp_all(rs)
                comp = img[..., :3] * 255 + R * (1 - img[..., 3:4])
                c360 = F.avg_pool2d(comp.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
                S = ssim_map(c360, R360)
                sv = (S * m360).sum((1, 2)) / m360.sum((1, 2)).clamp_min(1)
                loss = (1 - sv).mean()
                loss.backward()
                _canon_grads(opt)
                opt.step()
                for t in self.T:
                    t.project()
            val = float(sv.mean())
            self.log(f"[外观] resample {rs:.2f}：SSIM@360（{len(frames)} 帧，1280 渲染）{val:.4f}；"
                     + "；".join(f"{t.plan.id} blur {float(t.consts['blur']):.2f}" + (f" dofK {float(t.consts['dofK']):.2f}" if t.has_dof else "")
                                + (f" 对比 {float(t.consts['contrast']):.3f} 饱和 {float(t.consts['saturation']):.3f}" if t.plan.color_filter else "") for t in self.T))
            if best is None or val > best[0]:
                best = (val, rs, [{n: t.consts[n].detach().clone() for n in keys} for t in self.T])
        with torch.no_grad():
            for t, b in zip(self.T, best[2]):
                for n in keys:
                    t.consts[n].copy_(b[n])
                t.resample = best[1]
        torch.use_deterministic_algorithms(det0, warn_only=True)
        self._tick("appearance", t0)
        return {"ssim": best[0], "resample": best[1], "frames": frames}
