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


DEFAULT_REG = {"sd_inputs": 0.02, "sd_tf": 0.02, "sd_post": 0.3, "mag_post": 0.05, "speed": 1.0}


class Fitter:
    def __init__(self, trajs: list[Traj], unit, refs, log=print, effect_weight: float = 0.5):
        self.T, self.u, self.refs, self.log = trajs, unit, refs, log
        self.dev = trajs[0].dev
        self.wf = {f: (effect_weight if unit.cls[f] == "effect" else 1.0) for f in unit.frames}
        # 渲染帧：至少是一个图层的节点（新画面）的显示帧
        self.F_all = sorted({f for t in trajs for f in t.nodes if unit.start <= f < unit.end})
        self.timing: dict[str, float] = {}
        self.history: list[dict] = []

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
    def score(self, frames: list[int], W=640, H=360, kind="ssim", sig=(16.0, 8.0), resample=None, reuse=None, wmask=None) -> float:
        """frames 上的数据项（越小越好）：kind = ssim → Σ w (1 − SSIM)；blur → 粗尺度光度项。"""
        s = W / 1280.0
        with torch.no_grad():
            comp, _, outa = self.render(frames, s, W, H, resample=resample, reuse=reuse)
            self._last_outa = outa
            ref, excl = self.refs.at(W, H, frames)
            ref, excl = torch.tensor(ref, device=self.dev), torch.tensor(excl, device=self.dev)
            w = torch.tensor([self.wf[f] for f in frames], device=self.dev)
            if kind == "ssim":
                if wmask is not None:  # 只在给定区域（如某个图层的覆盖范围）里算 metric 口径的 SSIM
                    m = metric_mask(comp[..., 3], excl, s) * wmask
                    S, _ = masked_ssim(comp, ref, excl, s, mask=m)
                else:
                    S, _ = masked_ssim(comp, ref, excl, s)
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
            S, _ = masked_ssim(comp, ref, excl, s)
            flows = dis_flows(composite_on_ref(comp, ref), ref)
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
                lf_sum, w_sum, terms = 0.0, 0.0, {}
                for li, (t, after) in enumerate(zip(self.T, afters)):
                    tgts = ft["targets"][li]
                    if tgts:
                        uniq = sorted({g for g, _, _ in tgts.values()})
                        P = t.final_params(after, uniq)
                        out = t.rig.deform(P)
                        M = t.to_screen(t.tf_at(uniq), 1.0, uniq).float()
                        vs = torch.einsum("bij,bvj->bvi", M[:, :, :2], out.verts.to(M.dtype)) + M[:, None, :, 2]
                        for f, (g, vid, tgt) in tgts.items():
                            if vid.numel() == 0:
                                continue
                            d = (vs[uniq.index(g), vid].double() - tgt).norm(dim=-1)
                            lf = torch.log1p((d / st.flow_c) ** 2).mean()
                            # 每个图层的权重按它在该帧的有效点占比（大图层、可见多的图层权重大）
                            wgt = self.wf[f] * float(vid.numel())
                            data = data + wgt * lf
                            lf_sum += wgt * float(lf)
                            w_sum += wgt
                    if st.w_anchor > 0 and t.anchor_xy is not None:
                        pr = t.anchor_pred(t.anchor_frames, after)
                        d = (pr - t.anchor_xy).norm(dim=-1)
                        ed = (t.anchor_xy[:, 1] - t.anchor_xy[:, 0]).norm(dim=-1)
                        c = torch.clamp(0.15 * ed, min=10.0)[:, None]
                        la = (torch.log1p((d / c) ** 2) * t.anchor_w[:, None]).sum() / (2 * t.anchor_w.sum())
                        tot = tot + st.w_anchor * la
                        terms[f"anchor_px[{t.plan.id}]"] = float(d.mean())
                    for kname, v in t.reg_terms().items():
                        tot = tot + regw.get(kname, 0.0) * v
                        terms[kname] = terms.get(kname, 0.0) + float(v)
                # 数据项按有效点数加权平均（归一的分母固定为外层开始时的值，保证内层各步的目标一致）
                if k == 0:
                    self._wnorm = max(w_sum, 1.0)
                tot = tot + data / self._wnorm
                tot.backward()
                _canon_grads(opt)
                opt.step()
                for t in self.T:
                    t.project()
                self._tick("inner", t0)
                terms["flow"] = lf_sum / max(w_sum, 1.0)
            self.history.append({"stage": st.name, "outer": o, "ssim": ms, **terms,
                                 "lean": [float(t.consts["lean"]) for t in self.T], "sec": time.time() - t_stage})
            self.log(f"[{st.name}] 外层 {o:2d}（{len(frames)} 帧）SSIM(渲染时) {ms:.4f} flow {terms['flow']:.3f} "
                     + " ".join(f"{k} {v:.3g}" for k, v in terms.items() if k != "flow")
                     + f" lean {[round(float(t.consts['lean']), 2) for t in self.T]} ({time.time() - t_stage:.0f}s)")
        return ms

    _wnorm = 1.0

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
        """规划没固定前倾的图层：先用粗尺度光度项（σ 16 / 8px，在 ±16px 内单调）比较各值（每个值先 reanchor），
        取前两名各跑 trial_outer 次光流粗对齐（同一初值），按渲染 SSIM 选定，再从选中的那一支继续。
        单一的光度项比较在 −20 与 −30 之间常常打平，直接取第一名会落进局部极小（s24 上 0.735 vs 0.79+）。"""
        free = [i for i, t in enumerate(self.T) if t.lean_free]
        if not free:
            return
        frames = self.F_all[::max(1, len(self.F_all) // 24)]
        for i in free:
            t = self.T[i]
            res = []
            for v in values:
                with torch.no_grad():
                    t.consts["lean"].fill_(v)
                self.reanchor([i])
                sc = self.score(frames, 320, 180, kind="blur")
                res.append((sc, v))
                self.log(f"[lean {t.plan.id}] {v:6.1f}：粗尺度光度 {sc:.4f}")
            res.sort()
            snap0 = [x.snapshot() for x in self.T]
            best = None
            for sc, v in res[:2]:
                for x, sn in zip(self.T, snap0):
                    x.restore(sn)
                with torch.no_grad():
                    t.consts["lean"].fill_(v)
                self.reanchor([i])
                self.run_flow_stage(st_trial, trial_outer)
                q = self.score(frames, 640, 360, kind="ssim")
                self.log(f"[lean {t.plan.id}] 假设 {v:6.1f} → 光流粗对齐 {trial_outer} 次后 1−SSIM {q:.4f}（lean {float(t.consts['lean']):.2f}）")
                if best is None or q < best[0]:
                    best = (q, v, [x.snapshot() for x in self.T])
            for x, sn in zip(self.T, best[2]):
                x.restore(sn)
            self.log(f"[lean {t.plan.id}] 选定假设 {best[1]}（当前 {float(t.consts['lean']):.2f}）")

    # ------------------------------------------------------------------ 开关切换帧
    def estimate_switches(self, radius: int = 6, fades=(0, 3, 6, 9, 12, 18, 30), margin: float = 0.001) -> list[dict]:
        """开关切换（用户决定由拟合器估计）：shots 标注的切换帧 est 与渐变时长 fade 只作初值。
        坐标搜索：先在 [est − radius, est + radius] 内逐帧搜起点（fade 取标注值），再在 fades 里搜渐变时长，最后起点 ±2 复核。
        每个候选整段前向（切换参数可能经物理影响后续帧），数据项 = 受影响显示帧上的 SSIM@360 口径（640×360 渲染，与 runtime
        对拍差 < 1e-3）。受影响的帧 = [est0 − radius, est0 + radius + 最大 fade] 里的渲染帧（新画面帧）。
        画面上分辨不出的切换（最好的候选比标注只好不到 margin）保留标注值。同一参数在多个图层上（如 s41 两个实例同一个
        Movement3）各自估计。"""
        out = []
        for t in self.T:
            for sw in t.plan.switches:
                est0, fade0 = sw["est"], sw["fade"]
                lo, hi = est0 - radius, est0 + radius
                frames = [f for f in self.F_all if lo <= f < hi + max(fades) + 1 and self.active(t, f)]
                if not frames:
                    continue
                cache = {}
                li = self.T.index(t)
                self.score(frames)
                reuse = {k: v for k, v in enumerate(self._last_per) if k != li}  # 其他图层不随候选变化，只画一次
                # 数据项只在该图层的覆盖范围（标注状态下输出 α > 0.1，外扩 6px@360）里算：别的实例不随候选变，只会稀释信号
                with torch.no_grad():
                    am = torch.stack([self._last_outa[li].get(i, torch.zeros(360, 640, device=self.dev)) for i in range(len(frames))])
                    wm = F.max_pool2d((am > 0.1).float()[:, None], 13, 1, 6)[:, 0]
                cache[(est0, fade0)] = self.score(frames, reuse=reuse, wmask=wm)

                def ev(c, fd):
                    if (c, fd) not in cache:
                        sw["est"], sw["fade"] = c, fd
                        cache[(c, fd)] = self.score(frames, reuse=reuse, wmask=wm)
                    return cache[(c, fd)]
                starts = [c for c in range(lo, hi + 1) if t.plan.fin <= c < t.plan.fout]
                c_best = min(starts, key=lambda c: ev(c, fade0))
                f_best = min(sorted(set(fades) | {fade0}), key=lambda fd: ev(c_best, fd))
                c_best = min([c for c in range(c_best - 2, c_best + 3) if t.plan.fin <= c < t.plan.fout], key=lambda c: ev(c, f_best))
                if ev(c_best, f_best) > ev(est0, fade0) - margin:
                    c_best, f_best = est0, fade0
                sw["est"], sw["fade"] = c_best, f_best
                self.log(f"[开关 {t.plan.id}] {sw['param']} {sw['from']}→{sw['to']}：标注 {est0}/fade {fade0}（1−SSIM {ev(est0, fade0):.4f}）"
                         f" → 估计 {c_best}/fade {f_best}（{ev(c_best, f_best):.4f}；{len(frames)} 帧，{len(cache)} 个候选）")
                sw["est"], sw["fade"] = c_best, f_best
                out.append({"layer": t.plan.id, "param": sw["param"], "shots": [est0, fade0], "est": [c_best, f_best],
                            "scores": {f"{c}/{fd}": v for (c, fd), v in cache.items()}})
        return out

    # ------------------------------------------------------------------ 外观常量
    def fit_appearance(self, n_frames: int = 12, iters: int = 30, resamples=(1.0, 0.8, 0.6)) -> dict:
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
