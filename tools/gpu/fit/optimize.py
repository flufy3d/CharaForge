"""分阶段、由粗到细的整段联合优化（Adam）。

一次迭代：
  1. 整段前向到「要显示的帧」的最终参数 P_g 与 to_screen（含 VTS、物理 BPTT、postPhysics、变换插值），保留计算图；
  2. 把 P_g、to_screen 截成叶子，分块（显存）渲染、算数据项（SSIM + 边缘），逐块反向，梯度累加到这两个叶子与外观常量上；
  3. 用累加的梯度把整段图反向一次（同时加上锚点项、时间正则），Adam 一步，投影回可行域。
数据项按显示帧计：显示帧 f 的画面 = 帧 g = hold(f) 的渲染，与参考帧 f 比较（frameHold 段内一个渲染对应两帧参考）。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .loss import blur_l1, edge_loss, flow_loss, masked_ssim
from .render import compose_one
from .traj import Traj, interp_matrix


def compose_one_draw(buf, draw):
    return compose_one(buf, draw, quantize=False)


def interp_rows(xs, frames):
    return interp_matrix(xs, frames).numpy()


@dataclass
class Stage:
    name: str
    iters: int
    W: int
    H: int
    lr: dict
    w_ssim: float = 1.0
    w_edge: float = 0.0
    w_blur: float = 0.0                # 粗尺度光度项（blur_l1 / 20，各 σ 平均）
    blur_sig: tuple = (8.0,)           # 720p 像素
    w_flow: float = 0.0                # 光流项（Cauchy，尺度 flow_c 个 720p 像素）
    flow_c: float = 8.0
    edge_sigma: tuple = (2.0, 2.0)     # 起止（线性退火），单位：当前分辨率像素
    w_anchor: float = 0.0
    reg: dict = field(default_factory=dict)
    node_stride: int = 1               # 只渲染每 k 个节点（粗阶段省时间）
    chunk: int = 4
    aa: bool = True
    aa_layers: int | None = None
    lr_final: float = 0.3              # 余弦退火到 lr × lr_final
    outer: int = 0                     # > 0：光流对应阶段（run_flow_stage），外层渲染次数
    inner: int = 0                     # 每次渲染之后的内层步数（只跑物理 + 变形）


DEFAULT_REG = {"sd_inputs": 0.02, "sd_tf": 0.02, "sd_post": 0.3, "mag_post": 0.05, "speed": 1.0}


class Fitter:
    def __init__(self, traj: Traj, refs, log=print, effect_weight: float = 0.5):
        self.t, self.refs, self.log = traj, refs, log
        p = traj.plan
        self.effect_weight = effect_weight
        # 显示帧 → 求值帧
        self.disp = {}
        for f in p.frames:
            self.disp.setdefault(p.hold(f), []).append(f)
        self.wf = {f: (effect_weight if p.cls[f] == "effect" else 1.0) for f in p.frames}
        self._ref_gpu: dict = {}
        self.timing: dict[str, float] = {}
        self.history: list[dict] = []

    def ref_at(self, W: int, H: int):
        k = (W, H)
        if k not in self._ref_gpu:
            r, e = self.refs.at(W, H)
            self._ref_gpu = {k: (torch.tensor(r, device=self.t.dev), torch.tensor(e, device=self.t.dev))}
        return self._ref_gpu[k]

    # ------------------------------------------------------------------ 初值（只用 shots 锚点）
    def init_from_anchors(self) -> None:
        """变换关键帧初值：shots 眼睛采样（含低置信度，线性插值）给出两眼中点、眼距、连线角度 → x, y, ln s, rot（参考姿态下的刚体对齐）。"""
        t, p = self.t, self.t.plan
        fr = np.array([f for f, _ in p.eye_init], float)
        E = np.stack([e for _, e in p.eye_init])  # [M, 2, 2]
        tf = []
        for k in t.keys:
            e = np.array([[np.interp(k, fr, E[:, i, j]) for j in range(2)] for i in range(2)])
            mid = e.mean(0)
            d = e[1] - e[0]
            tf.append([mid[0], mid[1], math.log(np.hypot(*d) / t.eye_d0), math.degrees(math.atan2(d[1], d[0]))])
        t.set_tf(torch.tensor(tf, device=t.dev, dtype=torch.float64))

    def reanchor(self, lam: float = 0.05) -> None:
        """按当前姿态重新求变换关键帧：每个 shots 眼睛采样帧上，用该帧实际的虹膜网格质心（画布像素，含物理与偏移）
        与 shots 两眼求相似变换（两点精确），得到 (x, y, ln s, rot)，再对关键帧做带二阶差分正则的加权最小二乘。
        前倾、身体等姿态改变后虹膜在画布上的位置会变，刚体初值要跟着重算。"""
        t, p = self.t, self.t.plan
        fs = [f for f, _ in p.eye_init]
        gs = [p.hold(f) for f in fs]
        conf = {f: 0.3 for f in fs}
        for f, _, c in p.anchors:
            conf[f] = c
        with torch.no_grad():
            after = t.physics(t.pre_params())
            P = t.final_params(after, gs)
            c = t.rig.iris_centers(t.rig.deform(P).verts).double().cpu().numpy()  # [N, 2, 2]
        rows = []
        for (f, e), cc in zip(p.eye_init, c):
            de, dc = e[1] - e[0], cc[1] - cc[0]
            s = np.hypot(*de) / np.hypot(*dc)
            rot = math.atan2(de[1], de[0]) - math.atan2(dc[1], dc[0])
            R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            xy = e.mean(0) - s * R @ (cc.mean(0) - np.array([t.anchorX, t.anchorY]))
            rows.append([xy[0], xy[1], math.log(s), math.degrees(rot)])
        Y = np.array(rows)
        A = interp_rows(t.keys, gs)
        w = np.array([conf[f] for f in fs])
        n = len(t.keys)
        D = np.zeros((max(0, n - 2), n))
        for i in range(n - 2):
            D[i, i:i + 3] = [1, -2, 1]
        M = A.T @ (w[:, None] * A) + lam * w.sum() / max(1, n) * D.T @ D + 1e-9 * np.eye(n)
        X = np.linalg.solve(M, A.T @ (w[:, None] * Y))
        t.set_tf(torch.tensor(X, device=t.dev, dtype=torch.float64))

    # ------------------------------------------------------------------ 一次评估 / 梯度
    def evaluate(self, st: Stage, grad: bool, gs: list[int] | None = None, resample=None) -> dict:
        t = self.t
        s = st.W / 1280.0
        ref, excl = self.ref_at(st.W, st.H)
        if gs is None:
            gs = t.nodes[::st.node_stride]
            if t.nodes[-1] not in gs:
                gs = gs + [t.nodes[-1]]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        out = {"gs": gs}
        with ctx:
            t0 = time.time()
            pre = t.pre_params()
            after = t.physics(pre)
            P = t.final_params(after, gs)
            TS = t.to_screen(t.tf_at(gs), s)
            self._tick("physics", t0)
            Pl = P.detach().requires_grad_(grad)
            TSl = TS.detach().requires_grad_(grad)
            # 数据项权重：每个求值帧 g 对应的显示帧 f（参考帧 f），权重总和归一
            wsum = sum(self.wf[f] for g in gs for f in self.disp[g])
            ssim_f, edge_f, data = {}, {}, 0.0
            self._flow = {}
            sig = st.edge_sigma[0] + (st.edge_sigma[1] - st.edge_sigma[0]) * getattr(self, "_progress", 0.0)
            t0 = time.time()
            for i in range(0, len(gs), st.chunk):
                cg = gs[i:i + st.chunk]
                img, dout = t.render(Pl[i:i + st.chunk], TSl[i:i + st.chunk], s, st.W, st.H, aa=st.aa and grad, resample=resample, aa_layers=st.aa_layers)
                # 展开到显示帧
                rows, fs = [], []
                for j, g in enumerate(cg):
                    for f in self.disp[g]:
                        rows.append(j)
                        fs.append(f)
                ix = torch.tensor([f - t.plan.fin for f in fs], device=t.dev)
                imgs = img[rows]
                S, _ = masked_ssim(imgs, ref[ix], excl[ix], s)
                w = torch.tensor([self.wf[f] for f in fs], device=t.dev, dtype=S.dtype) / wsum
                lc = st.w_ssim * ((1 - S) * w).sum()
                if st.w_blur > 0:
                    Bl = sum(blur_l1(imgs, ref[ix], excl[ix], sg, s) for sg in st.blur_sig) / (20.0 * len(st.blur_sig))
                    lc = lc + st.w_blur * (Bl * w).sum()
                if st.w_flow > 0:
                    T = TSl[i:i + st.chunk].float()
                    vs = torch.einsum("bij,bvj->bvi", T[:, :, :2], dout.verts) + T[:, None, :, 2]
                    vd = t.rig.raster.geom.vtx_draw
                    valid = (dout.visible & (dout.opacity > 0.05))[:, vd]
                    gref = ref[torch.tensor([g - t.plan.fin for g in cg], device=t.dev)]
                    Fl, cnt = flow_loss(vs, valid, img, gref, st.flow_c * s)
                    wg = torch.tensor([sum(self.wf[f] for f in self.disp[g]) for g in cg], device=t.dev, dtype=Fl.dtype) / wsum
                    lc = lc + st.w_flow * (Fl * wg).sum()
                    for g, v, n in zip(cg, Fl.detach().tolist(), cnt):
                        self._flow[g] = (v, n)
                if st.w_edge > 0:
                    E = edge_loss(imgs, ref[ix], excl[ix], sig, s)
                    lc = lc + st.w_edge * (E * w).sum()
                    for f, v in zip(fs, E.detach().tolist()):
                        edge_f[f] = v
                for f, v in zip(fs, S.detach().tolist()):
                    ssim_f[f] = v
                if grad:
                    lc.backward()
                data += float(lc)
                del img, imgs
            self._tick("render", t0)
            t0 = time.time()
            tot = torch.zeros((), device=t.dev, dtype=torch.float64)
            terms = {}
            if st.w_anchor > 0 and t.anchor_xy is not None:
                pr = t.anchor_pred(t.anchor_frames, after)
                d = (pr - t.anchor_xy).norm(dim=-1)  # [A, 2]
                ed = (t.anchor_xy[:, 1] - t.anchor_xy[:, 0]).norm(dim=-1)
                c = torch.clamp(0.15 * ed, min=10.0)[:, None]
                rho = torch.log1p((d / c) ** 2)
                la = (rho * t.anchor_w[:, None]).sum() / (2 * t.anchor_w.sum())
                tot = tot + st.w_anchor * la
                terms["anchor"] = float(la)
                terms["anchor_px"] = float(d.mean())
            reg = t.reg_terms()
            regw = {**DEFAULT_REG, **st.reg}
            for k, v in reg.items():
                tot = tot + regw.get(k, 0.0) * v
                terms[k] = float(v)
            if grad:
                (tot + (P * Pl.grad).sum() + (TS * TSl.grad).sum()).backward()
            self._tick("backward", t0)
        if self._flow:
            terms["flow"] = float(np.mean([v for v, _ in self._flow.values()]))
            terms["flow_n"] = float(np.mean([n for _, n in self._flow.values()]))
        out.update(data=data, total=data + float(tot), ssim=ssim_f, edge=edge_f, terms=terms)
        return out

    def _tick(self, k, t0):
        torch.cuda.synchronize()
        self.timing[k] = self.timing.get(k, 0.0) + time.time() - t0

    # ------------------------------------------------------------------ 阶段
    def run_stage(self, st: Stage) -> dict:
        t = self.t
        groups = t.param_groups(st.lr)
        opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
        base = [g["lr"] for g in opt.param_groups]
        t_stage = time.time()
        last = None
        for it in range(st.iters):
            self._progress = it / max(1, st.iters - 1)
            k = 0.5 * (1 + math.cos(math.pi * self._progress))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * (st.lr_final + (1 - st.lr_final) * k)
            opt.zero_grad(set_to_none=True)
            for v in t.consts.values():
                v.grad = None
            r = self.evaluate(st, grad=True)
            opt.step()
            t.project()
            ms = np.mean(list(r["ssim"].values()))
            rec = {"stage": st.name, "it": it, "data": r["data"], "total": r["total"], "ssim": float(ms), **r["terms"],
                   "lean": float(t.consts["lean"]), "blur": float(t.consts["blur"]), "sec": time.time() - t_stage}
            self.history.append(rec)
            if it % 5 == 0 or it == st.iters - 1:
                self.log(f"[{st.name}] {it:3d} 数据 {r['data']:.4f} 总 {r['total']:.4f} SSIM {ms:.4f} "
                         + " ".join(f"{k} {v:.3g}" for k, v in r["terms"].items())
                         + f" lean {float(t.consts['lean']):.2f} blur {float(t.consts['blur']):.2f} ({time.time() - t_stage:.0f}s)")
            last = r
        self._progress = 0.0
        return last

    # ------------------------------------------------------------------ 光流对应阶段（外层渲染 + 光流，内层只做物理 + 变形）
    def flow_targets(self, st: Stage, gs: list[int]) -> dict:
        """渲染（不求导）当前解，算合成图 → 参考帧的 DIS 光流（前后向一致性检查），得到每帧有效顶点的目标位置（720p 像素）。"""
        from .loss import _sample, composite_on_ref, dis_flows
        t = self.t
        s = st.W / 1280.0
        ref, excl = self.ref_at(st.W, st.H)
        tg, ssim = {}, {}
        t0 = time.time()
        with torch.no_grad():
            after = t.physics(t.pre_params())
            P = t.final_params(after, gs)
            TS = t.to_screen(t.tf_at(gs), s)
            for i in range(0, len(gs), st.chunk):
                cg = gs[i:i + st.chunk]
                img, dout = t.render(P[i:i + st.chunk], TS[i:i + st.chunk], s, st.W, st.H, aa=False)
                ix = torch.tensor([g - t.plan.fin for g in cg], device=t.dev)
                S, _ = masked_ssim(img, ref[ix], excl[ix], s)
                T = TS[i:i + st.chunk].float()
                vs = torch.einsum("bij,bvj->bvi", T[:, :, :2], dout.verts) + T[:, None, :, 2]
                valid = (dout.visible & (dout.opacity > 0.05))[:, t.rig.raster.geom.vtx_draw]
                flows = dis_flows(composite_on_ref(img, ref[ix]), ref[ix])
                H, W = st.H, st.W
                for j, g in enumerate(cg):
                    fw, bw = (torch.tensor(x, device=t.dev) for x in flows[j])
                    p = vs[j]
                    f1 = _sample(fw, p)
                    f2 = _sample(bw, p + f1)
                    a = _sample(img[j, ..., 3:4], p)[:, 0]
                    ok = valid[j] & ((f1 + f2).norm(dim=-1) < 1.5) & (a > 0.5) & (p[:, 0] > 0) & (p[:, 1] > 0) & (p[:, 0] < W) & (p[:, 1] < H)
                    vid = ok.nonzero()[:, 0]
                    tg[g] = (vid, ((p + f1)[vid] / s).double())
                    ssim[g] = float(S[j])
        self._tick("render", t0)
        return {"targets": tg, "ssim": ssim}

    def run_flow_stage(self, st: Stage, outer: int, inner: int) -> None:
        """外层 outer 次：渲染 + 光流定目标；内层 inner 次 Adam：只跑 VTS + 物理 + 变形（不光栅化），让顶点走向光流目标
        （Cauchy，尺度 flow_c 个 720p 像素）+ 锚点 + 时间正则。光栅化是最贵的一步，这样每次渲染换来 inner 步优化（同 fit4 的光流 GN 思路）。"""
        t = self.t
        gs = t.nodes[::st.node_stride]
        if t.nodes[-1] not in gs:
            gs = gs + [t.nodes[-1]]
        opt = torch.optim.Adam(t.param_groups(st.lr), betas=(0.9, 0.99))
        base = [g["lr"] for g in opt.param_groups]
        wg = torch.tensor([sum(self.wf[f] for f in self.disp[g]) for g in gs], device=t.dev, dtype=torch.float64)
        wg = wg / wg.sum()
        t_stage = time.time()
        regw = {**DEFAULT_REG, **st.reg}
        n = outer * inner
        for o in range(outer):
            ft = self.flow_targets(st, gs)
            ms = float(np.mean(list(ft["ssim"].values())))
            for k in range(inner):
                prog = (o * inner + k) / max(1, n - 1)
                kk = 0.5 * (1 + math.cos(math.pi * prog))
                for g_, b in zip(opt.param_groups, base):
                    g_["lr"] = b * (st.lr_final + (1 - st.lr_final) * kk)
                opt.zero_grad(set_to_none=True)
                t0 = time.time()
                after = t.physics(t.pre_params())
                P = t.final_params(after, gs)
                out = t.rig.deform(P)
                M = t.to_screen(t.tf_at(gs)).float()
                vs = torch.einsum("bij,bvj->bvi", M[:, :, :2], out.verts) + M[:, None, :, 2]
                lf = []
                for j, g in enumerate(gs):
                    vid, tgt = ft["targets"][g]
                    if vid.numel() == 0:
                        lf.append(vs.sum() * 0)
                        continue
                    d = (vs[j, vid].double() - tgt).norm(dim=-1)
                    lf.append(torch.log1p((d / st.flow_c) ** 2).mean())
                lflow = (torch.stack(lf) * wg).sum()
                tot = st.w_flow * lflow
                terms = {"flow": float(lflow)}
                if st.w_anchor > 0 and t.anchor_xy is not None:
                    pr = t.anchor_pred(t.anchor_frames, after)
                    d = (pr - t.anchor_xy).norm(dim=-1)
                    ed = (t.anchor_xy[:, 1] - t.anchor_xy[:, 0]).norm(dim=-1)
                    c = torch.clamp(0.15 * ed, min=10.0)[:, None]
                    la = (torch.log1p((d / c) ** 2) * t.anchor_w[:, None]).sum() / (2 * t.anchor_w.sum())
                    tot = tot + st.w_anchor * la
                    terms["anchor_px"] = float(d.mean())
                for kname, v in t.reg_terms().items():
                    tot = tot + regw.get(kname, 0.0) * v
                    terms[kname] = float(v)
                tot.backward()
                opt.step()
                t.project()
                self._tick("inner", t0)
                self.history.append({"stage": st.name, "outer": o, "inner": k, "total": float(tot), "ssim": ms, **terms,
                                     "lean": float(t.consts["lean"]), "sec": time.time() - t_stage})
            self.log(f"[{st.name}] 外层 {o:2d} SSIM(渲染时) {ms:.4f} 总 {float(tot):.4f} "
                     + " ".join(f"{k} {v:.3g}" for k, v in terms.items())
                     + f" lean {float(t.consts['lean']):.2f} ({time.time() - t_stage:.0f}s)")

    # ------------------------------------------------------------------ 外观常量（全分辨率、缓存图层缓冲）
    def fit_appearance(self, n_frames: int = 15, iters: int = 40, resamples=(1.0, 0.8, 0.7, 0.6, 0.5)) -> dict:
        """blur σ、颜色滤镜、resample 因子。几何已定：在 1280×720 上把选中帧的图层缓冲（滤镜之前）渲染一次缓存下来，
        之后只对滤镜求导——每步只是几次卷积，很便宜。数据项就是 metric.py 的 SSIM@360 口径
        （合成到参考帧 → 2×2 区域平均降到 640×360 = INTER_AREA → 遮罩内 SSIM）。resample 在 A3 里不可导，逐个值各自优化后取最好。
        帧：优先普通帧，均匀取 n_frames 个节点。"""
        import torch.nn.functional as F
        from .loss import metric_mask, ssim_map
        t, p = self.t, self.t.plan
        normal = [g for g in t.nodes if all(p.cls[f] == "normal" for f in self.disp[g])]
        pool = normal if len(normal) >= n_frames else t.nodes
        gs = [pool[i] for i in np.linspace(0, len(pool) - 1, min(n_frames, len(pool))).astype(int)]
        ref, excl = self.ref_at(1280, 720)
        ix = torch.tensor([g - p.fin for g in gs], device=t.dev)
        R, E = ref[ix], excl[ix]
        t0 = time.time()
        with torch.no_grad():
            after = t.physics(t.pre_params())
            P = t.final_params(after, gs)
            TS = t.to_screen(t.tf_at(gs), 1.0)
            bufs = []
            for i in range(0, len(gs), 2):
                from .render import raster_layer
                b, _ = raster_layer(t.rig, P[i:i + 2], TS[i:i + 2].float(), 1280, 720, aa=False, quantize=True)
                bufs.append(b)
            buf = torch.cat(bufs)
            del bufs
            # 遮罩（metric 口径）：dilate(α > 0.1, 半径 3) − 排除区 → 降采样 > 0.5（几何固定，滤镜对 α 的影响很小，取无滤镜的 α）
            m = metric_mask(buf[..., 3], E, 1.0)
            m360 = F.avg_pool2d(m[:, None], 2)[:, 0] > 0.5
            R360 = F.avg_pool2d(R.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        self._tick("render", t0)
        k = t.consts
        keys = ["blur", "brightness", "contrast", "saturation", "multiply"]
        init = {n: k[n].detach().clone() for n in keys}
        best = None
        t0 = time.time()
        for rs in resamples:
            with torch.no_grad():
                for n in keys:
                    k[n].copy_(init[n])
            opt = torch.optim.Adam([{"params": [k["blur"]], "lr": 0.08}, {"params": [k[n] for n in keys[1:]], "lr": 0.01}])
            for it in range(iters):
                opt.zero_grad(set_to_none=True)
                img = compose_one_draw(buf, t.draw(1.0, rs))
                comp = img[..., :3] * 255 + R * (1 - img[..., 3:4])
                c360 = F.avg_pool2d(comp.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
                S = ssim_map(c360, R360)
                s = (S * m360).sum((1, 2)) / m360.sum((1, 2)).clamp_min(1)
                loss = (1 - s).mean()
                loss.backward()
                opt.step()
                t.project()
            val = float(s.mean())
            self.log(f"[外观] resample {rs:.2f}：SSIM@360（{len(gs)} 帧，1280 渲染）{val:.4f}，blur {float(k['blur']):.2f}，"
                     f"亮度 {float(k['brightness']):.3f} 对比 {float(k['contrast']):.3f} 饱和 {float(k['saturation']):.3f} 乘 {[round(x, 3) for x in k['multiply'].tolist()]}")
            if best is None or val > best[0]:
                best = (val, rs, {n: k[n].detach().clone() for n in keys})
        with torch.no_grad():
            for n in keys:
                k[n].copy_(best[2][n])
        t.resample = best[1]
        self._tick("appearance", t0)
        return {"ssim": best[0], "resample": best[1], "frames": gs}

    def search_lean(self, st: Stage, values=(0.0, -10.0, -20.0, -30.0)) -> float:
        """前倾（镜头常量）多假设：各自评估一次数据项，取最好（之后作为连续量继续优化）。"""
        best = None
        for v in values:
            with torch.no_grad():
                self.t.consts["lean"].fill_(v)
            self.reanchor()
            r = self.evaluate(st, grad=False)
            self.log(f"[lean] {v:6.1f}：数据 {r['data']:.4f}")
            if best is None or r["data"] < best[1]:
                best = (v, r["data"])
        with torch.no_grad():
            self.t.consts["lean"].fill_(best[0])
        self.reanchor()
        return best[0]

    def search_resample(self, st: Stage, values=(1.0, 0.8, 0.7, 0.6, 0.5)) -> float:
        """resample 因子在 A3 里不可导（决定缩小尺寸），离散搜索。"""
        best = None
        for v in values:
            r = self.evaluate(st, grad=False, resample=v)
            self.log(f"[resample] {v:.2f}：数据 {r['data']:.4f}")
            if best is None or r["data"] < best[1]:
                best = (v, r["data"])
        self.t.resample = best[0]
        return best[0]
