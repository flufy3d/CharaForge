"""场景图层的求值（numpy 参考实现，与 packages/scene/src/curve.ts、packages/runtime/src/evaluator.ts 同语义）。

用途：把任意 scene.json 图层（包括 A5 导出的图层）按规范 a～g 求成逐帧输入，交给 GPU 前向复算，
用来核对「导出 → runtime」与「GPU 前向」是否一致。只支持本项目拟合产出用到的特性：
参数轨（set/add/multiply、weight、range、prePhysics/postPhysics、keys/baked）、变换、filters、clip、opacity、frameHold、physics。
motion / expression 片段与部件不透明度轨不支持（遇到报错）。
"""
from __future__ import annotations

import math

import numpy as np
import torch


def cubic_bezier(x1, y1, x2, y2, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bx = lambda t: ((1 - 3 * x2 + 3 * x1) * t + (3 * x2 - 6 * x1)) * t * t + 3 * x1 * t
    by = lambda t: ((1 - 3 * y2 + 3 * y1) * t + (3 * y2 - 6 * y1)) * t * t + 3 * y1 * t
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if bx(mid) < x:
            lo = mid
        else:
            hi = mid
    return by((lo + hi) / 2)


def apply_ease(ease, u):
    if ease is None or ease == "linear":
        return u
    if ease == "step":
        return 0.0
    if ease == "smooth":
        return cubic_bezier(0.42, 0, 0.58, 1, u)
    return cubic_bezier(*ease, u)


def _lerp(a, b, w):
    if isinstance(a, (int, float)):
        return a + (b - a) * w
    return [x + (y - x) * w for x, y in zip(a, b)]


def is_curve(a) -> bool:
    return isinstance(a, dict) and ("keys" in a or "baked" in a)


def sample_curve(c: dict, f: float):
    if "baked" in c:
        s, v = c["baked"]["start"], c["baked"]["values"]
        # JS Math.round：floor(x + 0.5)
        i = min(max(int(math.floor(f - s + 0.5)), 0), len(v) - 1)
        return v[i]
    ks = c["keys"]
    if f < ks[0]["t"]:
        return ks[0]["v"]
    lo, hi = 0, len(ks) - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if ks[mid]["t"] <= f:
            lo = mid
        else:
            hi = mid - 1
    k0 = ks[lo]
    k1 = ks[lo + 1] if lo + 1 < len(ks) else None
    if k1 is None or k1["t"] == k0["t"]:
        return k0["v"]
    u = (f - k0["t"]) / (k1["t"] - k0["t"])
    return _lerp(k0["v"], k1["v"], apply_ease(k0.get("ease"), u))


def sample(a, f, fallback):
    if a is None:
        return fallback
    return sample_curve(a, f) if is_curve(a) else a


def hold_frame(layer: dict, f: int) -> int:
    """规范 4b：显示帧 f → 求值帧 g。"""
    for h in layer.get("frameHold") or []:
        if h["start"] <= f < h["end"]:
            return h["start"] + (f - h["start"]) // h["every"] * h["every"]
    return f


class LayerEval:
    """一个模型图层的逐帧求值（规范 a～g，不含 motion / expression / 部件不透明度）。"""

    def __init__(self, rig, layer: dict):
        if layer.get("motions") or layer.get("expressions") or layer.get("partOpacity"):
            raise NotImplementedError("motion / expression / partOpacity 不支持")
        self.rig, self.L = rig, layer
        ph = layer.get("physics") or {}
        if ph.get("stabilize"):
            raise NotImplementedError("physics.stabilize 未接入（A5 导出不用）")
        self.physics_on = ph.get("enabled", True)
        self.warmup = int(ph.get("warmup", 0))
        self.start = layer["in"] - self.warmup
        tr = []
        for t in layer.get("params") or []:
            if t["id"] not in rig.pindex:
                continue
            rg = t.get("range") or [layer["in"], layer["out"]]
            tr.append((rig.pindex[t["id"]], t, t.get("mode", "set"), t.get("stage", "prePhysics"), rg))
        self.pre = [x for x in tr if x[3] == "prePhysics"]
        self.post = [x for x in tr if x[3] == "postPhysics"]

    def _write(self, v: np.ndarray, i: int, x: float, mode: str, w: float) -> None:
        r = self.rig
        cur = float(v[i])
        if mode == "add":
            y = cur + x * w
        elif mode == "multiply":
            y = cur * (1 + (x - 1) * w)
        else:
            y = x if w == 1 else cur + (x - cur) * w
        if r.repeat[i]:
            lo, hi = float(r.pmin[i]), float(r.pmax[i])
            size = hi - lo
            if y > hi:
                o = math.fmod(y - hi, size)
                y = lo + o if not math.isnan(o) else hi
            if y < lo:
                o = math.fmod(lo - y, size)
                y = hi - o if not math.isnan(o) else lo
        else:
            y = min(max(y, float(np.float32(r.pmin[i]))), float(np.float32(r.pmax[i])))
        v[i] = np.float32(y)

    def pre_params(self, g_end: int) -> np.ndarray:
        """[T, P] float32：帧 start..g_end−1 的规范 a～d 结果（预热帧输入取帧 in）。"""
        r = self.rig
        T = g_end - self.start
        out = np.zeros((T, r.P), np.float32)
        for k in range(T):
            f = max(self.start + k, self.L["in"])
            v = r.pdef.astype(np.float32).copy()
            for i, t, mode, _, rg in self.pre:
                if f < rg[0] or f >= rg[1]:
                    continue
                self._write(v, i, float(sample_curve(t["curve"], f)), mode, float(sample(t.get("weight"), f, 1)))
            out[k] = v
        return out

    def apply_post(self, after: torch.Tensor, frames: list[int]) -> torch.Tensor:
        """物理之后的参数 [N, P]（帧 frames）→ 规范 f 之后。逐行做（numpy 精确语义）。"""
        a = after.detach().cpu().numpy().astype(np.float32).copy()
        for n, g in enumerate(frames):
            f = max(g, self.L["in"])
            for i, t, mode, _, rg in self.post:
                if f < rg[0] or f >= rg[1]:
                    continue
                self._write(a[n], i, float(sample_curve(t["curve"], f)), mode, float(sample(t.get("weight"), f, 1)))
        return torch.tensor(a, device=after.device)

    def rollout(self, g_end: int) -> torch.Tensor:
        """物理推进：返回 [T, P] 物理之后（未加 post 轨）的参数，帧 start..g_end−1。"""
        pre = torch.tensor(self.pre_params(g_end), device=self.rig.device)
        if not self.physics_on:
            return pre
        with torch.no_grad():
            _, out = self.rig.phys.rollout(pre[:, None].double())
        return out[:, 0].float()

    def params_at(self, frames: list[int]) -> torch.Tensor:
        """求值帧 g 列表 → [N, P] 最终参数（规范 g 之前）。"""
        g_end = max(frames) + 1
        post = self.rollout(g_end)
        sel = post[[g - self.start for g in frames]]
        return self.apply_post(sel, frames)

    def draw_at(self, g: int) -> dict:
        """帧 g 的图层变换与 draw 参数（filters 已采样为数值）。"""
        L, tr = self.L, self.L["transform"]
        s = lambda a, fb: float(sample(a, g, fb))
        cv = self.rig.static["canvas"]
        t = {
            "x": s(tr["x"], 0), "y": s(tr["y"], 0), "scale": s(tr["scale"], 1), "scaleX": s(tr.get("scaleX"), 1),
            "scaleY": s(tr.get("scaleY"), 1), "rotation": s(tr.get("rotation"), 0),
            "anchorX": tr.get("anchorX", cv["width"] / 2), "anchorY": tr.get("anchorY", cv["height"] / 2), "flipX": bool(tr.get("flipX", False)),
        }
        # 与 runtime.ts resolveLayerDraw 相同的缺省值
        filters = []
        for fl in L.get("filters") or []:
            if fl.get("enabled", True) is False:
                continue
            ty = fl["type"]
            if ty == "blur":
                filters.append({"type": "blur", "sigma": s(fl.get("sigma"), 0)})
            elif ty == "motionBlur":
                filters.append({"type": "motionBlur", "angle": s(fl.get("angle"), 0), "length": s(fl.get("length"), 0)})
            elif ty == "resample":
                filters.append({"type": "resample", "factor": s(fl.get("factor"), 1), "method": fl.get("method", "bilinear")})
            else:
                filters.append({"type": "color", "brightness": s(fl.get("brightness"), 0), "contrast": s(fl.get("contrast"), 1),
                                "saturation": s(fl.get("saturation"), 1), "multiply": list(sample(fl.get("multiply"), g, [1, 1, 1])),
                                "offset": list(sample(fl.get("offset"), g, [0, 0, 0]))})
        clip = None
        if L.get("clip"):
            c = L["clip"]
            clip = {k: s(c.get(k), 0) for k in ("x", "y", "width", "height", "radius", "feather")}
        return {"transform": t, "filters": filters, "clip": clip, "opacity": s(L.get("opacity"), 1),
                "blendMode": L.get("blendMode", "normal")}
