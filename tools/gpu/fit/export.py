"""导出：Traj → scene.json 图层（格式同 tools/fit/src/traj2.ts buildLayer 的产出）。

所有曲线都按帧烘焙（baked），值就是 GPU 前向在该帧用的数，导出不改变语义：
  - 变换：x、y、scale = exp(ln s)、rotation（翻牌时另有 scaleX），anchorX/Y = 参考姿态两眼虹膜中点；
  - clip（圆形开窗 / 画中画圆角矩形，区间外为覆盖全画面的大矩形）与不透明度：规划给的逐帧值（同 buildLayer）；
  - 参数轨（prePhysics set，顺序 = 前向里的写入顺序）：variant 常量、LeanInput、Clothes2、开关斜坡（估计后的切换帧）、全部 VTS 输出；
  - postPhysics add 轨：姿态输出与物理写入参数的偏移；
  - filters：color（规划要求时）→ resample → blur（有景深 / 模糊段时逐帧烘焙）；frameHold；physics：warmup 90、不 stabilize。
"""
from __future__ import annotations

import torch

from .plan import switch_value
from .traj import WARMUP, Traj


def build_layer(t: Traj) -> dict:
    p = t.plan
    fr = p.frames
    bk = lambda vals: {"baked": {"start": p.fin, "values": [float(v) for v in vals]}}
    with torch.no_grad():
        tf = t.tf_at(fr).cpu()
        tv = t.track_values()
        post = t.post_offsets(fr).cpu()
        sig = [float(t.sigma_at(g)) for g in fr]
    k = t.consts
    prov = {"source": "fit", "tool": "@charaforge/gpu-fit (A5)", "detail": {"unit": p.id, "what": "VTS forward model: face-tracking inputs (fitted per new frame) → mapping + smoothing"}}
    params = []
    for key, v in p.constants.items():
        if key not in ("Clothes2", "LeanInput"):
            params.append({"id": key, "curve": {"keys": [{"t": p.fin, "v": v}]}})
    params.append({"id": "LeanInput", "curve": bk([float(k["lean"])] * len(fr))})
    params.append({"id": "Clothes2", "curve": bk([t.clothes2] * len(fr))})
    for sw in p.switches:
        params.append({"id": sw["param"], "curve": bk([switch_value(sw, g) for g in fr])})
    # VTS 输出写在常量与开关之后（与前向 track_values 的覆盖顺序相同）
    for pid in t.vts.outputs:
        if pid in tv:
            params.append({"id": pid, "curve": bk(tv[pid].cpu().tolist()), "provenance": prov})
    pids = t.post_pid.tolist()
    for j, pi in enumerate(pids):
        params.append({"id": t.rig.param_ids[pi], "stage": "postPhysics", "mode": "add", "curve": bk(post[:, j].tolist()),
                       "provenance": {**prov, "detail": {**prov["detail"], "what": "postPhysics add offset, 2nd-difference regularized"}}})
    filters = []
    if p.color_filter:
        filters.append({"type": "color", "brightness": float(k["brightness"]), "contrast": float(k["contrast"]), "saturation": float(k["saturation"]),
                        "multiply": [float(x) for x in k["multiply"].tolist()]})
    filters.append({"type": "resample", "factor": float(t.resample), "method": "bilinear"})
    filters.append({"type": "blur", "sigma": bk(sig) if t.has_dof else sig[0]})
    transform = {"x": bk(tf[:, 0].tolist()), "y": bk(tf[:, 1].tolist()), "scale": bk(torch.exp(tf[:, 2]).tolist())}
    if p.scale_x is not None:
        transform["scaleX"] = bk(p.scale_x)
    transform.update({"rotation": bk(tf[:, 3].tolist()), "anchorX": t.anchorX, "anchorY": t.anchorY})
    extra = {}
    if p.clip is not None:
        extra["clip"] = {kk: (bk(p.clip[kk]) if isinstance(p.clip[kk], list) else p.clip[kk]) for kk in ("x", "y", "width", "height", "radius", "feather")}
    if p.opacity is not None:
        extra["opacity"] = bk(p.opacity)
    return {
        "id": p.id, "kind": "model", "model": "dsc", "in": p.fin, "out": p.fout,
        "transform": transform, **extra, "filters": filters,
        **({"frameHold": p.holds} if p.holds else {}),
        "physics": {"warmup": WARMUP, "stabilize": False},
        "params": params,
        "provenance": {"source": "fit", "tool": "@charaforge/gpu-fit (A5)", "detail": {
            "instance": p.id, "fittedFrames": len(t.nodes),
            "method": "GPU differentiable Live2D (A1 deform + A2 physics + A3 raster); flow-correspondence Adam with 2nd-diff regularization; appearance on SSIM@360",
            "switches": [{"param": sw["param"], "est": sw["est"], "fade": sw["fade"]} for sw in p.switches],
            "vts": {"smoothC": t.c, "decouple": any(s["input"].startswith("Body") for s in t.vts.settings)},
        }},
    }
