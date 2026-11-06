"""导出：Traj → scene.json 图层（格式同 tools/fit/src/traj2.ts buildLayer 的产出）。

所有曲线都按帧烘焙（baked），值就是 GPU 前向在该帧用的数（稀疏关键帧之间的线性插值已经算好），所以导出不改变语义：
  - 变换：x、y、scale = exp(ln s)、rotation，anchorX/Y = 参考姿态两眼虹膜中点；
  - 参数轨（prePhysics set，顺序 = 前向里的写入顺序）：variant 常量、LeanInput、Clothes2、开关斜坡、全部 VTS 输出；
  - postPhysics add 轨：姿态输出与次级物理分组偏移（组内每个参数同值）；
  - filters：color → resample → blur（常量）；frameHold = 规划的 cadence 段；physics：warmup 90、不 stabilize。
"""
from __future__ import annotations

import torch

from .plan import switch_value
from .traj import WARMUP, Traj


def _f(x) -> float:
    return float(x)


def build_layer(t: Traj) -> dict:
    p = t.plan
    fr = p.frames
    bk = lambda vals: {"baked": {"start": p.fin, "values": [float(v) for v in vals]}}
    with torch.no_grad():
        tf = t.tf_at(fr).cpu()
        tv = t.track_values()
        post = t.post_offsets(fr).cpu()
    k = t.consts
    prov = {"source": "fit", "tool": "@charaforge/gpu-fit (A5)", "detail": {"shot": p.shot["id"], "what": "VTS forward model: face-tracking inputs (fitted per new frame) → mapping + smoothing"}}
    params = []
    for key, v in p.constants.items():
        if key not in ("Clothes2", "LeanInput"):
            params.append({"id": key, "curve": {"keys": [{"t": p.fin, "v": v}]}})
    params.append({"id": "LeanInput", "curve": bk([_f(k["lean"])] * len(fr))})
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
                       "provenance": {**prov, "detail": {**prov["detail"], "what": "postPhysics add offset (pose output / grouped secondary physics), 2nd-difference regularized"}}})
    color = {"type": "color", "brightness": _f(k["brightness"]), "contrast": _f(k["contrast"]), "saturation": _f(k["saturation"]),
             "multiply": [float(x) for x in k["multiply"].tolist()]}
    layer = {
        "id": p.id, "name": p.id.replace(".", " "), "kind": "model", "model": "dsc", "in": p.fin, "out": p.fout,
        "transform": {"x": bk(tf[:, 0].tolist()), "y": bk(tf[:, 1].tolist()), "scale": bk(torch.exp(tf[:, 2]).tolist()),
                      "rotation": bk(tf[:, 3].tolist()), "anchorX": t.anchorX, "anchorY": t.anchorY},
        "filters": [color, {"type": "resample", "factor": float(t.resample), "method": "bilinear"},
                    {"type": "blur", "sigma": max(0.0, _f(k["blur"]))}],
        **({"frameHold": p.holds} if p.holds else {}),
        "physics": {"warmup": WARMUP, "stabilize": False},
        "params": params,
        "provenance": {"source": "fit", "tool": "@charaforge/gpu-fit (A5)", "detail": {
            "shot": p.shot["id"], "instance": p.id, "fittedFrames": len(t.nodes),
            "method": "GPU differentiable Live2D (A1 deform + A2 physics + A3 raster); whole-trajectory Adam on SSIM@360 + edge + anchors + 2nd-diff regularization",
            "vts": {"smoothC": t.c, "decouple": any(s["input"].startswith("Body") for s in t.vts.settings)},
        }},
    }
    return layer
