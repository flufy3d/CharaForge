"""镜头规划（只读 shots.json，冷启动）：图层区间、新画面节点（cadence）、变换关键帧、frameHold、锚点、排除区、帧类别、参考帧。

规则与 tools/fit/src/plans.ts（planAuto）、traj2.ts 一致的部分：
  - in = 轨迹里第一个有 bbox 的采样，out = 最后一个 bbox 之后第一个 bbox = null 的采样（没有则镜头末）；
  - 新画面：先做参考帧重复检测（shotsutil.ts detectNewFrames 的移植），再由 shots.json 的 cadence 段覆盖（段内只有对齐帧是新画面），
    frameHold 由节点间隔生成（finalize_nodes；make_plan 里先给只看 cadence 的初版）；
  - 变换关键帧：每 24 帧，zoom / pan / scale 镜头每 12 帧，最后一帧补一个；
  - 锚点：eyes 且 confidence ≥ 0.4；
  - 常量：variant 原样写成常量参数轨；switchEvents 写成斜坡（余弦淡入，同 traj2.buildLayer）。
不支持（遇到时报错，见 fit.md「推广到全片」）：多实例、clip、景深 blur 段、cardFlip、opacity 轨。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .rig import REPO

sys.path.insert(0, str(REPO / "tools/qa"))
import metric  # noqa: E402  （唯一的指标实现：参考帧解码、排除区、帧类别）

SHOTS = REPO / "scenes/deepseek-chan/shots.json"
ANCHOR_CONF = 0.4


@dataclass
class Plan:
    shot: dict
    inst: dict
    id: str
    fin: int
    fout: int
    nodes: list[int]                 # 新画面帧（升序，含 fin）
    keys: list[int]                  # 变换关键帧
    holds: list[dict]                # frameHold 段 {start, end, every}
    constants: dict[str, float]
    switches: list[dict]
    anchors: list[tuple[int, np.ndarray, float]]  # (帧, [[x,y],[x,y]] 720p 像素, confidence)
    eye_init: list[tuple[int, np.ndarray]]        # 全部 eyes 采样（初值用，含低置信度）
    cls: dict[int, str] = field(default_factory=dict)   # 帧 → normal / visual / effect
    color_filter: bool = True

    def hold(self, f: int) -> int:
        for h in self.holds:
            if h["start"] <= f < h["end"]:
                return h["start"] + (f - h["start"]) // h["every"] * h["every"]
        return f

    @property
    def frames(self) -> list[int]:
        return list(range(self.fin, self.fout))


def load_shot(sid: str) -> tuple[dict, dict]:
    s = json.loads(SHOTS.read_text(encoding="utf-8"))
    shot = next(x for x in s["shots"] if x["id"] == sid)
    return shot, s


def make_plan(sid: str) -> Plan:
    shot, allshots = load_shot(sid)
    insts = [i for i in shot["instances"] if any(t.get("bbox") for t in i["track"])]
    if len(insts) != 1:
        raise NotImplementedError(f"{sid}: {len(insts)} 个可见实例（A5 打样只支持单实例）")
    I = insts[0]
    kinds = {e["kind"] for e in shot["effects"]}
    for k in ("cardFlip", "depthOfField", "circleWindow"):
        if k in kinds:
            raise NotImplementedError(f"{sid}: 效果 {k} 未支持")
    if I.get("clip"):
        raise NotImplementedError(f"{sid}: clip 未支持")
    if any(isinstance(t.get("opacity"), (int, float)) and t["opacity"] < 0.99 for t in I["track"]):
        raise NotImplementedError(f"{sid}: opacity 轨未支持")
    if any((t.get("blurSigma") or 0) >= 1.5 for t in I["track"]):
        raise NotImplementedError(f"{sid}: 模糊段未支持")
    seen = [t for t in I["track"] if t.get("bbox")]
    last = seen[-1]["frame"]
    after = next((t for t in I["track"] if t["frame"] > last and "bbox" in t and t["bbox"] is None), None)
    fin = seen[0]["frame"]
    fout = min(shot["end"], after["frame"] if after else shot["end"])
    # 新画面节点：cadence 段内对齐的帧；段外每帧
    is_new = {f: True for f in range(fin, fout)}
    holds = []
    for c in allshots.get("cadence") or []:
        a, b, ev = max(c["start"], fin), min(c["end"], fout), c["every"]
        if b <= a:
            continue
        first = a + (-(a - c["start"])) % ev  # 段内第一个对齐帧（之前的帧按新画面处理）
        for f in range(first, b):
            is_new[f] = (f - c["start"]) % ev == 0
        if b - first >= 2:
            holds.append({"start": first, "end": b, "every": ev})
    is_new[fin] = True
    nodes = sorted(f for f, v in is_new.items() if v)
    zoom = bool(kinds & {"zoom", "pan", "scale"})
    every = 12 if zoom else 24
    keys = list(range(fin, fout - 1, every)) + [fout - 1]
    anchors, eye_init = [], []
    for t in I["track"]:
        if t.get("eyes") and fin <= t["frame"] < fout:
            e = np.asarray(t["eyes"], np.float64)
            e = e[np.argsort(e[:, 0])]  # 画面左、右
            eye_init.append((t["frame"], e))
            if (t.get("confidence") or 0) >= ANCHOR_CONF:
                anchors.append((t["frame"], e, float(t["confidence"])))
    # 开关：variant 常量 + switchEvents 斜坡（from = 切换前累计值）
    cur = dict(I.get("variant") or {})
    switches = []
    for e in sorted(I.get("switchEvents") or [], key=lambda e: e["frame"]):
        switches.append({"param": e["param"], "from": cur.get(e["param"], 0), "to": e["value"], "est": e["frame"], "fade": e.get("fade", 0) or 0})
        cur[e["param"]] = e["value"]
    p = Plan(shot=shot, inst=I, id=I["id"], fin=fin, fout=fout, nodes=nodes, keys=keys, holds=holds,
             constants=dict(I.get("variant") or {}), switches=switches, anchors=anchors, eye_init=eye_init)
    p.cls = {f: metric.classify(shot, f)[0] for f in range(fin, fout)}
    return p


def detect_new_frames(plan: Plan, full: np.ndarray) -> dict[int, float]:
    """参考帧重复检测（tools/fit/src/shotsutil.ts detectNewFrames 的移植，判据相同）：实例头部框（缺省 bbox）中间区域、
    隔 2 像素采样的 RGB 平均绝对差 d(f)；d < 0.35 或 d < 0.6 × min(d(f−1), d(f+1)) 判为重复；孤立的单帧「新」（前后两帧都新、
    隔一帧都不新）也判为重复。返回 d。结果写回 plan.nodes / plan.holds（见 finalize_nodes）。"""
    a, b = plan.fin, plan.fout
    d = {}
    for f in range(a + 1, b):
        cur, prev = full[f - a].astype(np.int32), full[f - a - 1].astype(np.int32)
        box = metric._interp_skip_null(plan.inst["track"], f, "head") or metric._interp_skip_null(plan.inst["track"], f, "bbox")
        x0, y0, x1, y1 = 0, 62, 1280, 652
        if box:
            x, y, w, h = box
            x0, x1 = max(0, round(x + w * 0.2)), min(1280, round(x + w * 0.8))
            y0, y1 = max(62, round(y + h * 0.25)), min(652, round(y + h * 0.9))
        d[f] = float(np.abs(cur[y0:y1:2, x0:x1:2] - prev[y0:y1:2, x0:x1:2]).mean()) if y1 > y0 and x1 > x0 else 0.0
    return d


def finalize_nodes(plan: Plan, full: np.ndarray) -> dict:
    """新画面节点与 frameHold：先按画面重复检测，再用 shots.json 的 cadence 段覆盖（段内只有对齐帧是新画面）——
    与 traj2.ts 相同的优先级；frameHold 段由节点间隔生成（shotsutil.ts holdSegments）。"""
    a, b = plan.fin, plan.fout
    d = detect_new_frames(plan, full)
    is_new = {a: True}
    for f in range(a + 1, b):
        nb = min(d.get(f - 1, np.inf), d.get(f + 1, np.inf))
        is_new[f] = not (d[f] < 0.35 or d[f] < 0.6 * nb)
    for f in range(a + 2, b - 2):
        if is_new[f] and not is_new[f - 2] and not is_new[f + 2] and is_new[f - 1] and is_new[f + 1]:
            is_new[f] = False
    _, allshots = load_shot(plan.shot["id"])
    nc = 0
    for c in allshots.get("cadence") or []:
        for f in range(max(c["start"], a + 1), min(c["end"], b)):
            is_new[f] = (f - c["start"]) % c["every"] == 0
            nc += 1
    news = sorted(f for f, v in is_new.items() if v)
    segs = []
    for i, g in enumerate(news):
        k = (news[i + 1] if i + 1 < len(news) else b) - g
        if k < 2:
            continue
        if segs and segs[-1]["end"] == g and segs[-1]["every"] == k:
            segs[-1]["end"] = g + k
        else:
            segs.append({"start": g, "end": g + k, "every": k})
    plan.nodes, plan.holds = news, segs
    # 变换关键帧止于最后一个节点（之后的显示帧都保持在它上面；再往后的关键帧没有任何数据约束）
    every = plan.keys[1] - plan.keys[0] if len(plan.keys) > 1 else 12
    last = news[-1]
    plan.keys = list(range(a, last, every)) + [last]
    return {"d": d, "cadenceFrames": nc}


def switch_value(sw: dict, g: int) -> float:
    """traj2.buildLayer 的 ramp：est 之前 from，余弦淡入 fade 帧。"""
    if g < sw["est"]:
        return sw["from"]
    if sw["fade"] <= 0 or g >= sw["est"] + sw["fade"]:
        return sw["to"]
    import math
    return sw["from"] + (sw["to"] - sw["from"]) * (0.5 - 0.5 * math.cos(math.pi * (g - sw["est"]) / sw["fade"]))


class Refs:
    """参考帧与排除区（按 metric.py 口径解码），按需降采样到拟合分辨率。"""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.full = metric.decode_ref(plan.fin, plan.fout)  # [N, 720, 1280, 3] uint8
        self.excl = np.stack([metric.exclusion_mask(plan.shot, f) for f in plan.frames])  # [N, 720, 1280] bool
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def at(self, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
        """(参考 RGB float32 0..255 [N, H, W, 3], 有效区权重 [N, H, W]：1 − 排除区覆盖率)。"""
        k = (W, H)
        if k not in self._cache:
            if (W, H) == (1280, 720):
                r = self.full.astype(np.float32)
                e = 1.0 - self.excl.astype(np.float32)
            else:
                r = np.stack([cv2.resize(x.astype(np.float64), (W, H), interpolation=cv2.INTER_AREA) for x in self.full]).astype(np.float32)
                e = np.stack([cv2.resize(1.0 - x.astype(np.float64), (W, H), interpolation=cv2.INTER_AREA) for x in self.excl]).astype(np.float32)
            self._cache[k] = (r, e)
        return self._cache[k]
