"""拟合单元与图层规划（冷启动：只读 shots.json 与 tools/fit 的规划代码）。

规划本身由 tools/gpu/scripts/a5_plan.ts 调用 tools/fit/src/plans.ts 的 fitUnits + planAuto 生成（同一份代码，语义与 fit4 完全一致）：
  - 拟合单元：continues 连起来的镜头合成一个单元，同一条 continues 链 = 一个图层（物理不重置）；
  - 图层区间、z、variant 常量、switchEvents 斜坡（from = 切换前累计值、fade、±8 帧窗口）、前倾 / 体型的固定值、颜色滤镜开关；
  - 逐帧量按 traj2.ts buildLayer 的规则采样好：景深 / 模糊段的 σ（dofK × 强度）、翻牌 scaleX、clip（圆形开窗 / 画中画圆角矩形）、不透明度。
这里补上：
  - 新画面节点与 frameHold：参考帧重复检测（shotsutil.ts detectNewFrames 的移植）+ shots.json 的 cadence 覆盖（traj2 的优先级），
    frameHold 段由节点间隔生成（shotsutil.ts holdSegments）；
  - 变换关键帧：每 24 帧，单元里有 zoom / pan / scale 效果时每 12 帧，止于最后一个节点；
  - 锚点：eyes 且 confidence ≥ 0.4；全部 eyes 采样（含 planAuto 由 head 框估的 confidence 0）用于变换初值；
  - 帧类别（普通 / 目测 / 特效）与排除区按 metric.py（以帧所在的镜头为准）。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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
class LayerPlan:
    id: str
    fin: int
    fout: int
    z: float
    constants: dict
    switches: list[dict]
    fixed: dict
    color_filter: bool
    dof: list                       # 逐帧：景深 / 模糊段内的强度（σ = dofK × 它），段外 None
    scale_x: list | None            # 逐帧翻牌 scaleX
    clip: dict | None               # 逐帧 clip（x, y, width, height, radius 各一列；feather 常量）
    opacity: list | None            # 逐帧不透明度
    track: list
    unit: str = ""                  # 所属拟合单元（多单元批处理时区分）
    nodes: list[int] = field(default_factory=list)
    keys: list[int] = field(default_factory=list)
    holds: list[dict] = field(default_factory=list)
    anchors: list = field(default_factory=list)    # (帧, [[x,y],[x,y]], confidence)
    eye_init: list = field(default_factory=list)   # (帧, [[x,y],[x,y]], confidence)

    def hold(self, f: int) -> int:
        for h in self.holds:
            if h["start"] <= f < h["end"]:
                return h["start"] + (f - h["start"]) // h["every"] * h["every"]
        return f

    def active(self, f: int) -> bool:
        return self.fin <= f < self.fout

    @property
    def frames(self) -> list[int]:
        return list(range(self.fin, self.fout))

    def at(self, name: str, g: int):
        """逐帧量在帧 g 的值（区间外取端点）。"""
        a = getattr(self, name)
        if a is None:
            return None
        i = min(max(g - self.fin, 0), self.fout - self.fin - 1)
        return a[i]

    def clip_at(self, g: int) -> dict | None:
        if self.clip is None:
            return None
        i = min(max(g - self.fin, 0), self.fout - self.fin - 1)
        return {k: (self.clip[k][i] if isinstance(self.clip[k], list) else self.clip[k]) for k in ("x", "y", "width", "height", "radius", "feather")}


@dataclass
class UnitPlan:
    unit: str
    shot_ids: list[str]
    start: int
    end: int
    effects: list[dict]
    layers: list[LayerPlan]
    skipped: list
    issues: list
    shots: dict = field(default_factory=dict)       # 镜头 ID → shots.json 的镜头
    cls: dict = field(default_factory=dict)         # 帧 → 类别
    units: list = field(default_factory=list)       # 多单元批：各单元的 UnitPlan（单个单元时为空）

    def shot_of(self, f: int) -> dict:
        for sid in self.shot_ids:
            s = self.shots[sid]
            if s["start"] <= f < s["end"]:
                return s
        raise KeyError(f)

    @property
    def frames(self) -> list[int]:
        if self.units:
            return [f for u in self.units for f in u.frames]
        return list(range(self.start, self.end))


def make_batch(units: list["UnitPlan"]) -> "UnitPlan":
    """多单元批：把几个拟合单元摞成一个「虚拟单元」。各单元在时间上不重叠，图层只在自己的区间内有效，
    所以合成、光流、数据项天然互不干扰；物理与变形则对全部图层一次批量求值（批处理的收益所在）。"""
    if len(units) == 1:
        return units[0]
    b = UnitPlan(unit="+".join(u.unit for u in units), shot_ids=[s for u in units for s in u.shot_ids],
                 start=min(u.start for u in units), end=max(u.end for u in units),
                 effects=[e for u in units for e in u.effects], layers=[l for u in units for l in u.layers],
                 skipped=[k for u in units for k in u.skipped], issues=[k for u in units for k in u.issues],
                 shots={k: v for u in units for k, v in u.shots.items()}, units=list(units))
    b.cls = {f: c for u in units for f, c in u.cls.items()}
    return b


def load_shots() -> dict:
    return json.loads(SHOTS.read_text(encoding="utf-8"))


def make_unit(sid: str) -> UnitPlan:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "plan.json"
        subprocess.run(["node", "--experimental-strip-types", "--disable-warning=ExperimentalWarning",
                        str(REPO / "tools/gpu/scripts/a5_plan.ts"), sid, str(out)], check=True, cwd=REPO, capture_output=True)
        q = json.loads(out.read_text())
    allshots = load_shots()
    shots = {s["id"]: s for s in allshots["shots"] if s["id"] in q["shots"]}
    zoom = any(e["kind"] in ("zoom", "pan", "scale") for e in q["effects"])
    layers = []
    for L in q["layers"]:
        lp = LayerPlan(id=L["id"], fin=L["in"], fout=L["out"], z=L["z"], constants=L["constants"], switches=L["switches"],
                       fixed=L["fixed"], color_filter=L["colorFilter"], dof=L["dof"], scale_x=L["scaleX"], clip=L["clip"],
                       opacity=L["opacity"], track=L["track"], unit=q["unit"])
        for t in lp.track:
            if t.get("eyes") and lp.fin <= t["frame"] < lp.fout:
                e = np.asarray(t["eyes"], np.float64)
                e = e[np.argsort(e[:, 0])]  # 画面左、右
                c = float(t.get("confidence") or 0)
                lp.eye_init.append((t["frame"], e, c))
                if c >= ANCHOR_CONF:
                    lp.anchors.append((t["frame"], e, c))
        lp.keys = list(range(lp.fin, lp.fout - 1, 12 if zoom else 24)) + [lp.fout - 1]
        layers.append(lp)
    u = UnitPlan(unit=q["unit"], shot_ids=q["shots"], start=q["start"], end=q["end"], effects=q["effects"], layers=layers,
                 skipped=q["skipped"], issues=q["issues"], shots=shots)
    u.cls = {f: metric.classify(u.shot_of(f), f)[0] for f in u.frames}
    return u


def _interp_box(track: list, f: int):
    return metric._interp_skip_null(track, f, "head") or metric._interp_skip_null(track, f, "bbox")


def finalize_nodes(u: UnitPlan, refs: "Refs") -> None:
    """每个图层：参考帧重复检测（shotsutil.ts detectNewFrames：实例头框中间区域隔 2 像素采样的 RGB 平均绝对差 d，
    d < 0.35 或 d < 0.6 × min(相邻 d) 判重复；孤立单帧「新」判重复）→ cadence 段覆盖 → frameHold 段（holdSegments）→ 关键帧止于最后一个节点。"""
    cad = load_shots().get("cadence") or []
    for lp in u.layers:
        a, b = lp.fin, lp.fout
        d = {}
        for f in range(a + 1, b):
            cur, prev = refs.frame(f).astype(np.int32), refs.frame(f - 1).astype(np.int32)
            box = _interp_box(lp.track, f)
            x0, y0, x1, y1 = 0, 62, 1280, 652
            if box:
                x, y, w, h = box
                x0, x1 = max(0, round(x + w * 0.2)), min(1280, round(x + w * 0.8))
                y0, y1 = max(62, round(y + h * 0.25)), min(652, round(y + h * 0.9))
            d[f] = float(np.abs(cur[y0:y1:2, x0:x1:2] - prev[y0:y1:2, x0:x1:2]).mean()) if y1 > y0 and x1 > x0 else 0.0
        is_new = {a: True}
        for f in range(a + 1, b):
            nb = min(d.get(f - 1, np.inf), d.get(f + 1, np.inf))
            is_new[f] = not (d[f] < 0.35 or d[f] < 0.6 * nb)
        for f in range(a + 2, b - 2):
            if is_new[f] and not is_new[f - 2] and not is_new[f + 2] and is_new[f - 1] and is_new[f + 1]:
                is_new[f] = False
        # 补充（A5）：连续两帧以上的保持（如 s21 2092～2094）里，后面几帧之间只有压缩噪声，d 不比相邻帧小多少，
        # 上面的判据会把它们都当成新画面，渲染随插值连续变化而原视频静止 → 闪烁尖峰。再加一条：
        # d 小于 ±3 帧内最大 d 的 0.2 倍，且绝对值 < 2.5（全片新画面的 d 一般 > 8），判为重复。
        for f in range(a + 1, b):
            loc = max(d.get(g, 0.0) for g in range(f - 3, f + 4))
            if is_new[f] and d[f] < 2.5 and d[f] < 0.2 * loc:
                is_new[f] = False
        for c in cad:
            for f in range(max(c["start"], a + 1), min(c["end"], b)):
                is_new[f] = (f - c["start"]) % c["every"] == 0
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
        every = lp.keys[1] - lp.keys[0] if len(lp.keys) > 1 else 24
        lp.nodes, lp.holds = news, segs
        lp.keys = list(range(a, news[-1], every)) + [news[-1]]


def switch_value(sw: dict, g: int) -> float:
    """traj2.buildLayer 的 ramp：est 之前 from，余弦淡入 fade 帧。"""
    if g < sw["est"]:
        return sw["from"]
    if sw["fade"] <= 0 or g >= sw["est"] + sw["fade"]:
        return sw["to"]
    import math
    return sw["from"] + (sw["to"] - sw["from"]) * (0.5 - 0.5 * math.cos(math.pi * (g - sw["est"]) / sw["fade"]))


class MultiRefs:
    """多单元批：按帧号分派到各单元的 Refs。"""

    def __init__(self, b: UnitPlan):
        self.parts = [(u, Refs(u)) for u in b.units]

    def _r(self, f: int) -> "Refs":
        for u, r in self.parts:
            if u.start <= f < u.end:
                return r
        raise KeyError(f)

    def frame(self, f: int):
        return self._r(f).frame(f)

    def excl(self, f: int):
        return self._r(f).excl(f)

    def at(self, W: int, H: int, frames: list[int]):
        rs, es = zip(*[self._r(f).at(W, H, [f]) for f in frames])
        return np.concatenate(rs), np.concatenate(es)


def make_refs(u: UnitPlan):
    return MultiRefs(u) if u.units else Refs(u)


class Refs:
    """参考帧（metric.py 口径解码，uint8 常驻内存）与排除区；按需降采样到拟合分辨率（只取请求的帧）。"""

    def __init__(self, u: UnitPlan):
        self.u = u
        self.full = metric.decode_ref(u.start, u.end)  # [N, 720, 1280, 3] uint8
        self._excl: dict[int, np.ndarray] = {}
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def frame(self, f: int) -> np.ndarray:
        return self.full[f - self.u.start]

    def excl(self, f: int) -> np.ndarray:
        if f not in self._excl:
            self._excl[f] = metric.exclusion_mask(self.u.shot_of(f), f)
        return self._excl[f]

    def at(self, W: int, H: int, frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(参考 RGB float32 0..255 [N, H, W, 3], 有效区权重 [N, H, W]：1 − 排除区覆盖率)。按 (W, H, 帧) 缓存。"""
        rs, es = [], []
        for f in frames:
            k = (W, H, f)
            if k not in self._cache:
                if (W, H) == (1280, 720):  # 全分辨率不缓存（内存）
                    rs.append(self.frame(f).astype(np.float32))
                    es.append(1.0 - self.excl(f).astype(np.float32))
                    continue
                r = cv2.resize(self.frame(f).astype(np.float64), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
                e = cv2.resize(1.0 - self.excl(f).astype(np.float64), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
                self._cache[k] = (r, e)
            r, e = self._cache[k]
            rs.append(r)
            es.append(e)
        return np.stack(rs), np.stack(es)
