"""时间稳定性（闪烁）检测：我们的帧间变化是否远大于原视频的帧间变化。

做法（与总负责人核实闪烁时用的脚本一致）：
  - 渲染（非预乘 RGB，不合成到原视频）和参考帧都用 INTER_AREA 降到 320×180 灰度；
  - 区域 = 相邻两帧渲染 alpha 都 > 0.5 的像素（降采样后），少于 200 像素不判定；
  - d_ours = 区域内我们相邻两帧灰度差的平均，d_ref = 同一区域内原视频相邻两帧灰度差的平均；
  - 尖峰帧：d_ours > 3 × max(d_ref, 1) 且 d_ours > 4；
  - 镜头第一帧（切点）不判定（调用方在每个镜头开始时 reset）。
另记「原视频静止但我们在动」：d_ref < 0.3 且 d_ours > 2（原视频是重复帧/定格时我们在抖），只做参考，不算尖峰。
"""

from __future__ import annotations

import cv2
import numpy as np

FW, FH = 320, 180
MIN_PIX = 200
SPIKE_RATIO, SPIKE_ABS, REF_FLOOR = 3.0, 4.0, 1.0
STATIC_REF, STATIC_OURS = 0.3, 2.0


def _gray(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb.astype(np.float32), (FW, FH), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


class FlickerTracker:
    """逐帧喂入（同一镜头内按帧序），返回该帧相对上一帧的闪烁度量。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev = None  # (ours_gray, ref_gray, alpha) 或 None

    def update(self, rgb: np.ndarray | None, alpha: np.ndarray | None, ref: np.ndarray) -> dict | None:
        """rgb: float 0–255 (H,W,3)，alpha: 0–1 (H,W)；没有渲染时传 None。返回
        {'flicker': d_ours, 'flickerRef': d_ref, 'flickerSpike': bool, 'flickerStatic': bool} 或 None（不判定）。"""
        refg = _gray(ref)
        if rgb is None or alpha is None:
            self.prev = None
            return None
        g = _gray(rgb)
        a = cv2.resize(alpha.astype(np.float32), (FW, FH), interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, (g, refg, a)
        if prev is None:
            return None
        pg, pref, pa = prev
        m = (a > 0.5) & (pa > 0.5)
        if int(m.sum()) <= MIN_PIX:
            return None
        d_ours = float(np.abs(g - pg)[m].mean())
        d_ref = float(np.abs(refg - pref)[m].mean())
        return {"flicker": d_ours, "flickerRef": d_ref,
                "flickerSpike": bool(d_ours > SPIKE_RATIO * max(d_ref, REF_FLOOR) and d_ours > SPIKE_ABS),
                "flickerStatic": bool(d_ref < STATIC_REF and d_ours > STATIC_OURS)}


def summarize(rows: list[dict]) -> dict:
    """镜头/全片的闪烁汇总。rows 为逐帧行（含 flicker* 字段，未判定为 None）。"""
    samp = [r for r in rows if r.get("flicker") is not None]
    spikes = [int(r["frame"]) for r in samp if r["flickerSpike"]]
    return {"samples": len(samp), "spikes": len(spikes), "spikeFrames": spikes,
            "staticMoving": sum(1 for r in samp if r["flickerStatic"]),
            "medianOurs": float(np.median([r["flicker"] for r in samp])) if samp else None,
            "medianRef": float(np.median([r["flickerRef"] for r in samp])) if samp else None}
