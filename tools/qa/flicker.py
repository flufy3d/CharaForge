"""时间稳定性（闪烁）检测：我们的帧间变化是否远大于原视频的帧间变化。全仓库唯一实现
（report.py、metric.py score 和 tools/fit/scripts/flicker.py 都调用这里）。

口径（docs/handoff.md 第 3 节）：
  - 渲染（非预乘 RGB，不合成到原视频）和参考帧（与 metric.py 同一套 ffmpeg 颜色转换得到的全范围 RGB）
    都用 INTER_AREA 降到 320×180 灰度；
  - 区域 = 相邻两帧渲染 alpha 都 > 0.5 的像素（降采样后），不超过 200 像素不判定；
  - d_ours = 区域内我们相邻两帧灰度差的平均，d_ref = 同一区域内原视频相邻两帧灰度差的平均；
  - 尖峰帧：d_ours > 3 × max(d_ref, 1) 且 d_ours > 4；
  - 切点不判定：每个镜头的第一帧不和上一镜头比较（调用方在镜头开始时 reset）；
  - 原视频重复帧单独统计：d_ref < 0.3（原视频在这一区域基本没变，多为重复帧/定格）且 d_ours > 2（我们在变）。
    这类帧记为 flickerRefDup；其中同时满足尖峰条件的也计入尖峰（汇总里另给 refDupSpikes 便于区分）。

命令行（取代旧的 tools/fit/scripts/flicker.py，旧路径仍可用、会转调这里）：
  tools/.venv/bin/python -I tools/qa/flicker.py <渲染目录> <起帧-止帧(含)> [--json out.json]
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FW, FH = 320, 180
MIN_PIX = 200
SPIKE_RATIO, SPIKE_ABS, REF_FLOOR = 3.0, 4.0, 1.0
DUP_REF, DUP_OURS = 0.3, 2.0
EMPTY = {"flicker": None, "flickerRef": None, "flickerSpike": None, "flickerRefDup": None}


def _gray(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb.astype(np.float32), (FW, FH), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


class FlickerTracker:
    """逐帧喂入（同一镜头内按帧序），返回该帧相对上一帧的闪烁度量。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """切点：丢掉上一帧，下一帧不判定。"""
        self.prev = None  # (ours_gray, ref_gray, alpha) 或 None

    def update(self, rgb: np.ndarray | None, alpha: np.ndarray | None, ref: np.ndarray) -> dict | None:
        """rgb: 0–255 (H,W,3)，alpha: 0–1 (H,W)；没有渲染时传 None。返回
        {'flicker': d_ours, 'flickerRef': d_ref, 'flickerSpike': bool, 'flickerRefDup': bool} 或 None（不判定）。"""
        if rgb is None or alpha is None:
            self.prev = None
            return None
        refg = _gray(ref)
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
                "flickerRefDup": bool(d_ref < DUP_REF and d_ours > DUP_OURS)}


def summarize(rows: list[dict]) -> dict:
    """镜头/全片的闪烁汇总。rows 为逐帧行（含 flicker* 字段，未判定为 None）。"""
    samp = [r for r in rows if r.get("flicker") is not None]
    spikes = [int(r["frame"]) for r in samp if r["flickerSpike"]]
    dup = [int(r["frame"]) for r in samp if r["flickerRefDup"]]
    return {"samples": len(samp), "spikes": len(spikes), "spikeFrames": spikes,
            "refDupMoving": len(dup), "refDupMovingFrames": dup,
            "refDupSpikes": sum(1 for r in samp if r["flickerSpike"] and r["flickerRefDup"]),
            "medianOurs": float(np.median([r["flicker"] for r in samp])) if samp else None,
            "medianRef": float(np.median([r["flickerRef"] for r in samp])) if samp else None}


def by_shot(rows: list[dict]) -> dict[str, dict]:
    """rows 需带 'shot' 字段；按镜头分组汇总（只列有判定样本的镜头）。"""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["shot"], []).append(r)
    return {sid: s for sid, rs in groups.items() if (s := summarize(rs))["samples"]}


# ---------------- 命令行 ----------------

def main(argv=None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import metric  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders", help="渲染帧目录（frame_NNNNNN.png）；缺帧按无渲染处理（不判定）")
    ap.add_argument("frames", help="起帧-止帧（含两端），如 3478-3661")
    ap.add_argument("--json", help="把逐帧结果和汇总写到这个 JSON")
    args = ap.parse_args(argv)
    fa, fb = (int(x) for x in args.frames.split("-"))
    shots = sorted(metric.load_shots()["shots"], key=lambda s: s["start"])
    starts = [s["start"] for s in shots]
    ids = [s["id"] for s in shots]
    tr = FlickerTracker()
    rows = []
    for i, ref in enumerate(metric.ref_stream(fa, fb + 1)):
        f = fa + i
        k = bisect.bisect_right(starts, f) - 1
        if f in starts:
            tr.reset()  # 镜头切点
        p = Path(args.renders) / f"frame_{f:06d}.png"
        rgb, alpha = metric.load_render(p) if p.exists() else (None, None)
        rows.append({"frame": f, "shot": ids[k] if k >= 0 else None, **(tr.update(rgb, alpha, ref) or EMPTY)})
    tot = summarize(rows)
    per = by_shot(rows)
    print(f"样本帧 {tot['samples']}  闪烁尖峰帧 {tot['spikes']}  原视频重复帧但我们在变 {tot['refDupMoving']}"
          f"（其中也是尖峰 {tot['refDupSpikes']}）")
    for sid, s in sorted(per.items(), key=lambda t: (-t[1]["spikes"], -t[1]["refDupMoving"])):
        if s["spikes"] or s["refDupMoving"]:
            print(f"  {sid}: 尖峰 {s['spikes']} {s['spikeFrames'][:20]}{' …' if s['spikes'] > 20 else ''}"
                  f" | 重复帧在变 {s['refDupMoving']} {s['refDupMovingFrames'][:10]}")
    if tot["samples"]:
        print(f"帧间变化中位：渲染 {tot['medianOurs']:.2f}  原视频 {tot['medianRef']:.2f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"overall": tot, "shots": per, "perFrame": rows}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
