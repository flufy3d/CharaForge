"""锚点加密：在相邻两个人工锚点之间，用 LK 光流跟踪头部区域的角点，
逐帧估计相似变换（平移+旋转+等比缩放，RANSAC），把锚点的 head 框 / eyes / bbox 传播到中间帧。

- 前向（从锚点 A 往后）和后向（从锚点 B 往前）各跟一次，按时间线性加权融合，
  这样两端都精确等于人工锚点，中间漂移被两端约束；
- 跟踪失败（内点太少）的帧退化为两锚点线性插值；
- rotation：锚点给定值 + 跟踪得到的相对旋转；opacity / blurSigma 只在锚点间线性插值（不跟踪）。
- 输出：锚点帧 + 约每 10 帧一个采样（只在整数帧），供 shots.json 的 track 使用。

trackMode（annotations.json 实例字段）：
  "lk"（默认）光流跟踪；"linear" 只在锚点间线性插值（遮挡严重/模糊/很短的镜头用）。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import read_frame  # noqa: E402

STEP = 10
SCALE = 0.5   # 跟踪在半分辨率灰度图上做


def gray(f):
    g = cv2.cvtColor(read_frame(f), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def rect_pts(r):
    x, y, w, h = r
    return np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], float)


def pts_rect(p):
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    return [x0, y0, p[:, 0].max() - x0, p[:, 1].max() - y0]


def track_seq(frames, region):
    """沿 frames 顺序跟踪 region（视频像素 [x,y,w,h]）内的角点，返回每帧相对首帧的 2x3 相似变换（视频像素）或 None。"""
    g0 = gray(frames[0])
    x, y, w, h = [v * SCALE for v in region]
    mask = np.zeros_like(g0)
    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
    p0 = cv2.goodFeaturesToTrack(g0, 300, 0.01, 4, mask=mask)
    out = [np.array([[1, 0, 0], [0, 1, 0]], float)]
    if p0 is None or len(p0) < 8:
        return out + [None] * (len(frames) - 1)
    ref = p0.reshape(-1, 2).copy()
    cur = p0.copy()
    alive = np.ones(len(ref), bool)
    prev = g0
    for f in frames[1:]:
        g = gray(f)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, cur, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, nxt, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm((back - cur).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        alive &= ok
        cur = nxt
        prev = g
        if alive.sum() < 8:
            out.append(None)
            alive[:] = False
            continue
        M, inl = cv2.estimateAffinePartial2D(ref[alive], cur.reshape(-1, 2)[alive], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5)
        if M is None or inl.sum() < 8:
            out.append(None)
            continue
        M = M.copy()
        M[:, 2] /= SCALE
        out.append(M)
    return out


def apply(M, p):
    return p @ M[:, :2].T + M[:, 2]


def rot_of(M):
    return float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def lerp(a, b, t):
    if a is None or b is None:
        return a if t < 0.5 else b
    return (np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t).tolist()


def sample_from(anchor, M):
    s = {}
    if M is None:
        return None
    for k in ("head", "bbox"):
        if anchor.get(k):
            s[k] = pts_rect(apply(M, rect_pts(anchor[k])))
    if anchor.get("eyes"):
        s["eyes"] = apply(M, np.asarray(anchor["eyes"], float)).tolist()
    s["drot"] = rot_of(M)
    return s


def densify(start, end, inst):
    anchors = sorted(inst["anchors"], key=lambda a: a["frame"])
    mode = inst.get("trackMode", "lk")
    want = set(range(start, end, STEP)) | {end - 1} | {a["frame"] for a in anchors}
    res = {a["frame"]: dict(a) for a in anchors}
    for A, B in zip(anchors, anchors[1:]):
        fa, fb = A["frame"], B["frame"]
        inner = [f for f in sorted(want) if fa < f < fb]
        if not inner:
            continue
        fw = bw = None
        if mode == "lk" and A.get("head") and B.get("head"):
            seq = list(range(fa, fb + 1))
            reg_a = A.get("trackRegion") or A["head"]
            reg_b = B.get("trackRegion") or B["head"]
            fw = dict(zip(seq, track_seq(seq, reg_a)))
            bw = dict(zip(seq[::-1], track_seq(seq[::-1], reg_b)))
        for f in inner:
            t = (f - fa) / (fb - fa)
            s = {"frame": f}
            sf = sample_from(A, fw[f]) if fw else None
            sb = sample_from(B, bw[f]) if bw else None
            for k in ("head", "bbox", "eyes"):
                lin = lerp(A.get(k), B.get(k), t)
                if A.get(k) is None or B.get(k) is None:
                    # 一端不可见（null）：从不可见到可见的过渡期间保持不可见，直到可见锚点；
                    # 从可见到不可见则保持可见值。只有两端都没给这个键时才跳过。
                    if k in A and A[k] is None:
                        s[k] = None
                    elif A.get(k) is not None:
                        s[k] = A[k]
                    elif lin is not None:
                        s[k] = lin
                    continue
                a = sf.get(k) if sf else None
                b = sb.get(k) if sb else None
                if a is not None and b is not None:
                    s[k] = lerp(a, b, t)
                elif a is not None:
                    s[k] = lerp(a, B[k], t)
                elif b is not None:
                    s[k] = lerp(A[k], b, t)
                else:
                    s[k] = lin
            if "rotation" in A or "rotation" in B:
                ra, rb = A.get("rotation", 0.0), B.get("rotation", 0.0)
                cand = []
                if sf:
                    cand.append((1 - t, ra + sf["drot"]))
                if sb:
                    cand.append((t, rb + sb["drot"]))
                if cand and sum(w for w, _ in cand) > 0:
                    s["rotation"] = sum(w * v for w, v in cand) / sum(w for w, _ in cand)
                else:
                    s["rotation"] = ra * (1 - t) + rb * t
            for k in ("opacity", "blurSigma"):
                if k in A or k in B:
                    s[k] = A.get(k, B.get(k)) * (1 - t) + B.get(k, A.get(k)) * t
            s["source"] = "flow" if (sf is not None or sb is not None) else "linear"
            s["confidence"] = 0.5 if s["source"] == "flow" else 0.35
            res[f] = s
    for a in anchors:
        res[a["frame"]].setdefault("source", "manual")
        res[a["frame"]].setdefault("confidence", 0.7)
    out = []
    for f in sorted(res):
        s = {k: v for k, v in res[f].items() if k not in ("trackRegion", "note")}
        out.append(s)
    return out
