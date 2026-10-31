"""转场期间角色可见区域的自动估计 → tools/recon/transitions.json（manual.py 读取）。

思路：转场（泼溅/剪影/网点/云雾/翻页擦除）开始前一帧 ref 是“干净”的旧画面。转场中第 f 帧里，
与 ref 像素差小（max 通道差 < 28，开运算去噪）的像素视为“仍是旧画面”；与旧实例 bbox 求交：
  - 可见框 visible[f] = 交集的外接框（像素数 < 可区分区域 0.3% 视为不可见 → null）
  - 只统计 ref 与「覆盖帧」颜色可区分的像素（白底遇白剪影这类像素无法判断，当作被遮挡）
  - 遮挡框 occluder[f] = 旧实例 bbox 内「原本可区分、现在与覆盖帧颜色一致」像素的外接框（作为排除区）
head / eyes 是否仍可见用「与覆盖帧（转场末帧）明显不同」判断（headFrac ≥ 0.5、两眼 5x5 邻域 60% 以上未覆盖）。
反过来，新镜头开头仍残留擦除物时，用擦除结束后的干净帧作 ref，求新实例 bbox 内的不匹配区域作为排除区。

角色在转场期间仍在动（多为 30fps 小幅动作），边缘会被误判为“不匹配”，所以遮挡框偏大、可见框偏小；
这正好符合“排除区宁大勿小”。旋转万花筒转场（1509~1527）不适用，由 manual.py 手工给出。

依赖上一次 build 生成的 shots.json 里实例在 ref 帧的 bbox（不存在时退化为整帧）。
用法：tools/.venv/bin/python -I tools/recon/transitions.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import SHOTS_JSON, read_frame  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "transitions.json")
S = 0.5  # 在半分辨率上算

# 旧画面延续：(转场起, 转场止, ref 帧, [(实例全局 id)], 方向 'old')
# 新画面提前出现：(镜头起, 擦除物消失帧, ref 帧, [(实例全局 id)], 'new')
SPECS = [
    (372, 394, 371, ["s04.main"], "old"),
    (652, 676, 651, ["s06.main", "s06.pipL", "s06.pipR"], "old"),
    (776, 789, 775, ["s08.bowl", "s08.loading", "s08.wink"], "old"),
    (948, 962, 947, ["s10.left", "s10.right"], "old"),
    (1157, 1187, 1156, ["s13.main"], "old"),
    (1896, 1910, 1895, ["s19.bowl", "s19.glasses"], "old"),
    (2608, 2614, 2607, ["s26.main"], "old"),
    (2608, 2614, 2614, ["s28.body", "s28.full"], "new"),
    (2969, 2982, 2968, ["s30.main", "s30.full"], "old"),
    (3280, 3296, 3279, ["s36.main"], "old"),
    (3382, 3392, 3381, ["s38.main", "s38.side"], "old"),
    (3662, 3668, 3661, ["s41.main", "s41.full"], "old"),
    # 新镜头开头的擦除残留（排除区）
    (394, 404, 404, ["s06.main"], "new"),
    (676, 684, 684, ["s08.bowl"], "new"),
    (789, 795, 795, ["s10.left", "s10.right"], "new"),
    (962, 970, 970, ["s12.main"], "new"),
    (1187, 1199, 1199, ["s15.left"], "new"),
    (1910, 1922, 1922, ["s21.main", "s21.side"], "new"),
    (2982, 3004, 3004, ["s32.main", "s32.full"], "new"),
    (3296, 3305, 3305, ["s38.main", "s38.side"], "new"),
    (3392, 3404, 3404, ["s40.chibi", "s40.full"], "new"),
    (3668, 3688, 3688, ["s43.main"], "new"),
]


def interp_bbox(track, f):
    pts = [(t["frame"], t["bbox"]) for t in track if t.get("bbox")]
    if not pts:
        return None
    if f <= pts[0][0]:
        return pts[0][1]
    if f >= pts[-1][0]:
        return pts[-1][1]
    for (fa, a), (fb, b) in zip(pts, pts[1:]):
        if fa <= f <= fb:
            t = (f - fa) / max(1, fb - fa)
            return [a[i] * (1 - t) + b[i] * t for i in range(4)]


def find_inst(doc, gid):
    sid, iid = gid.split(".", 1)
    for sh in doc["shots"]:
        for inst in sh["instances"]:
            if inst["id"] == gid or (sh["id"] == sid and inst["id"] == iid):
                return inst
    return None


def interp_key(track, f, key):
    pts = [(t["frame"], t[key]) for t in track if t.get(key) is not None]
    if not pts:
        return None
    if f <= pts[0][0]:
        return pts[0][1]
    if f >= pts[-1][0]:
        return pts[-1][1]
    for (fa, a), (fb, b) in zip(pts, pts[1:]):
        if fa <= f <= fb:
            t = (f - fa) / max(1, fb - fa)
            return (np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t).round(1).tolist()


def find_bbox(doc, gid, f):
    inst = find_inst(doc, gid)
    return interp_bbox(inst["track"], f) if inst else None


def small(f):
    return cv2.resize(read_frame(f), None, fx=S, fy=S, interpolation=cv2.INTER_AREA).astype(np.int16)


def rect_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [round(xs.min() / S, 1), round(ys.min() / S, 1), round((xs.max() - xs.min() + 1) / S, 1),
            round((ys.max() - ys.min() + 1) / S, 1)]


def main():
    doc = json.load(open(SHOTS_JSON, encoding="utf-8")) if os.path.exists(SHOTS_JSON) else {"shots": []}
    out = []
    k = np.ones((5, 5), np.uint8)
    for a, b, ref, gids, mode in SPECS:
        R = small(ref)
        H, W = R.shape[:2]
        # 「覆盖帧」：旧画面完全被盖住（old 取转场末帧）/ 擦除物最多（new 取镜头首帧）。
        # ref 与覆盖帧颜色相近的像素无法区分（白底遇白色剪影等），一律当作“被遮挡”。
        C = small(b - 1 if mode == "old" else a)
        distinct = (np.abs(R - C).max(2) >= 40).astype(np.uint8)
        regions = {}
        for g in gids:
            bb = find_bbox(doc, g, ref) or [0, 0, 1280, 720]
            x, y, w, h = [int(round(v * S)) for v in bb]
            regions[g] = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
        rec = {"start": a, "end": b, "ref": ref, "mode": mode, "instances": {g: [] for g in gids}, "refState": {}}
        for g in gids:
            inst = find_inst(doc, g)
            tr = inst["track"] if inst else []
            rec["refState"][g] = {k: interp_key(tr, ref, k) for k in ("bbox", "head", "eyes", "rotation")}
        for f in range(a, b):
            F = small(f)
            same = ((np.abs(F - R).max(2) < 28) & (distinct > 0)).astype(np.uint8)
            # 「尚未被覆盖」：与覆盖帧 C 明显不同的像素（不要求与 ref 相同，允许角色在转场中小幅移动）
            uncov = (np.abs(F - C).max(2) >= 40).astype(np.uint8)
            same = cv2.morphologyEx(same, cv2.MORPH_OPEN, k)
            for g, (x0, y0, x1, y1) in regions.items():
                sub = same[y0:y1, x0:x1]
                area = max(1, int(distinct[y0:y1, x0:x1].sum()))
                vis = sub.sum() / area
                m = np.zeros_like(same); m[y0:y1, x0:x1] = sub
                # 遮挡物 = 原本可区分、现在却与覆盖帧颜色一致的像素（即已被擦除物盖住）
                o = np.zeros_like(same); o[y0:y1, x0:x1] = (distinct * (1 - uncov))[y0:y1, x0:x1]
                o = cv2.morphologyEx(o, cv2.MORPH_OPEN, k)
                # 只保留面积 ≥ 区域 0.5% 的连通域：去掉角色白衣服等与白色剪影同色的零散像素
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats(o)
                keep = [j for j in range(1, n_) if st_[j, cv2.CC_STAT_AREA] >= 0.005 * area]
                o = np.isin(lab_, keep).astype(np.uint8) if keep else np.zeros_like(o)
                st = rec["refState"][g]
                hf = None
                if st.get("head"):
                    hx, hy, hw, hh = [int(round(v * S)) for v in st["head"]]
                    hd = distinct[max(0, hy):hy + hh, max(0, hx):hx + hw]
                    hs = (uncov * distinct)[max(0, hy):hy + hh, max(0, hx):hx + hw]
                    hf = round(float(hs.sum() / max(1, hd.sum())), 3)
                ev = None
                if st.get("eyes"):
                    ev = bool(vis > 0.003) and all(
                        uncov[max(0, int(ey * S) - 2):int(ey * S) + 3, max(0, int(ex * S) - 2):int(ex * S) + 3].mean() > 0.6
                        for ex, ey in st["eyes"])
                rec["instances"][g].append({
                    "frame": f, "visibleFrac": round(float(vis), 3), "headFrac": hf, "eyesVisible": ev,
                    "visible": rect_of(m) if vis > 0.003 else None,
                    "occluder": rect_of(o) if o.sum() > 0.003 * area else None})
        out.append(rec)
        print(a, b, mode, {g: [r["visibleFrac"] for r in v][::2] for g, v in rec["instances"].items()})
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=0)


if __name__ == "__main__":
    main()
