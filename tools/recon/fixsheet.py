"""锚点人工修正用：每个有角色的镜头，把锚点帧以 0.5 倍缩放平铺（2 列），
叠加视频像素网格（细线每 20px，粗线+标签每 100px），并画出当前锚点（细化后）的 head 框（各实例不同色）与 eyes。
输出 review/fix_<shot>.jpg。
用法：fixsheet.py [s04,s06,...]
"""
from __future__ import annotations
import json, sys
import cv2
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import ANNOT, DATA, read_frame  # noqa: E402
from sheet import save  # noqa: E402
import build  # noqa: E402

COL = [(255, 60, 60), (60, 220, 60), (60, 140, 255), (255, 200, 0)]
S = 0.5

def render(f, marks):
    im = read_frame(f)
    t = cv2.resize(im, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    ov = t.copy()
    for gx in range(0, 1281, 20):
        cv2.line(ov, (int(gx * S), 0), (int(gx * S), t.shape[0]), (255, 255, 0) if gx % 100 == 0 else (0, 160, 0), 1)
    for gy in range(0, 721, 20):
        cv2.line(ov, (0, int(gy * S)), (t.shape[1], int(gy * S)), (255, 255, 0) if gy % 100 == 0 else (0, 160, 0), 1)
    t = cv2.addWeighted(ov, 0.3, t, 0.7, 0)
    for gx in range(100, 1280, 100):
        cv2.putText(t, str(gx), (int(gx * S) + 1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 0, 255), 1)
    for gy in range(100, 720, 100):
        cv2.putText(t, str(gy), (1, int(gy * S) - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 0, 255), 1)
    for k, (iid, a) in enumerate(marks):
        c = COL[k % 4]
        if a.get("head"):
            x, y, w, h = a["head"]
            cv2.rectangle(t, (int(x * S), int(y * S)), (int((x + w) * S), int((y + h) * S)), c, 1)
            cv2.putText(t, iid, (int(x * S), int(y * S) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
        for ex, ey in a.get("eyes") or []:
            cv2.drawMarker(t, (int(ex * S), int(ey * S)), c, cv2.MARKER_CROSS, 10, 1)
    cv2.rectangle(t, (0, t.shape[0] - 14), (60, t.shape[0]), (0, 0, 0), -1)
    cv2.putText(t, str(f), (2, t.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)

def main(argv):
    only = set(argv[0].split(",")) if argv else None
    ann = json.load(open(ANNOT, encoding="utf-8"))
    for sh in ann["shots"]:
        if not sh["instances"] or (only and sh["id"] not in only):
            continue
        byf = {}
        for inst in sh["instances"]:
            for a in inst["anchors"]:
                if a.get("head") or a.get("eyes"):
                    byf.setdefault(a["frame"], []).append((inst["id"], build.refine_anchor(dict(a))))
        if not byf:
            continue
        tiles = [render(f, byf[f]) for f in sorted(byf)]
        while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
        save(np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), DATA / "review" / f"fix_{sh['id']}.jpg", 90)
    print("ok")

if __name__ == "__main__":
    main(sys.argv[1:])
