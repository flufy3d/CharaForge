"""人工精修锚点用（anchor_index.json 记录每格的裁切变换：video = (display - (ox,oy)) / s + (x0,y0)）：对 annotations.json 里每个带 head 的锚点，按当前 head 框放大裁切（外扩到 2.2 倍），
叠加「视频坐标」网格（细线每 step，标签每 5*step），并画出当前 head 框（红）和 eyes（绿十字）。
每张图 2x2 个锚点，输出 review/anchor_<n>.jpg 和索引 review/anchor_index.json。
用法：anchorsheet.py [--shots s04,s06]
"""
from __future__ import annotations
import json, sys
import cv2
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import ANNOT, DATA, read_frame  # noqa: E402
from sheet import save  # noqa: E402

TW, TH = 900, 506

def tile(f, a, label):
    im = read_frame(f)
    x, y, w, h = a["head"]
    cx, cy = x + w / 2, y + h / 2
    span = max(w, h * 16 / 9) * 2.2
    span = max(span, 160)
    rw, rh = span, span * 9 / 16
    x0, y0 = int(cx - rw / 2), int(cy - rh / 2)
    s = TW / rw
    step = 5 if s >= 3 else 10 if s >= 1.5 else 20 if s >= 0.8 else 40
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    c = cv2.warpAffine(im, M, (TW, TH), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
    ov = c.copy()
    for gx in range((x0 // step + 1) * step, int(x0 + rw) + 1, step):
        X = int((gx - x0) * s); major = gx % (5 * step) == 0
        cv2.line(ov, (X, 0), (X, TH), (0, 255, 255) if major else (0, 180, 0), 1)
    for gy in range((y0 // step + 1) * step, int(y0 + rh) + 1, step):
        Y = int((gy - y0) * s); major = gy % (5 * step) == 0
        cv2.line(ov, (0, Y), (TW, Y), (0, 255, 255) if major else (0, 180, 0), 1)
    c = cv2.addWeighted(ov, 0.3, c, 0.7, 0)
    L = 5 * step
    for gx in range((x0 // L + 1) * L, int(x0 + rw), L):
        X = int((gx - x0) * s)
        for Y in (12, TH // 2, TH - 4):
            cv2.putText(c, str(gx), (X + 2, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(c, str(gx), (X + 2, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for gy in range((y0 // L + 1) * L, int(y0 + rh), L):
        Y = int((gy - y0) * s)
        for X in (2, TW // 2, TW - 40):
            cv2.putText(c, str(gy), (X, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(c, str(gy), (X, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 255), 1)
    p = lambda X, Y: (int((X - x0) * s), int((Y - y0) * s))  # noqa: E731
    cv2.rectangle(c, p(x, y), p(x + w, y + h), (0, 0, 255), 2)
    for ex, ey in a.get("eyes") or []:
        cv2.drawMarker(c, p(ex, ey), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.rectangle(c, (0, TH - 22), (TW, TH), (0, 0, 0), -1)
    cv2.putText(c, f"{label}  step={step}px", (4, TH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    return c, (x0, y0, s)

def main(argv):
    only = set(argv[argv.index("--shots") + 1].split(",")) if "--shots" in argv else None
    ann = json.load(open(ANNOT, encoding="utf-8"))
    items = []
    for sh in ann["shots"]:
        if only and sh["id"] not in only: continue
        for inst in sh["instances"]:
            for a in inst["anchors"]:
                if a.get("head"):
                    items.append((sh["id"], inst["id"], a))
    idx = []
    out = DATA / "review"
    for n in range(0, len(items), 4):
        res = [tile(a["frame"], a, f"{sid}/{iid} f={a['frame']}") for sid, iid, a in items[n:n + 4]]
        ts = [r[0] for r in res]; xf = [r[1] for r in res]
        while len(ts) < 4: ts.append(np.zeros_like(ts[0]))
        img = np.vstack([np.hstack([ts[0], np.full((TH, 4, 3), 255, np.uint8), ts[1]]),
                         np.full((4, TW * 2 + 4, 3), 255, np.uint8),
                         np.hstack([ts[2], np.full((TH, 4, 3), 255, np.uint8), ts[3]])])
        name = f"anchor_{n // 4:03d}.jpg"
        save(img, out / name, 90)
        offs = [(0, 0), (TW + 4, 0), (0, TH + 4), (TW + 4, TH + 4)]
        idx.append({"file": name, "items": [
            {"key": f"{s}/{i}@{a['frame']}", "tile": k, "x0": xf[k][0], "y0": xf[k][1], "s": xf[k][2],
             "ox": offs[k][0], "oy": offs[k][1]} for k, (s, i, a) in enumerate(items[n:n + 4])]})
    (out / "anchor_index.json").write_text(json.dumps(idx, indent=0), encoding="utf-8")
    print(len(items), "个锚点，", len(idx), "张图")

if __name__ == "__main__":
    main(sys.argv[1:])
