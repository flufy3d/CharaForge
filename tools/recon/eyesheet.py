"""复核眼睛位置用：对 shots.json 中带 eyes 的采样做脸部放大裁切，叠加 eyes（绿十字=采样值）和 head 框。
用法：eyesheet.py out.jpg [--shots s04,s06] [--every N] [--max M]
"""
from __future__ import annotations
import json, sys
import cv2
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import SHOTS_JSON, read_frame  # noqa: E402
from sheet import save  # noqa: E402

def crop_tile(f, t, label, tw=300, th=170):
    im = read_frame(f)
    (lx, ly), (rx, ry) = t["eyes"]
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    half = max(40, abs(rx - lx)) * 1.2
    x0, y0 = int(cx - half), int(cy - half * 0.57)
    x1, y1 = int(cx + half), int(cy + half * 0.57)
    pad = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(1280, x1), min(720, y1)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = im[sy0:sy1, sx0:sx1]
    s = tw / pad.shape[1]
    c = cv2.cvtColor(cv2.resize(pad, (tw, th), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
    for ex, ey in t["eyes"]:
        p = (int((ex - x0) * s), int((ey - y0) * s))
        cv2.drawMarker(c, p, (0, 255, 0), cv2.MARKER_CROSS, 18, 1)
        cv2.circle(c, p, max(2, int(3 * s)), (0, 0, 255), 1)   # 红圈 = ±3px
    cv2.rectangle(c, (0, 0), (tw, 16), (0, 0, 0), -1)
    cv2.putText(c, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
    return c

def main(argv):
    out = argv[0]
    only = None; every = 3; mx = 48
    if "--shots" in argv: only = set(argv[argv.index("--shots") + 1].split(","))
    if "--every" in argv: every = int(argv[argv.index("--every") + 1])
    if "--max" in argv: mx = int(argv[argv.index("--max") + 1])
    d = json.load(open(SHOTS_JSON, encoding="utf-8"))
    tiles = []
    for sh in d["shots"]:
        if only and sh["id"] not in only: continue
        for inst in sh["instances"]:
            ts = [t for t in inst["track"] if t.get("eyes")]
            for t in ts[::every]:
                tiles.append(crop_tile(t["frame"], t, f"{sh['id']}/{inst['id']} {t['frame']}"))
    tiles = tiles[:mx]
    cols = 6
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    save(np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), out)
    print(out, len(tiles))

if __name__ == "__main__":
    main(sys.argv[1:])
