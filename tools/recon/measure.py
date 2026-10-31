"""人工量坐标用：把若干帧的同一区域放大并叠加绝对坐标网格（视频像素）。

用法：measure.py f1,f2,... x,y,w,h out.jpg [--scale 2] [--grid 10] [--cols 3]
网格：细线每 grid 像素，粗线+坐标标签每 5*grid 像素。
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import read_frame  # noqa: E402
from sheet import save  # noqa: E402


def tile(f, x, y, w, h, scale, g):
    im = read_frame(f)
    pad = np.zeros((h, w, 3), np.uint8)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(im.shape[1], x + w), min(im.shape[0], y + h)
    pad[y0 - y:y1 - y, x0 - x:x1 - x] = im[y0:y1, x0:x1]
    t = cv2.resize(pad, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    ov = t.copy()
    xs = [gx for gx in range((x // g + 1) * g, x + w, g)]
    ys = [gy for gy in range((y // g + 1) * g, y + h, g)]
    for gx in xs:
        X = int((gx - x) * scale)
        cv2.line(ov, (X, 0), (X, t.shape[0]), (0, 255, 255) if gx % (5 * g) == 0 else (0, 200, 0), 1)
    for gy in ys:
        Y = int((gy - y) * scale)
        cv2.line(ov, (0, Y), (t.shape[1], Y), (0, 255, 255) if gy % (5 * g) == 0 else (0, 200, 0), 1)
    # 坐标标签：每隔 10*grid 的交点都标 x,y，读数更方便
    L = 10 * g
    for gx in range((x // L + 1) * L, x + w, L):
        for gy in range((y // L + 1) * L, y + h, L):
            X, Y = int((gx - x) * scale), int((gy - y) * scale)
            cv2.putText(t, f"{gx},{gy}", (X + 2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(t, f"{gx},{gy}", (X + 2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    t = cv2.addWeighted(ov, 0.35, t, 0.65, 0)
    cv2.rectangle(t, (0, t.shape[0] - 20), (70, t.shape[0]), (0, 0, 0), -1)
    cv2.putText(t, str(f), (3, t.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return t


def main(argv):
    scale, g, cols = 2.0, 10, 3
    for k in ("--scale", "--grid", "--cols"):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]
            if k == "--scale": scale = float(v)
            elif k == "--grid": g = int(v)
            else: cols = int(v)
    frames = [int(v) for v in argv[0].split(",")]
    x, y, w, h = (int(v) for v in argv[1].split(","))
    tiles = [tile(f, x, y, w, h, scale, g) for f in frames]
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (th + 4), cols * (tw + 4), 3), 30, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * (th + 4):r * (th + 4) + th, c * (tw + 4):c * (tw + 4) + tw] = t
    save(out, argv[2], 92)
    print(argv[2])


if __name__ == "__main__":
    main(sys.argv[1:])
