"""复核用叠画图：在参考帧上画出 shots.json 的 bbox / head / eyes / clip / 排除区，并标出帧类别（metric.classify）。

细框 = bbox，粗框 = head，圆 + 连线 = eyes，白框 = clip，红色半透明 = 排除区；每个实例一种颜色，
左上角写帧号、帧类别及判定依据，实例框旁写 实例名 / eyesState / blurSigma / opacity / 眼距。
插值规则与 tools/qa/metric.py 一致（bbox 任一端为 null 则区间内不可见；head / eyes 跳过 null 插值）。

用法：tools/.venv/bin/python -I tools/recon/overlay.py s41 3538,3543,3548 out.jpg [--shots path] [--scale 0.5]
      [--cols 3] [--crop x,y,w,h] [--grid 100]
帧可写 a:b:step（左闭右开）或逗号列表。
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qa"))
from common import SHOTS_JSON, read_frame  # noqa: E402
from metric import _interp_rects, _interp_skip_null, classify  # noqa: E402

COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 160, 0), (0, 128, 255), (255, 255, 0)]


def _clip_at(clip, f):
    if not clip:
        return None
    ks = [{"frame": c["frame"], "r": list(c["rect"]) + [c.get("radius", 0)]} for c in clip]
    return _interp_rects(ks, f, "r")


def _scalar(track, f, key):
    ks = [{"frame": t["frame"], "v": [t[key]]} for t in track if t.get(key) is not None]
    if not ks:
        return None
    return _interp_rects(ks, f, "v")[0]


def _state(track, f):
    prev = [t for t in track if t["frame"] <= f and t.get("eyesState")]
    return prev[-1]["eyesState"] if prev else "?"


def draw(shot, f, scale=0.5, crop=None, grid=0):
    img = read_frame(f).copy()
    ov = img.copy()
    for e in shot["exclusions"]:
        if e["start"] <= f < e["end"]:
            r = _interp_rects(e["rects"], f, "rect")
            if r:
                x, y, w, h = map(int, map(round, r))
                cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 0, 0), -1)
    img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    if grid:
        for x in range(0, 1280, grid):
            cv2.line(img, (x, 0), (x, 719), (128, 128, 128), 1)
            cv2.putText(img, str(x), (x + 2, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for y in range(0, 720, grid):
            cv2.line(img, (0, y), (1279, y), (128, 128, 128), 1)
            cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    lw = max(1, int(round(1 / scale)))
    for k, inst in enumerate(shot["instances"]):
        c = COLORS[k % len(COLORS)]
        tr = inst["track"]
        bb = _interp_rects([t for t in tr if "bbox" in t], f, "bbox")
        cl = _clip_at(inst.get("clip"), f)
        if cl:
            x, y, w, h, _ = cl
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), lw)
        if bb is None:
            continue
        x, y, w, h = bb
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, lw)
        hd = _interp_skip_null(tr, f, "head")
        if hd:
            hx, hy, hw, hh = hd
            cv2.rectangle(img, (int(hx), int(hy)), (int(hx + hw), int(hy + hh)), c, 3 * lw)
        ey = _interp_skip_null([{"frame": t["frame"], "e": sum(t["eyes"], [])} for t in tr if t.get("eyes")], f, "e")
        st = _state(tr, f)
        d = None
        if ey and st != "hidden":
            ey = [ey[:2], ey[2:]]
            p = [(int(round(e[0])), int(round(e[1]))) for e in ey]
            d = float(np.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1]))
            for q in p:
                cv2.circle(img, q, max(4, int(0.12 * d)), c, lw)
            cv2.line(img, p[0], p[1], c, lw)
        bs = _scalar(tr, f, "blurSigma")
        op = _scalar(tr, f, "opacity")
        txt = f"{inst['id'].split('.', 1)[1]} {st}" + (f" b{bs:.1f}" if bs is not None else "") + \
              (f" o{op:.2f}" if op is not None else "") + (f" d{d:.0f}" if d else "")
        tx, ty = int(max(0, min(1000, x + 4))), int(max(60, min(700, y + 60)))
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 / scale, (0, 0, 0), 4 * lw)
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 / scale, c, lw)
    cls, why = classify(shot, f)
    lab = f"{f} {cls} " + ";".join(w.split("[")[0] for w in why)
    if crop:
        x, y, w, h = crop
        img = img[y:y + h, x:x + w]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.putText(img, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(img, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def parse_frames(s):
    if ":" in s:
        a = list(map(int, s.split(":")))
        return list(range(a[0], a[1], a[2] if len(a) > 2 else 1))
    return [int(x) for x in s.split(",")]


def main(argv):
    pos, opt = [], {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            opt[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    sid, frames, out = pos
    shots = json.loads(open(opt.get("shots", SHOTS_JSON), encoding="utf-8").read())
    shot = next(s for s in shots["shots"] if s["id"] == sid)
    scale = float(opt.get("scale", 0.5))
    cols = int(opt.get("cols", 3))
    crop = list(map(int, opt["crop"].split(","))) if "crop" in opt else None
    tiles = [draw(shot, f, scale, crop, int(opt.get("grid", 0))) for f in parse_frames(frames)]
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * (th + 4), min(cols, len(tiles)) * (tw + 4), 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * (th + 4):r * (th + 4) + th, c * (tw + 4):c * (tw + 4) + tw] = t
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, canvas.shape)


if __name__ == "__main__":
    main(sys.argv[1:])
