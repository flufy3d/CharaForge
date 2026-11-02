"""变换初值的粗配准：多尺度模板匹配（梯度幅值的掩码归一化互相关），与掩码面积无关，不会被「挪出画面」骗。

  tools/.venv/bin/python -I tools/fit/reg.py <manifest.json>
manifest：{"frames": [{"frame": f, "ours": ".../ours_f.rgba"（只含本图层，预乘 RGBA）, "ref": ..., "refIndex": i,
                       "excl": [[x,y,w,h]...], "anchor": [ax, ay]（图层锚点的屏幕位置，缩放围绕它）}], "search": 像素, "logScale": 范围}
输出：JSON {帧号: {"dx", "dy", "dls", "ncc0", "ncc"}}（全分辨率像素；ncc0 = 不动时的相关）
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
S = 0.5  # 在半分辨率上匹配


def grad(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def one(args):
    it, search, lsr = args
    cv2.setNumThreads(1)
    ours = np.fromfile(it["ours"], np.uint8).reshape(H, W, 4).astype(np.float32)
    ref = np.asarray(np.memmap(it["ref"], np.uint8, "r")[it["refIndex"] * W * H * 3:(it["refIndex"] + 1) * W * H * 3]).reshape(H, W, 3).astype(np.float32)
    a = ours[..., 3] / 255.0
    if (a > 0.5).sum() < 2000:
        return it["frame"], None
    # 我们：预乘颜色本身就是「贴在黑底上」，灰度梯度 + 轮廓；参考：灰度梯度，排除区置 0
    go = grad(cv2.cvtColor(ours[..., :3], cv2.COLOR_RGB2GRAY))
    gr = grad(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY))
    em = np.ones((H, W), np.float32)
    for x, y, w, h in it.get("excl", []):
        em[max(0, int(y)):max(0, int(y + h)), max(0, int(x)):max(0, int(x + w))] = 0
    gr *= em
    go = cv2.resize(go, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    am = cv2.resize(a, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    gr = cv2.resize(gr, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
    ax, ay = it["anchor"][0] * S, it["anchor"][1] * S
    # 只用头部（锚点 = 两眼中点，半径 2.2 × 眼距）：身体与头发随姿态变化大，整身模板容易被假峰骗
    if it.get("eyeDist"):
        yy, xx = np.mgrid[0:am.shape[0], 0:am.shape[1]]
        am = am * ((xx - ax) ** 2 + (yy - ay) ** 2 < (2.2 * it["eyeDist"] * S) ** 2)
    best = None
    ncc0 = None
    pad = int(search * S)
    big = cv2.copyMakeBorder(gr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for ls, rot in [(l, r) for l in np.linspace(-lsr, lsr, 2 * int(round(lsr / 0.05)) + 1) for r in (-12, -6, 0, 6, 12)]:
        k = float(np.exp(ls))
        M = cv2.getRotationMatrix2D((float(ax), float(ay)), -rot, k).astype(np.float32)  # 围绕锚点缩放、旋转
        t = cv2.warpAffine(go, M, (gr.shape[1], gr.shape[0]))
        m = cv2.warpAffine(am, M, (gr.shape[1], gr.shape[0]))
        ys, xs = np.nonzero(m > 0.3)
        if len(ys) < 500:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tt, mm = t[y0:y1, x0:x1], (m[y0:y1, x0:x1] > 0.3).astype(np.float32)
        win = big[y0:y1 + 2 * pad, x0:x1 + 2 * pad]
        if win.shape[0] < tt.shape[0] or win.shape[1] < tt.shape[1]:
            continue
        r = cv2.matchTemplate(win, tt, cv2.TM_CCORR_NORMED, mask=mm)
        r[~np.isfinite(r)] = 0
        if abs(ls) < 1e-9 and rot == 0:
            ncc0 = float(r[pad, pad])
        _, v, _, loc = cv2.minMaxLoc(r)
        # 轻微偏好小改动（避免在平坦相关面上乱跳）
        v2 = v - 0.02 * abs(ls) / lsr - 0.02 * np.hypot(loc[0] - pad, loc[1] - pad) / max(1, pad) - 0.01 * abs(rot) / 12
        if best is None or v2 > best[0]:
            best = (v2, v, (loc[0] - pad) / S, (loc[1] - pad) / S, ls, rot)
    if best is None:
        return it["frame"], None
    return it["frame"], {"dx": best[2], "dy": best[3], "dls": best[4], "drot": best[5], "ncc0": ncc0 if ncc0 is not None else 0.0, "ncc": best[1]}


def main() -> None:
    man = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    search, lsr = man.get("search", 160), man.get("logScale", 0.8)
    with Pool(min(10, max(1, len(man["frames"])))) as pool:
        res = pool.map(one, [(it, search, lsr) for it in man["frames"]])
    print(json.dumps({str(f): v for f, v in res}))


if __name__ == "__main__":
    main()
