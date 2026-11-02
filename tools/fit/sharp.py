"""清晰度估计（与几何无关）：给定我们「不加 blur / resample」的渲染，搜索 (blur σ, resample factor)，
使角色内部的多尺度梯度能量与参考帧一致。

  tools/.venv/bin/python -I tools/fit/sharp.py <manifest.json>     （manifest 格式同 flow.py）

方法：区域 = 我们 α>0.95 的部分腐蚀 6px、扣除排除区（角色内部，避开边缘与背景）；
对候选滤镜（在图像域模拟：预乘 RGBA 先按 factor 区域平均缩小再双线性放大，再高斯模糊 σ）后的合成图与参考帧，
分别算灰度在尺度 1、2、4（逐次 2× 区域下采样）上的 Sobel 梯度幅值均值，距离 = Σ_尺度 (log(ours/ref))²，跨帧求和。
梯度能量只看「有多锐」，对 1~3 px 的错位不敏感，所以不会像像素损失那样用模糊去掩盖几何误差。
输出 JSON：{"blur": σ, "resample": factor, "dist": 距离, "grid": [...]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720


def energy(gray: np.ndarray, mask: np.ndarray) -> list[float]:
    out = []
    g, m = gray, mask.astype(np.float32)
    for _ in range(3):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        mm = m > 0.99
        out.append(float(mag[mm].mean()) if mm.any() else 1.0)
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (m.shape[1] // 2, m.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return out


def apply(premul: np.ndarray, blur: float, factor: float) -> np.ndarray:
    x = premul
    if factor < 0.999:
        sw, sh = max(1, round(W * factor)), max(1, round(H * factor))
        x = cv2.resize(cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
    if blur > 1e-3:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    return x


def main() -> None:
    man = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    frames = []
    for it in man["frames"]:
        ours = np.fromfile(it["ours"], np.uint8).reshape(H, W, 4).astype(np.float32)
        ref = np.asarray(np.memmap(it["ref"], np.uint8, "r")[it["refIndex"] * W * H * 3:(it["refIndex"] + 1) * W * H * 3]).reshape(H, W, 3).astype(np.float32)
        a = ours[..., 3] / 255.0
        mask = cv2.erode((a > 0.95).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        for x, y, w, h in it.get("excl", []):
            x0, y0 = max(0, int(np.ceil(x - 0.5))), max(0, int(np.ceil(y - 0.5)))
            x1, y1 = min(W, int(np.ceil(x + w - 0.5))), min(H, int(np.ceil(y + h - 0.5)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = False
        if mask.sum() < 5000:
            continue
        gref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
        frames.append((ours, ref, mask, energy(gref, mask)))
    grid = []
    best = None
    for factor in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        for blur in np.arange(0, 3.01, 0.2):
            d = 0.0
            for ours, ref, mask, eref in frames:
                p = apply(ours, float(blur), factor)
                a = p[..., 3:] / 255.0
                comp = p[..., :3] + (1 - a) * ref
                e = energy(cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY), mask)
                d += sum(np.log(max(x, 1e-6) / max(y, 1e-6)) ** 2 for x, y in zip(e, eref))
            grid.append((round(float(blur), 2), factor, d))
            if best is None or d < best[2]:
                best = (round(float(blur), 2), factor, d)
    print(json.dumps({"blur": best[0], "resample": best[1], "dist": best[2], "frames": len(frames), "grid": grid}))


if __name__ == "__main__":
    main()
