"""诊断：「几何完美时」的外观上限。把我们的渲染按稠密光流（参考 → 我们）扭到参考帧上再算 SSIM@360（同 metric 定义的掩码与 SSIM 图）。
扭完仍然低 → 差距主要在外观（纹理、明暗、清晰度、颜色）；扭完很高 → 差距主要在几何（姿态/形变/物理）。
  tools/.venv/bin/python -I tools/fit/warpcheck.py <manifest.json>（同 flow.py；ours 为预乘 RGBA）
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

W, H = 1280, 720
DISK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def ssim360(comp, ref, mask):
    c = cv2.resize(comp, (640, 360), interpolation=cv2.INTER_AREA)
    r = cv2.resize(ref, (640, 360), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
    s = sum(structural_similarity(c[..., k], r[..., k], gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                  data_range=255, K1=0.01, K2=0.03, full=True)[1] for k in range(3)) / 3
    return float(s[m].mean())


def one(it):
    cv2.setNumThreads(1)
    ours = np.fromfile(it["ours"], np.uint8).reshape(H, W, 4).astype(np.float32)
    ref = np.asarray(np.memmap(it["ref"], np.uint8, "r")[it["refIndex"] * W * H * 3:(it["refIndex"] + 1) * W * H * 3]).reshape(H, W, 3).astype(np.float32)
    a = ours[..., 3:] / 255.0
    comp = ours[..., :3] + (1 - a) * ref
    mask = cv2.dilate((a[..., 0] > 0.1).astype(np.uint8), DISK) > 0
    for x, y, w, h in it.get("excl", []):
        x0, y0 = max(0, int(np.ceil(x - 0.5))), max(0, int(np.ceil(y - 0.5)))
        x1, y1 = min(W, int(np.ceil(x + w - 0.5))), min(H, int(np.ceil(y + h - 0.5)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = False
    base = ssim360(comp.astype(np.float64), ref.astype(np.float64), mask)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g1 = cv2.GaussianBlur(cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    bw = dis.calc(g2, g1, None)  # 参考像素 → 我们的对应位置
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xs + bw[..., 0], ys + bw[..., 1]
    wa = cv2.remap(a[..., 0], mx, my, cv2.INTER_LINEAR, borderValue=0)
    wrgb = cv2.remap(ours[..., :3], mx, my, cv2.INTER_LINEAR, borderValue=0)
    wcomp = wrgb + (1 - wa[..., None]) * ref
    wmask = cv2.dilate((wa > 0.1).astype(np.uint8), DISK) > 0
    wmask &= cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    warped = ssim360(wcomp.astype(np.float64), ref.astype(np.float64), wmask)
    magi = np.hypot(bw[..., 0], bw[..., 1])
    mag = magi[wmask]
    if it.get("vis"):
        hm = cv2.applyColorMap(np.clip(magi * 255 / 20, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        vis = np.where(wmask[..., None], 0.6 * hm + 0.4 * ref, ref * 0.5)
        im = np.concatenate([ref, comp, vis], 1)[::2, ::2]
        cv2.imwrite(it["vis"], cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return it["frame"], base, warped, float(np.median(mag)), float(np.percentile(mag, 90))


def main():
    man = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if len(sys.argv) > 4:  # 第 3 个参数：可视化的帧号；第 4 个：输出 jpg（参考 | 我们 | 位移热图，红 ≥ 20px）
        for it in man["frames"]:
            if it["frame"] == int(sys.argv[3]):
                it["vis"] = sys.argv[4]
    with Pool(8) as pool:
        rs = pool.map(one, man["frames"][::step])
    for f, b, w, m, p in rs:
        print(f"{f}: SSIM@360 {b:.3f} → 扭正后 {w:.3f}（光流位移中位 {m:.1f}px、p90 {p:.1f}px）")
    print(f"均值：{np.mean([r[1] for r in rs]):.3f} → {np.mean([r[2] for r in rs]):.3f}")


if __name__ == "__main__":
    main()
