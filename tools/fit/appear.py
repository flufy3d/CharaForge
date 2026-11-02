"""诊断：分区的「外观上限」与「几何上限」。

对每帧：
  1. 原样：分区 SSIM@360（与 metric 同一张 SSIM 图、同一掩码定义）；
  2. 扭正：用稠密光流（参考 → 我们，DIS）把我们的渲染（与分区标签）扭到参考帧上，再算分区 SSIM@360
     ——近似「几何完全对上」时的分数（光流可以把发丝逐条对齐，所以是偏乐观的几何上限 / 外观下限）；
  3. 外观扫描（在扭正后的画面上，只改外观）：附加高斯模糊 σ、降采样再升采样（模拟低分辨率渲染 / mipmap）、
     逐通道线性颜色（脸部最小二乘），看脸部和整体 SSIM@360 最多能到多少。

  tools/.venv/bin/python -I tools/fit/appear.py <mesh.json> --renders <png 目录> --shot s24 [--vis 2420 out.jpg]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/qa"))
sys.path.insert(0, str(REPO / "tools/fit"))
import metric  # noqa: E402
import regions  # noqa: E402

W, H = 1280, 720
BLURS = [0.0, 0.5, 1.0, 1.5, 2.0]
RESAMPLES = [1.0, 0.75, 0.5, 0.375]


def ssim_regions(comp, ref, mask, lab):
    c360 = cv2.resize(comp, (640, 360), interpolation=cv2.INTER_AREA)
    r360 = cv2.resize(ref, (640, 360), interpolation=cv2.INTER_AREA)
    m360 = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
    s = metric.ssim_map(c360, r360)
    l360 = lab[1::2, 1::2]
    out = {"all": float(s[m360].mean())}
    for k, name in enumerate(regions.CLASSES):
        m = m360 & (l360 == k)
        if m.sum() > 50:
            out[name] = float(s[m].mean())
    return out


def fill_labels(lab):
    if (lab < 0).any() and (lab >= 0).any():
        src = (lab < 0).astype(np.uint8)
        _, idx = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        zy, zx = np.nonzero(src == 0)
        lut = np.zeros(idx.max() + 1, np.int16)
        lut[idx[zy, zx]] = lab[zy, zx]
        lab = np.where(lab < 0, lut[idx], lab)
    return lab


def one(job):
    cv2.setNumThreads(1)
    fr, png, ref, excl, texs, vis = job
    rgb, a = metric.load_render(Path(png))
    ref = ref.astype(np.float64)
    pm = rgb * a[..., None]  # 预乘
    comp = pm + ref * (1 - a[..., None])
    mask = cv2.dilate((a > 0.1).astype(np.uint8), metric.DISK) > 0
    mask &= ~excl
    lab = fill_labels(regions.label_map(fr, texs))
    base = ssim_regions(comp, ref, mask, lab)
    # 光流：参考像素 → 我们的对应位置
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g1 = cv2.GaussianBlur(cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    bw = dis.calc(g2, g1, None)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xs + bw[..., 0], ys + bw[..., 1]
    wpm = cv2.remap(pm.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0).astype(np.float64)
    wa = cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0).astype(np.float64)
    wlab = cv2.remap(lab.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderValue=-1).astype(np.int16)
    wmask = (cv2.dilate((wa > 0.1).astype(np.uint8), metric.DISK) > 0) & ~excl
    wmask &= cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    wcomp = wpm + ref * (1 - wa[..., None])
    warped = ssim_regions(wcomp, ref, wmask, wlab)
    mag = np.hypot(bw[..., 0], bw[..., 1])
    disp = {name: float(np.median(mag[wmask & (wlab == k)])) for k, name in enumerate(regions.CLASSES) if (wmask & (wlab == k)).sum() > 200}
    # 外观扫描（扭正后的画面）
    sweep = {}
    face = wmask & (wlab == 0)
    for rs in RESAMPLES:
        if rs < 1:
            sw, sh = int(W * rs), int(H * rs)
            p2 = cv2.resize(cv2.resize(wpm, (sw, sh), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
            a2 = cv2.resize(cv2.resize(wa, (sw, sh), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            p2, a2 = wpm, wa
        for bl in BLURS:
            if bl > 0:
                p3, a3 = cv2.GaussianBlur(p2, (0, 0), bl), cv2.GaussianBlur(a2, (0, 0), bl)
            else:
                p3, a3 = p2, a2
            c = p3 + ref * (1 - a3[..., None])
            sweep[f"r{rs}_b{bl}"] = ssim_regions(c, ref, wmask, wlab)
    # 颜色：脸部逐通道线性（最小二乘），作用在整个图层
    cc = wcomp.copy()
    if face.sum() > 500:
        for k in range(3):
            A = np.stack([wcomp[..., k][face], np.ones(face.sum())], 1)
            g, o = np.linalg.lstsq(A, ref[..., k][face], rcond=None)[0]
            cc[..., k] = np.where(wa > 0.01, wcomp[..., k] * g + o, wcomp[..., k])
    color = ssim_regions(cc, ref, wmask, wlab)
    if vis:
        hm = cv2.applyColorMap(np.clip(mag * 255 / 20, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        v = np.where(wmask[..., None], 0.6 * hm + 0.4 * ref, ref * 0.5)
        im = np.concatenate([ref, comp, wcomp, v], 1)[::2, ::2]
        cv2.imwrite(vis, cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"frame": fr["frame"], "base": base, "warped": warped, "disp": disp, "sweep": sweep, "color": color}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--renders", required=True)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out")
    ap.add_argument("--vis", nargs=2)
    ap.add_argument("--control", type=int, default=0, help="对照：用帧 f−N 的渲染去比帧 f 的参考（检验光流扭正会不会把错的画面也「扭对」）")
    args = ap.parse_args()
    mesh = json.loads(Path(args.mesh).read_text(encoding="utf-8"))
    model3 = json.loads((regions.MODEL_DIR / "大肥鱼.model3.json").read_text(encoding="utf-8"))
    texs = [np.asarray(Image.open(regions.MODEL_DIR / t).getchannel("A")) for t in model3["FileReferences"]["Textures"]]
    shot = next(s for s in metric.load_shots()["shots"] if s["id"] == args.shot)
    fs = [fr["frame"] for fr in mesh["frames"]]
    refs = metric.decode_ref(min(fs), max(fs) + 1)
    jobs = [(fr, str(Path(args.renders) / f"frame_{fr['frame'] - args.control:06d}.png"), refs[fr["frame"] - min(fs)], metric.exclusion_mask(shot, fr["frame"]), texs,
             args.vis[1] if args.vis and int(args.vis[0]) == fr["frame"] else None) for fr in mesh["frames"]]
    with Pool(6) as pool:
        rows = pool.map(one, jobs)
    names = ["all"] + regions.CLASSES
    mean = lambda key, n: np.mean([r[key][n] for r in rows if n in r[key]])  # noqa: E731
    print(f"{len(rows)} 帧；分区 SSIM@360：原样 → 光流扭正（扭正位移中位 px）")
    for n in names:
        if any(n in r["base"] for r in rows):
            d = np.mean([r["disp"][n] for r in rows if n in r["disp"]]) if n != "all" else np.nan
            print(f"  {n:10s} {mean('base', n):.3f} → {mean('warped', n):.3f}   位移 {d:.1f}")
    best = {}
    for k in rows[0]["sweep"]:
        best[k] = (np.mean([r["sweep"][k]["face"] for r in rows if "face" in r["sweep"][k]]), np.mean([r["sweep"][k]["all"] for r in rows]))
    print("外观扫描（扭正后）：resample × blur → 脸 / 全部")
    for k, (fa, al) in sorted(best.items(), key=lambda x: -x[1][0])[:6]:
        print(f"  {k:12s} 脸 {fa:.3f}  全部 {al:.3f}")
    print(f"  r1.0_b0.0（不改）脸 {best['r1.0_b0.0'][0]:.3f}  全部 {best['r1.0_b0.0'][1]:.3f}")
    print(f"颜色（脸部最小二乘线性）：脸 {mean('color', 'face'):.3f}  全部 {mean('color', 'all'):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
