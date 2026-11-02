"""分区误差分析：把我们渲染的角色遮罩按 drawable 所属部件分成脸、前发、侧发、后发、耳、头饰、领子领结、身体，
分别统计 SSIM@360（与 tools/qa/metric.py 同一定义：同一张 SSIM 图，只是按区域取均值）。

  tools/.venv/bin/python -I tools/fit/regions.py <mesh.json> --renders <dir> --shot s24 [--out stats.json] [--png 2400]
mesh.json 由 tools/fit/scripts/meshes.ts 导出（屏幕坐标网格 + UV + 部件路径）。标签图按 render order 从后往前画，
每个三角形内按 UV 取纹理 alpha（> 0.5 才算覆盖），所以前发大网格的透明部分不会抢走下面的标签。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/qa"))
import metric  # noqa: E402  （同仓库的指标实现，只调用其函数）

W, H = 1280, 720
MODEL_DIR = REPO / "assets/models/deepseek-chan/DeepSeek"
CLASSES = ["face", "frontHair", "sideHair", "backHair", "ears", "headAcc", "collar", "body", "other"]
COLORS = np.array([[255, 210, 180], [40, 80, 220], [80, 160, 255], [20, 30, 120], [200, 200, 200], [255, 220, 0], [255, 0, 160], [0, 200, 80], [128, 128, 128]], np.uint8)


def classify(path: str) -> int:
    p = path.split("/")
    top = p[0] if p else ""
    if top == "后发":
        return 3
    if top == "头":
        if "侧发" in p or "下侧发" in p:
            return 2
        if len(p) > 1 and p[1] in ("前发", "厚度头发"):
            return 1
        if len(p) > 1 and p[1] == "发饰":
            return 5
        return 0
    if top == "身体":
        if len(p) > 1 and p[1] == "耳朵":
            return 4
        if len(p) > 1 and p[1] == "女仆发饰":
            return 5
        if len(p) > 1 and p[1] in ("领子", "脖子"):
            return 6
        return 7
    return 8


def label_map(frame: dict, texs: list[np.ndarray]) -> np.ndarray:
    lab = np.full((H, W), -1, np.int16)
    ds = sorted(frame["drawables"], key=lambda d: d["order"])
    for d in ds:
        if d["blend"] != 0 or d["masked"] or d["opacity"] < 0.5:
            continue  # 叠加/乘法/被遮罩的高光阴影层不占标签
        cls = classify(d["path"])
        pts = np.array(d["pts"], np.float64).reshape(-1, 2)
        uv = np.array(d["uv"], np.float64).reshape(-1, 2)
        tri = np.array(d["idx"], np.int64).reshape(-1, 3)
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1 + 1), min(H, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        idm = np.zeros((y1 - y0, x1 - x0), np.int32)
        for t, (a, b, c) in enumerate(tri):
            poly = np.round((pts[[a, b, c]] - [x0, y0]) * 4).astype(np.int32)
            cv2.fillPoly(idm, [poly], t + 1, lineType=cv2.LINE_8, shift=2)
        ys, xs = np.nonzero(idm)
        if not len(ys):
            continue
        t = idm[ys, xs] - 1
        A, B, C = pts[tri[t, 0]], pts[tri[t, 1]], pts[tri[t, 2]]
        P = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], 1)
        v0, v1, v2 = B - A, C - A, P - A
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den[np.abs(den) < 1e-9] = 1e-9
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        l0 = 1 - l1 - l2
        u = l0 * uv[tri[t, 0], 0] + l1 * uv[tri[t, 1], 0] + l2 * uv[tri[t, 2], 0]
        v = l0 * uv[tri[t, 0], 1] + l1 * uv[tri[t, 1], 1] + l2 * uv[tri[t, 2], 1]
        tex = texs[d["tex"]]
        th, tw = tex.shape
        tx = np.clip((u * tw).astype(int), 0, tw - 1)
        ty = np.clip(((1 - v) * th).astype(int), 0, th - 1)
        ok = tex[ty, tx] > 127
        lab[ys[ok] + y0, xs[ok] + x0] = cls
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--renders", required=True)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out")
    ap.add_argument("--png", type=int, help="把这一帧的标签图与 SSIM 图写成 PNG（同目录）")
    args = ap.parse_args()
    mesh = json.loads(Path(args.mesh).read_text(encoding="utf-8"))
    model3 = json.loads((MODEL_DIR / "大肥鱼.model3.json").read_text(encoding="utf-8"))
    texs = [np.asarray(Image.open(MODEL_DIR / t).getchannel("A")) for t in model3["FileReferences"]["Textures"]]
    shot = next(s for s in metric.load_shots()["shots"] if s["id"] == args.shot)
    frames = [fr["frame"] for fr in mesh["frames"]]
    refs = metric.decode_ref(min(frames), max(frames) + 1)
    rows = []
    for fr in mesh["frames"]:
        f = fr["frame"]
        rgb, alpha = metric.load_render(Path(args.renders) / f"frame_{f:06d}.png")
        ref = refs[f - min(frames)].astype(np.float64)
        comp = rgb * alpha[..., None] + ref * (1 - alpha[..., None])
        mask = cv2.dilate((alpha > 0.1).astype(np.uint8), metric.DISK) > 0
        mask &= ~metric.exclusion_mask(shot, f)
        c360 = cv2.resize(comp, (640, 360), interpolation=cv2.INTER_AREA)
        r360 = cv2.resize(ref, (640, 360), interpolation=cv2.INTER_AREA)
        m360 = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
        s = metric.ssim_map(c360, r360)
        lab = label_map(fr, texs)
        # 遮罩内（膨胀带）没有标签的像素：取最近的标签
        if (lab < 0).any():
            dist_src = (lab < 0).astype(np.uint8)
            _, idx = cv2.distanceTransformWithLabels(dist_src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            zy, zx = np.nonzero(dist_src == 0)
            lut = np.zeros(idx.max() + 1, np.int16)
            lut[idx[zy, zx]] = lab[zy, zx]
            lab = np.where(lab < 0, lut[idx], lab)
        lab360 = lab[1::2, 1::2]
        row = {"frame": f, "ssim360": float(s[m360].mean()), "regions": {}}
        for k, name in enumerate(CLASSES):
            m = m360 & (lab360 == k)
            if m.sum() > 50:
                row["regions"][name] = {"ssim": round(float(s[m].mean()), 3), "area": round(float(m.sum() / m360.sum()), 3),
                                        "deficit": round(float((1 - s[m]).sum() / (1 - s[m360]).sum()), 3)}
        rows.append(row)
        if args.png == f:
            out = Path(args.out or args.mesh).with_suffix("")
            col = COLORS[np.clip(lab, 0, len(CLASSES) - 1)]
            col[lab < 0] = 0
            vis = (0.5 * col + 0.5 * comp).astype(np.uint8)
            cv2.imwrite(f"{out}_labels_{f}.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)[::2, ::2])
    # 汇总
    agg = {}
    for name in CLASSES:
        xs = [r["regions"][name] for r in rows if name in r["regions"]]
        if xs:
            agg[name] = {k: round(float(np.mean([x[k] for x in xs])), 3) for k in ("ssim", "area", "deficit")}
    res = {"frames": rows, "mean": float(np.mean([r["ssim360"] for r in rows])), "regions": agg}
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"SSIM@360 均值 {res['mean']:.3f}（{len(rows)} 帧）")
    for name, a in agg.items():
        print(f"  {name:10s} SSIM {a['ssim']:.3f}  面积 {a['area']:.3f}  误差占比 {a['deficit']:.3f}")


if __name__ == "__main__":
    main()
