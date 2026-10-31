"""验收指标（docs/pipeline.md 最后一节的唯一实现）与逐镜头对照视频。

用法（Python 一律 -I 运行）：

  # 1) 指标：渲染帧目录（pnpm render 的输出，frame_NNNNNN.png，非预乘 RGBA）+ 镜头
  tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s43 --shots s43 \
      --out .data/qa/metric_s43.json

  # 2) 对照视频 + 关键帧截图（四宫格：原视频 | 渲染合成到原视频背景 | 差异图 | SSIM 热力图）
  tools/.venv/bin/python -I tools/qa/metric.py review --renders .data/qa/render/s43 --shots s43 \
      --out .data/fit/review

定义（与 pipeline.md 逐条对应）：
  1. M_f = dilate(α_f > 0.1, 半径 3px 圆盘) − 排除区（shots 里覆盖帧 f 的 exclusions 矩形之并，关键帧线性插值）；
  2. 渲染结果按 alpha 合成到参考帧上；RGB 三通道分别算 SSIM 图（skimage.structural_similarity，
     gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255, K1=0.01, K2=0.03），三通道平均；
  3. 帧 SSIM = SSIM 图在 M_f 上的均值；|M_f| < 1% 画面记为「无角色」，不参与统计；
  4. 覆盖率 = |M_f ∩ bbox| / |bbox − 排除区|，bbox = shots 里该帧各实例 bbox 之并（线性插值，null 表示不可见）；
  5. 镜头指标：帧 SSIM 均值、最小值、5% 分位数、≥0.9 的帧比例；全片 = 所有有效帧的平均。

参考帧用 ffmpeg 解码，颜色转换参数与 tools/fit/src/refvideo.ts 逐字一致。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
VIDEO = REPO / "assets/video/DeepSeek-chan.mp4"
AUDIO = REPO / "assets/video/DeepSeek-chan.wav"
SHOTS = REPO / "scenes/deepseek-chan/shots.json"
W, H, FPS = 1280, 720, 60
MIN_MASK = 0.01 * W * H


def ffmpeg_vf(a: int, b: int) -> str:
    # 与 refvideo.ts 的 ffmpegArgs 保持一致
    return (
        f"select='between(n\\,{a}\\,{b - 1})',"
        "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24"
    )


def decode_ref(a: int, b: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(VIDEO), "-vf", ffmpeg_vf(a, b),
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8)
    n = fr.size // (W * H * 3)
    if n != b - a:
        raise RuntimeError(f"解码帧数 {n} ≠ {b - a}")
    return fr.reshape(n, H, W, 3)


def load_shots() -> dict:
    return json.loads(SHOTS.read_text(encoding="utf-8"))


def _interp_rects(keys: list[dict], f: int, field: str):
    """关键帧线性插值；区间外取首/尾值。keys 里每项有 frame 和 field。"""
    ks = [k for k in keys if field in k]
    if not ks:
        return None
    if f <= ks[0]["frame"]:
        return ks[0].get(field)
    if f >= ks[-1]["frame"]:
        return ks[-1].get(field)
    for a, b in zip(ks, ks[1:]):
        if a["frame"] <= f < b["frame"]:
            va, vb = a.get(field), b.get(field)
            if va is None or vb is None:
                # null 表示不可见：只在采样帧本身取非空值，区间内视为不可见
                return va if f == a["frame"] else None
            t = (f - a["frame"]) / (b["frame"] - a["frame"])
            return [x + (y - x) * t for x, y in zip(va, vb)]
    return ks[-1].get(field)


def exclusion_mask(shot: dict, f: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for e in shot["exclusions"]:
        if not (e["start"] <= f < e["end"]):
            continue
        r = _interp_rects(e["rects"], f, "rect")
        rect_fill(m, r)
    return m


def rect_fill(m: np.ndarray, r) -> None:
    """像素中心落在矩形内的像素置 True。"""
    x, y, w, h = r
    x0, x1 = max(0, int(np.ceil(x - 0.5))), min(W, int(np.ceil(x + w - 0.5)))
    y0, y1 = max(0, int(np.ceil(y - 0.5))), min(H, int(np.ceil(y + h - 0.5)))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True


def bbox_mask(shot: dict, f: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for inst in shot["instances"]:
        ks = [t for t in inst["track"] if "bbox" in t]
        if not ks:
            continue
        # bbox 为 null 表示不可见：与相邻采样之间不插值，按「任一端为 null 则不可见」处理
        r = _interp_rects(ks, f, "bbox")
        if r is not None:
            rect_fill(m, r)
    return m


DISK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 半径 3px 圆盘


def load_render(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 pnpm render 输出的 PNG（非预乘 RGBA）→ (rgb float64, alpha float64 ∈ [0,1])"""
    im = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64)
    return im[..., :3], im[..., 3] / 255.0


def frame_metric(rgb: np.ndarray, alpha: np.ndarray, ref: np.ndarray, shot: dict, f: int, want_maps=False):
    comp = rgb * alpha[..., None] + ref.astype(np.float64) * (1.0 - alpha[..., None])
    excl = exclusion_mask(shot, f)
    mask = cv2.dilate((alpha > 0.1).astype(np.uint8), DISK) > 0
    mask &= ~excl
    area = int(mask.sum())
    out = {"frame": f, "maskArea": area}
    bb = bbox_mask(shot, f)
    denom = int((bb & ~excl).sum())
    out["coverage"] = (float((mask & bb).sum()) / denom) if denom > 0 else None
    if area < MIN_MASK:
        out["ssim"] = None
        if want_maps:
            return out, comp, None, mask
        return out
    smap = np.zeros((H, W), np.float64)
    for c in range(3):
        _, s = structural_similarity(
            comp[..., c], ref[..., c].astype(np.float64), gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, data_range=255, K1=0.01, K2=0.03, full=True,
        )
        smap += s
    smap /= 3.0
    out["ssim"] = float(smap[mask].mean())
    if want_maps:
        return out, comp, smap, mask
    return out


def shot_frames(shot: dict, args) -> range:
    a, b = shot["start"], shot["end"]
    if args.frames:
        lo, hi = (int(x) for x in args.frames.split("-"))
        a, b = max(a, lo), min(b, hi + 1)
    return range(a, b)


def summarize(rows: list[dict]) -> dict:
    vals = np.array([r["ssim"] for r in rows if r["ssim"] is not None], dtype=np.float64)
    cov = np.array([r["coverage"] for r in rows if r["ssim"] is not None and r["coverage"] is not None])
    if vals.size == 0:
        return {"frames": len(rows), "valid": 0}
    return {
        "frames": len(rows),
        "valid": int(vals.size),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "p5": float(np.percentile(vals, 5)),
        "fracGE0.9": float((vals >= 0.9).mean()),
        "coverageMean": float(cov.mean()) if cov.size else None,
        "coverageMin": float(cov.min()) if cov.size else None,
        "worstFrame": int(rows[int(np.argmin([r["ssim"] if r["ssim"] is not None else 9 for r in rows]))]["frame"]),
    }


def render_path(args, f: int) -> Path:
    return Path(args.renders) / f"frame_{f:06d}.png"


def cmd_score(args) -> None:
    shots = {s["id"]: s for s in load_shots()["shots"]}
    report = {"definition": "docs/pipeline.md#验收指标的定义", "renders": str(args.renders), "shots": {}}
    all_vals = []
    for sid in args.shots.split(","):
        shot = shots[sid]
        fr = shot_frames(shot, args)
        ref = decode_ref(fr.start, fr.stop)
        rows = []
        for f in fr:
            p = render_path(args, f)
            if not p.exists():
                raise FileNotFoundError(p)
            rgb, alpha = load_render(p)
            rows.append(frame_metric(rgb, alpha, ref[f - fr.start], shot, f))
        summ = summarize(rows)
        report["shots"][sid] = {"summary": summ, "perFrame": rows}
        all_vals += [r["ssim"] for r in rows if r["ssim"] is not None]
        print(f"{sid}: {json.dumps(summ, ensure_ascii=False)}")
    if all_vals:
        report["overall"] = {"valid": len(all_vals), "mean": float(np.mean(all_vals))}
        print(f"overall: {json.dumps(report['overall'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------- 对照视频 ----------------

def heat(smap: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    """SSIM 热力图：遮罩内按 SSIM 着色（0→红、1→绿，0.9 处为黄绿），遮罩外深灰。"""
    out = np.full((H, W, 3), 40, np.uint8)
    if smap is None:
        return out
    v = np.clip(smap, 0, 1)
    # 0.5 以下全红，0.5~1 线性过渡到绿；0.9 附近为黄绿，便于看出不达标区域
    t = np.clip((v - 0.5) / 0.5, 0, 1)
    col = np.stack([255 * (1 - t), 255 * np.clip(t * 1.6, 0, 1), np.zeros_like(t)], -1)
    out[mask] = col[mask].astype(np.uint8)
    return out


def label(img: np.ndarray, text: str, scale=1.0) -> None:
    cv2.putText(img, text, (12, int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 0), int(5 * scale), cv2.LINE_AA)
    cv2.putText(img, text, (12, int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)


def quad(ref: np.ndarray, comp: np.ndarray, smap, mask, excl, info: dict, half=True) -> np.ndarray:
    diff = np.clip(np.abs(comp - ref.astype(np.float64)).mean(-1) * 3, 0, 255).astype(np.uint8)
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)[..., ::-1].copy()
    hm = heat(smap, mask)
    hm[excl] = (hm[excl] * 0.4 + np.array([60, 60, 160]) * 0.6).astype(np.uint8)
    panels = [ref.copy(), np.clip(comp, 0, 255).astype(np.uint8), diff, hm]
    if half:
        panels = [cv2.resize(p, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for p in panels]
    top = np.concatenate(panels[:2], 1)
    bot = np.concatenate(panels[2:], 1)
    img = np.ascontiguousarray(np.concatenate([top, bot], 0))
    s = info.get("ssim")
    txt = f"f={info['frame']}  SSIM={s:.3f}" if s is not None else f"f={info['frame']}  SSIM=n/a (mask<1%)"
    label(img, txt, 1.0 if half else 1.6)
    return img


def cmd_review(args) -> None:
    shots = {s["id"]: s for s in load_shots()["shots"]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in args.shots.split(","):
        shot = shots[sid]
        fr = shot_frames(shot, args)
        ref = decode_ref(fr.start, fr.stop)
        video = out_dir / f"{sid}.mp4"
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-nostdin",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-ss", f"{fr.start / FPS:.6f}", "-t", f"{len(fr) / FPS:.6f}", "-i", str(AUDIO),
             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(video)],
            stdin=subprocess.PIPE,
        )
        rows = []
        for f in fr:
            rgb, alpha = load_render(render_path(args, f))
            info, comp, smap, mask = frame_metric(rgb, alpha, ref[f - fr.start], shot, f, want_maps=True)
            rows.append(info)
            enc.stdin.write(quad(ref[f - fr.start], comp, smap, mask, exclusion_mask(shot, f), info).tobytes())
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg 编码失败")
        # 关键帧：最差、中位、最好（有效帧中），全分辨率四宫格
        valid = [r for r in rows if r["ssim"] is not None]
        picks = []
        if valid:
            srt = sorted(valid, key=lambda r: r["ssim"])
            picks = [("worst", srt[0]), ("median", srt[len(srt) // 2]), ("best", srt[-1])]
        for tag, r in picks:
            f = r["frame"]
            rgb, alpha = load_render(render_path(args, f))
            info, comp, smap, mask = frame_metric(rgb, alpha, ref[f - fr.start], shot, f, want_maps=True)
            img = quad(ref[f - fr.start], comp, smap, mask, exclusion_mask(shot, f), info, half=False)
            Image.fromarray(img).save(out_dir / f"{sid}_{tag}_{f}.png")
        print(f"{sid}: {video}  关键帧 {[(t, r['frame'], round(r['ssim'], 4)) for t, r in picks]}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "review"):
        p = sub.add_parser(name)
        p.add_argument("--renders", required=True, help="渲染帧目录（frame_NNNNNN.png）")
        p.add_argument("--shots", required=True, help="镜头 ID，逗号分隔")
        p.add_argument("--frames", help="只算这一段（含两端），如 3690-3721")
        p.add_argument("--out", help="score：报告 JSON 路径；review：输出目录")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        cmd_score(args)
    else:
        if not args.out:
            ap.error("review 需要 --out")
        cmd_review(args)


if __name__ == "__main__":
    sys.exit(main())
