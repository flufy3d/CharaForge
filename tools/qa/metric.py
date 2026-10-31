"""验收指标（docs/pipeline.md 最后一节的唯一实现）与逐镜头对照视频。

用法（Python 一律 -I 运行）：

  # 1) 指标：渲染帧目录（pnpm render 的输出，frame_NNNNNN.png，非预乘 RGBA）+ 镜头
  tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s43 --shots s43 \
      --out .data/qa/metric_s43.json

  # 2) 对照视频 + 关键帧截图（四宫格：原视频 | 渲染合成到原视频背景 | 差异图 | SSIM 热力图）
  tools/.venv/bin/python -I tools/qa/metric.py review --renders .data/qa/render/s43 --shots s43 \
      --out .data/fit/review

定义（与 pipeline.md 逐条对应）：
  1. M_f = dilate(α_f > 0.1, 半径 3px 圆盘) − 排除区（全分辨率；shots 里覆盖帧 f 的 exclusions 矩形之并，关键帧线性插值）；
  2. 渲染结果按 alpha 合成到参考帧上得到 C_f，参考帧 R_f；
  3. 主指标 SSIM@360：C_f、R_f 用 cv2.resize(INTER_AREA) 降到 640×360，遮罩同样降采样后取 > 0.5；
     RGB 三通道分别算 SSIM 图（skimage.structural_similarity，gaussian_weights=True, sigma=1.5,
     use_sample_covariance=False, data_range=255, K1=0.01, K2=0.03），三通道平均，再在降采样遮罩上取平均；
  4. 辅助指标 SSIM@720：同上但不降采样；
  5. |M_f|（全分辨率）< 1% 画面记为「无角色」，不参与统计；
  6. 覆盖率 = |M_f ∩ bbox| / |bbox − 排除区|，bbox = shots 里该帧各实例 bbox 之并（线性插值，null 表示不可见）；
  7. 镜头指标：两种 SSIM 各自的均值、最小值、5% 分位数、≥0.9 的帧比例；全片 = 所有有效帧的平均；
  8. 每帧分类：普通帧 / 特效帧（落在 halftone、radialBlur、zoomBlur、swirl、silhouette、fadeFromWhite、
     fadeFromBlack、particles 效果区间内；落在 transitionIn 过渡区间内；或任一实例估计 blurSigma > 3），
     输出类别与判定依据；普通帧门槛 SSIM@360 ≥ 0.85，特效帧单独统计、以肉眼对比为准。

参考帧用 ffmpeg 解码，颜色转换参数与 tools/fit/src/refvideo.ts 逐字一致。
--video / --video-offset 可换成别的参考视频（上限实验的合成参考）：视频第 0 帧对应场景帧 offset。
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


def decode_ref(a: int, b: int, video: Path = VIDEO, offset: int = 0) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vf", ffmpeg_vf(a - offset, b - offset),
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


def ssim_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RGB 三通道 SSIM 图的平均（float64 输入，数据范围 0–255）"""
    m = np.zeros(x.shape[:2], np.float64)
    for c in range(3):
        _, s = structural_similarity(
            x[..., c], y[..., c], gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, data_range=255, K1=0.01, K2=0.03, full=True,
        )
        m += s
    return m / 3.0


EFFECT_KINDS = {"halftone", "radialBlur", "zoomBlur", "swirl", "silhouette", "fadeFromWhite", "fadeFromBlack", "particles"}
THRESHOLD_360 = 0.85


def classify(shot: dict, f: int) -> tuple[str, list[str]]:
    """普通帧 / 特效帧（docs/pipeline.md 验收门槛）。返回类别与判定依据列表。"""
    why: list[str] = []
    for e in shot["effects"]:
        if e["kind"] in EFFECT_KINDS and e["start"] <= f < e["end"]:
            why.append(f"effect:{e['kind']}[{e['start']},{e['end']})")
    tr = shot["transitionIn"]
    if tr.get("duration", 0) > 0 and shot["start"] <= f < shot["start"] + tr["duration"]:
        why.append(f"transitionIn:{tr['type']}[{shot['start']},{shot['start'] + tr['duration']})")
    for inst in shot["instances"]:
        ks = [t for t in inst["track"] if t.get("blurSigma") is not None]
        # 首个 blurSigma 采样之前视为未模糊（shots 只在模糊出现后才写 blurSigma）
        if not ks or f < ks[0]["frame"]:
            continue
        sig = _interp_rects([{**k, "b": [k["blurSigma"]]} for k in ks], f, "b")
        if sig is not None and sig[0] > 3:
            why.append(f"blurSigma>3:{inst['id']}({sig[0]:.1f})")
    return ("effect" if why else "normal"), why


def frame_metric(rgb: np.ndarray, alpha: np.ndarray, ref: np.ndarray, shot: dict, f: int, want_maps=False):
    refd = ref.astype(np.float64)
    comp = rgb * alpha[..., None] + refd * (1.0 - alpha[..., None])
    excl = exclusion_mask(shot, f)
    mask = cv2.dilate((alpha > 0.1).astype(np.uint8), DISK) > 0
    mask &= ~excl
    area = int(mask.sum())
    cls, why = classify(shot, f)
    out = {"frame": f, "class": cls, "why": why, "maskArea": area}
    bb = bbox_mask(shot, f)
    denom = int((bb & ~excl).sum())
    out["coverage"] = (float((mask & bb).sum()) / denom) if denom > 0 else None
    if area < MIN_MASK:
        out["ssim360"] = out["ssim720"] = None
        return (out, comp, None, mask) if want_maps else out
    # 主指标 SSIM@360
    c360 = cv2.resize(comp, (640, 360), interpolation=cv2.INTER_AREA)
    r360 = cv2.resize(refd, (640, 360), interpolation=cv2.INTER_AREA)
    m360 = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
    s360 = ssim_map(c360, r360)
    out["ssim360"] = float(s360[m360].mean()) if m360.any() else None
    # 辅助指标 SSIM@720
    s720 = ssim_map(comp, refd)
    out["ssim720"] = float(s720[mask].mean())
    if want_maps:
        # 热力图用 SSIM@360 的图（最近邻放大回全分辨率）
        return out, comp, cv2.resize(s360, (W, H), interpolation=cv2.INTER_NEAREST), mask
    return out


def shot_frames(shot: dict, args) -> range:
    a, b = shot["start"], shot["end"]
    if args.frames:
        lo, hi = (int(x) for x in args.frames.split("-"))
        a, b = max(a, lo), min(b, hi + 1)
    return range(a, b)


def _stats(rows: list[dict], key: str) -> dict:
    vals = np.array([r[key] for r in rows if r[key] is not None], dtype=np.float64)
    if vals.size == 0:
        return {"valid": 0}
    valid = [r for r in rows if r[key] is not None]
    out = {
        "valid": int(vals.size),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "p5": float(np.percentile(vals, 5)),
        "fracGE0.9": float((vals >= 0.9).mean()),
        "worstFrame": int(valid[int(np.argmin(vals))]["frame"]),
    }
    if key == "ssim360":
        out["fracGE0.85"] = float((vals >= THRESHOLD_360).mean())
        out["framesBelow0.85"] = [int(r["frame"]) for r in valid if r[key] < THRESHOLD_360]
    return out


def summarize(rows: list[dict]) -> dict:
    out: dict = {"frames": len(rows)}
    for key in ("ssim360", "ssim720"):
        out[key] = _stats(rows, key)
    # 普通帧 / 特效帧分开统计（门槛只对普通帧：SSIM@360 ≥ 0.85）
    for cls in ("normal", "effect"):
        sub = [r for r in rows if r["class"] == cls]
        out[cls] = {"frames": len(sub), "ssim360": _stats(sub, "ssim360"), "ssim720": _stats(sub, "ssim720")}
    nv = out["normal"]["ssim360"]
    out["normalPass"] = bool(nv.get("valid", 0) > 0 and nv["min"] >= THRESHOLD_360)
    cov = np.array([r["coverage"] for r in rows if r["ssim360"] is not None and r["coverage"] is not None])
    out["coverageMean"] = float(cov.mean()) if cov.size else None
    out["coverageMin"] = float(cov.min()) if cov.size else None
    out["framesWithBboxButNoMask"] = int(sum(1 for r in rows if r["ssim360"] is None and r["coverage"] is not None))
    return out


def render_path(args, f: int) -> Path:
    return Path(args.renders) / f"frame_{f:06d}.png"


def cmd_score(args) -> None:
    shots = {s["id"]: s for s in load_shots()["shots"]}
    report = {"definition": "docs/pipeline.md#验收指标的定义", "renders": str(args.renders), "shots": {}}
    all_vals = []
    for sid in args.shots.split(","):
        shot = shots[sid]
        fr = shot_frames(shot, args)
        ref = decode_ref(fr.start, fr.stop, Path(args.video), args.video_offset)
        rows = []
        for f in fr:
            p = render_path(args, f)
            if not p.exists():
                raise FileNotFoundError(p)
            rgb, alpha = load_render(p)
            rows.append(frame_metric(rgb, alpha, ref[f - fr.start], shot, f))
        summ = summarize(rows)
        report["shots"][sid] = {"summary": summ, "perFrame": rows}
        all_vals += [(r["ssim360"], r["ssim720"]) for r in rows if r["ssim360"] is not None]
        print(f"{sid}: {json.dumps(summ, ensure_ascii=False)}")
    if all_vals:
        a = np.array(all_vals)
        report["overall"] = {"valid": len(all_vals), "ssim360": float(a[:, 0].mean()), "ssim720": float(a[:, 1].mean())}
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
    s = info.get("ssim360")
    tag = "EFFECT" if info.get("class") == "effect" else "normal"
    txt = (f"f={info['frame']} [{tag}]  SSIM@360={s:.3f}  @720={info['ssim720']:.3f}" if s is not None
           else f"f={info['frame']} [{tag}]  SSIM=n/a (mask<1%)")
    label(img, txt, 0.8 if half else 1.3)
    return img


def cmd_review(args) -> None:
    shots = {s["id"]: s for s in load_shots()["shots"]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in args.shots.split(","):
        shot = shots[sid]
        fr = shot_frames(shot, args)
        ref = decode_ref(fr.start, fr.stop, Path(args.video), args.video_offset)
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
        valid = [r for r in rows if r["ssim360"] is not None]
        picks = []
        if valid:
            srt = sorted(valid, key=lambda r: r["ssim360"])
            picks = [("worst", srt[0]), ("median", srt[len(srt) // 2]), ("best", srt[-1])]
        for tag, r in picks:
            f = r["frame"]
            rgb, alpha = load_render(render_path(args, f))
            info, comp, smap, mask = frame_metric(rgb, alpha, ref[f - fr.start], shot, f, want_maps=True)
            img = quad(ref[f - fr.start], comp, smap, mask, exclusion_mask(shot, f), info, half=False)
            Image.fromarray(img).save(out_dir / f"{sid}_{tag}_{f}.png")
        print(f"{sid}: {video}  关键帧 {[(t, r['frame'], round(r['ssim360'], 4)) for t, r in picks]}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "review"):
        p = sub.add_parser(name)
        p.add_argument("--renders", required=True, help="渲染帧目录（frame_NNNNNN.png）")
        p.add_argument("--shots", required=True, help="镜头 ID，逗号分隔")
        p.add_argument("--frames", help="只算这一段（含两端），如 3690-3721")
        p.add_argument("--out", help="score：报告 JSON 路径；review：输出目录")
        p.add_argument("--video", default=str(VIDEO), help="参考视频（缺省为原 PV）")
        p.add_argument("--video-offset", type=int, default=0, help="参考视频第 0 帧对应的场景帧号")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        cmd_score(args)
    else:
        if not args.out:
            ap.error("review 需要 --out")
        cmd_review(args)


if __name__ == "__main__":
    sys.exit(main())
